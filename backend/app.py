import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import TransactionLedger, get_db, init_db

ALLOW_THRESHOLD = 0.35
BLOCK_THRESHOLD = 0.70
VELOCITY_WINDOW = 10
VELOCITY_LIMIT = 3
WHITELIST_TYPES = frozenset({"CASH_IN", "PAYMENT", "DEBIT"})

velocity_cache: Dict[str, List[float]] = {}
session_timestamps: Dict[str, List[float]] = {}
TRAINING_METRICS: Dict = {}
start_time = time.time()
total_requests = 0

ENSEMBLE_PATH = "backend/fraud_ensemble.pkl"
CATEGORY_MAP_PATH = "backend/category_map.json"
FRAUD_CENTROID_PATH = "backend/fraud_centroid.npy"

MODELS = None
CATEGORY_MAP: Dict[str, int] = {}
FRAUD_CENTROID: np.ndarray | None = None
SCALER_STATS: Dict[str, Dict[str, float]] = {}
FEATURE_ORDER: List[str] = []
CONTINUOUS_FEATURE_ORDER: List[str] = []


class TransactionRequest(BaseModel):
    type: str
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    session_id: str = "SESSION-99X"


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class ConfigUpdate(BaseModel):
    allow_threshold: float = Field(..., ge=0.0, le=1.0)
    block_threshold: float = Field(..., ge=0.0, le=1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS, CATEGORY_MAP, FRAUD_CENTROID, SCALER_STATS
    global FEATURE_ORDER, CONTINUOUS_FEATURE_ORDER

    init_db()

    try:
        with open(ENSEMBLE_PATH, "rb") as fp:
            artifact = joblib.load(fp)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Ensemble not found at {ENSEMBLE_PATH}. Run train_ensemble.py first."
        ) from exc

    try:
        with open(CATEGORY_MAP_PATH, "r") as fp:
            CATEGORY_MAP = json.load(fp)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Category map not found at {CATEGORY_MAP_PATH}") from exc

    try:
        FRAUD_CENTROID = np.load(FRAUD_CENTROID_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Fraud centroid not found at {FRAUD_CENTROID_PATH}"
        ) from exc

    MODELS = artifact["models"]
    SCALER_STATS = artifact["scaler_stats"]
    FEATURE_ORDER = artifact["feature_order"]
    CONTINUOUS_FEATURE_ORDER = artifact["continuous_feature_order"]
    TRAINING_METRICS.update(artifact.get("training_metrics", {}))
    print(f"Loaded ensemble: RF + XGB + LGBM")
    yield


app = FastAPI(title="Explainable Fraud Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def scale_cont(value: float, col: str) -> float:
    s = SCALER_STATS[col]
    return (value - s["mean"]) / (s["std"] if s["std"] != 0 else 1.0)


def check_velocity(session_id: str) -> bool:
    now = time.time()
    if session_id not in velocity_cache:
        velocity_cache[session_id] = []
    velocity_cache[session_id] = [
        t for t in velocity_cache[session_id] if now - t < VELOCITY_WINDOW
    ]
    if len(velocity_cache[session_id]) >= VELOCITY_LIMIT:
        return True
    velocity_cache[session_id].append(now)
    return False


def record_session_ts(session_id: str):
    now = time.time()
    if session_id not in session_timestamps:
        session_timestamps[session_id] = []
    session_timestamps[session_id].append(now)
    session_timestamps[session_id] = [
        t for t in session_timestamps[session_id] if now - t < 60
    ]


# ----------------------------------------------------------------
# PREDICTION ENGINE
# ----------------------------------------------------------------


def run_predict(req: TransactionRequest) -> dict:
    global total_requests
    total_requests += 1
    txn_uuid = str(uuid.uuid4())

    newbalance_orig = req.oldbalanceOrg - req.amount
    newbalance_dest = req.oldbalanceDest + req.amount
    type_upper = req.type.strip().upper()

    # --- 1. Velocity check (pre-ML) ---
    if check_velocity(req.session_id):
        entry = TransactionLedger(
            txn_uuid=txn_uuid,
            session_id=req.session_id,
            timestamp=datetime.now(timezone.utc),
            type=req.type,
            amount=req.amount,
            oldbalanceOrg=req.oldbalanceOrg,
            newbalanceOrig=req.oldbalanceOrg,
            oldbalanceDest=req.oldbalanceDest,
            newbalanceDest=req.oldbalanceDest,
            risk_score=100.0,
            decision="BLOCK",
            heuristic_triggered=None,
            model_votes=json.dumps({"rf": 100.0, "xgb": 100.0, "lgbm": 100.0}),
        )
        return _build_block_response(txn_uuid, req, "VELOCITY_EXCEEDED", entry)

    # --- 2. Whitelist Heuristic (zero-fraud types, skip ML entirely) ---
    if type_upper in WHITELIST_TYPES:
        entry = TransactionLedger(
            txn_uuid=txn_uuid,
            session_id=req.session_id,
            timestamp=datetime.now(timezone.utc),
            type=req.type,
            amount=req.amount,
            oldbalanceOrg=req.oldbalanceOrg,
            newbalanceOrig=newbalance_orig,
            oldbalanceDest=req.oldbalanceDest,
            newbalanceDest=newbalance_dest,
            risk_score=0.0,
            decision="ALLOW",
            heuristic_triggered="System Default: Low-Risk Transaction Type",
            model_votes=json.dumps({"rf": 0.0, "xgb": 0.0, "lgbm": 0.0}),
        )
        return _build_whitelist_response(txn_uuid, req, entry)

    # --- 3. Deterministic Heuristic ---
    heuristic_triggered = None
    if (
        type_upper in ("TRANSFER", "CASH_OUT")
        and abs(req.amount - req.oldbalanceOrg) < 0.001
    ):
        heuristic_triggered = "Account Drain Detected"

    # --- 4. Feature engineering ---
    type_encoded = float(CATEGORY_MAP.get(type_upper, -1))

    raw_cont = {
        "amount": req.amount,
        "oldbalanceOrg": req.oldbalanceOrg,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": req.oldbalanceDest,
        "newbalanceDest": newbalance_dest,
    }
    scaled_cont = {
        col: scale_cont(float(raw_cont[col]), col) for col in CONTINUOUS_FEATURE_ORDER
    }

    feature_values = {"type_encoded": type_encoded, **scaled_cont}
    row = np.array([[feature_values[col] for col in FEATURE_ORDER]], dtype=np.float64)

    # --- 5. Ensemble prediction (max voting) ---
    rf_prob = float(MODELS["rf"].predict_proba(row)[0][1])
    xgb_prob = float(MODELS["xgb"].predict_proba(row)[0][1])
    lgbm_prob = float(MODELS["lgbm"].predict_proba(row)[0][1])
    max_vote = max(rf_prob, xgb_prob, lgbm_prob)
    confidence_score = round(max_vote * 100, 2)

    model_votes = {
        "rf": round(rf_prob * 100, 2),
        "xgb": round(xgb_prob * 100, 2),
        "lgbm": round(lgbm_prob * 100, 2),
    }

    # --- 6. State routing with heuristic override ---
    if heuristic_triggered == "Account Drain Detected":
        if max_vote > BLOCK_THRESHOLD:
            decision = "BLOCK"
        else:
            decision = "REVIEW"
    elif max_vote > BLOCK_THRESHOLD:
        decision = "BLOCK"
    elif max_vote >= ALLOW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    # --- 7. XAI (isolated from routing) ---
    cont_vec = np.array(
        [scaled_cont[col] for col in CONTINUOUS_FEATURE_ORDER], dtype=np.float64
    )
    distance = float(np.linalg.norm(cont_vec - FRAUD_CENTROID))

    contributions = []
    for idx, col in enumerate(CONTINUOUS_FEATURE_ORDER):
        delta = abs(float(cont_vec[idx] - FRAUD_CENTROID[idx]))
        suspicion = float(max(0.0, 1.0 - min(delta / 3.0, 1.0)) * 100.0)
        contributions.append(
            {
                "feature": col,
                "suspicion_score": suspicion,
                "distance_component": delta,
                "value": float(cont_vec[idx]),
                "fraud_centroid_value": float(FRAUD_CENTROID[idx]),
            }
        )
    contributions.sort(key=lambda x: x["suspicion_score"], reverse=True)

    # --- 8. Session tracking + Persist ---
    record_session_ts(req.session_id)
    entry = TransactionLedger(
        txn_uuid=txn_uuid,
        session_id=req.session_id,
        timestamp=datetime.now(timezone.utc),
        type=req.type,
        amount=req.amount,
        oldbalanceOrg=req.oldbalanceOrg,
        newbalanceOrig=newbalance_orig,
        oldbalanceDest=req.oldbalanceDest,
        newbalanceDest=newbalance_dest,
        risk_score=confidence_score,
        decision=decision,
        heuristic_triggered=heuristic_triggered,
        model_votes=json.dumps(model_votes),
    )
    return _build_response(
        txn_uuid,
        req,
        decision,
        confidence_score,
        distance,
        contributions,
        model_votes,
        newbalance_orig,
        newbalance_dest,
        heuristic_triggered or "System Default: No heuristic rule triggered.",
        entry,
    )


def _build_block_response(
    txn_uuid: str, req: TransactionRequest, reason: str, entry: TransactionLedger
) -> dict:
    from backend.database import SessionLocal

    db = SessionLocal()
    db.add(entry)
    db.commit()
    db.close()
    return {
        "status": "BLOCK",
        "reason": reason,
        "confidence_score": 100.0,
        "euclidean_distance_to_fraud": 0.0,
        "feature_contributions": [],
        "top_suspicious_feature": None,
        "newbalanceOrig": req.oldbalanceOrg,
        "newbalanceDest": req.oldbalanceDest,
        "type": req.type,
        "amount": req.amount,
        "oldbalanceOrg": req.oldbalanceOrg,
        "session_id": req.session_id,
        "txn_uuid": txn_uuid,
        "model_votes": {"rf": 100.0, "xgb": 100.0, "lgbm": 100.0},
        "heuristic_triggered": "System Default: Rate limit / velocity threshold exceeded.",
    }


def _build_whitelist_response(
    txn_uuid: str, req: TransactionRequest, entry: TransactionLedger
) -> dict:
    from backend.database import SessionLocal

    db = SessionLocal()
    db.add(entry)
    db.commit()
    db.close()
    return {
        "status": "ALLOW",
        "confidence_score": 0.0,
        "euclidean_distance_to_fraud": 0.0,
        "feature_contributions": [],
        "top_suspicious_feature": None,
        "newbalanceOrig": req.oldbalanceOrg - req.amount,
        "newbalanceDest": req.oldbalanceDest + req.amount,
        "type": req.type,
        "amount": req.amount,
        "oldbalanceOrg": req.oldbalanceOrg,
        "session_id": req.session_id,
        "txn_uuid": txn_uuid,
        "model_votes": {"rf": 0.0, "xgb": 0.0, "lgbm": 0.0},
        "heuristic_triggered": "System Default: Low-Risk Transaction Type",
    }


def _build_response(
    txn_uuid,
    req,
    decision,
    confidence_score,
    distance,
    contributions,
    model_votes,
    newbalance_orig,
    newbalance_dest,
    heuristic_triggered,
    entry,
) -> dict:
    from backend.database import SessionLocal

    db = SessionLocal()
    db.add(entry)
    db.commit()
    db.close()
    return {
        "status": decision,
        "confidence_score": confidence_score,
        "euclidean_distance_to_fraud": distance,
        "feature_contributions": contributions,
        "top_suspicious_feature": contributions[0]["feature"]
        if contributions
        else None,
        "newbalanceOrig": newbalance_orig,
        "newbalanceDest": newbalance_dest,
        "type": req.type,
        "amount": req.amount,
        "oldbalanceOrg": req.oldbalanceOrg,
        "session_id": req.session_id,
        "txn_uuid": txn_uuid,
        "model_votes": model_votes,
        "heuristic_triggered": heuristic_triggered,
    }


# ----------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------


@app.post("/api/v1/predict")
async def predict(req: TransactionRequest):
    if MODELS is None or FRAUD_CENTROID is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return run_predict(req)


@app.post("/api/v1/analyze")
async def analyze(req: TransactionRequest):
    return await predict(req)


@app.get("/api/v1/history")
async def get_history(limit: int = 50, db: Session = Depends(get_db)):
    rows = (
        db.query(TransactionLedger)
        .order_by(TransactionLedger.id.desc())
        .limit(limit)
        .all()
    )
    items = []
    for r in rows:
        votes = {}
        if r.model_votes:
            try:
                votes = json.loads(r.model_votes)
            except json.JSONDecodeError:
                pass
        items.append(
            {
                "id": r.id,
                "txn_uuid": r.txn_uuid,
                "session_id": r.session_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                "type": r.type,
                "amount": r.amount,
                "oldbalanceOrg": r.oldbalanceOrg,
                "newbalanceOrig": r.newbalanceOrig,
                "oldbalanceDest": r.oldbalanceDest,
                "newbalanceDest": r.newbalanceDest,
                "risk_score": r.risk_score,
                "decision": r.decision,
                "heuristic_triggered": r.heuristic_triggered,
                "model_votes": votes,
            }
        )
    return {"count": len(items), "items": items}


@app.get("/api/v1/config")
async def get_config():
    return {
        "allow_threshold": ALLOW_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
        "velocity_window": VELOCITY_WINDOW,
        "velocity_limit": VELOCITY_LIMIT,
    }


@app.post("/api/v1/config")
async def update_config(cfg: ConfigUpdate):
    global ALLOW_THRESHOLD, BLOCK_THRESHOLD
    if cfg.allow_threshold >= cfg.block_threshold:
        raise HTTPException(
            status_code=400, detail="allow_threshold must be < block_threshold"
        )
    ALLOW_THRESHOLD = cfg.allow_threshold
    BLOCK_THRESHOLD = cfg.block_threshold
    return {
        "status": "updated",
        "allow_threshold": ALLOW_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
    }


@app.post("/api/v1/batch")
async def batch_analyze(req: BatchRequest):
    if len(req.transactions) > 500:
        raise HTTPException(
            status_code=400, detail="Maximum 500 transactions per batch"
        )
    results = [run_predict(txn) for txn in req.transactions]
    summary: Dict[str, int] = {"ALLOW": 0, "BLOCK": 0, "REVIEW": 0, "DECLINED": 0}
    total_conf = 0.0
    for r in results:
        s = r["status"]
        if s in summary:
            summary[s] += 1
        total_conf += r["confidence_score"]
    return {
        "total": len(results),
        "summary": summary,
        "avg_confidence": round(total_conf / max(len(results), 1), 2),
        "results": results,
    }


@app.get("/api/v1/health")
async def get_health(db: Session = Depends(get_db)):
    uptime_secs = time.time() - start_time
    hours = int(uptime_secs // 3600)
    minutes = int((uptime_secs % 3600) // 60)
    txn_count = db.query(TransactionLedger).count()
    return {
        "status": "operational",
        "uptime": f"{hours}h {minutes}m",
        "uptime_seconds": round(uptime_secs, 1),
        "models_loaded": list(MODELS.keys()) if MODELS else [],
        "model_count": len(MODELS) if MODELS else 0,
        "database": "connected",
        "total_requests": total_requests,
        "total_transactions_db": txn_count,
        "active_sessions": len(velocity_cache),
    }


@app.get("/api/v1/feature-importance")
async def get_feature_importance():
    with open(ENSEMBLE_PATH, "rb") as fp:
        artifact = joblib.load(fp)
    importance = artifact.get("feature_importance", {})
    items = [{"feature": k, "importance": v} for k, v in importance.items()]
    items.sort(key=lambda x: x["importance"], reverse=True)
    return {"features": items}


@app.get("/api/v1/alerts")
async def get_alerts(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(TransactionLedger)
        .filter(TransactionLedger.decision.in_(["BLOCK", "DECLINED"]))
        .order_by(TransactionLedger.id.desc())
        .limit(limit)
        .all()
    )
    items = []
    for r in rows:
        alert_type = (
            "VELOCITY"
            if r.risk_score == 100.0 and r.decision == "BLOCK"
            else r.decision
        )
        items.append(
            {
                "txn_uuid": r.txn_uuid,
                "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                "type": r.type,
                "amount": r.amount,
                "decision": r.decision,
                "alert_type": alert_type,
                "session_id": r.session_id,
                "risk_score": r.risk_score,
            }
        )
    return {"count": len(items), "items": items}


@app.get("/api/v1/export/{txn_uuid}")
async def export_report(txn_uuid: str, db: Session = Depends(get_db)):
    row = (
        db.query(TransactionLedger)
        .filter(TransactionLedger.txn_uuid == txn_uuid)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Transaction not found")
    votes = {}
    if row.model_votes:
        try:
            votes = json.loads(row.model_votes)
        except json.JSONDecodeError:
            pass
    return {
        "report_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transaction": {
            "uuid": row.txn_uuid,
            "session_id": row.session_id,
            "timestamp": row.timestamp.isoformat() if row.timestamp else "",
            "type": row.type,
            "amount": row.amount,
            "oldbalanceOrg": row.oldbalanceOrg,
            "newbalanceOrig": row.newbalanceOrig,
        },
        "analysis": {
            "decision": row.decision,
            "risk_score": row.risk_score,
            "model_votes": votes,
            "heuristic_triggered": row.heuristic_triggered,
        },
        "engine": {
            "ensemble": "RF + XGBoost + LightGBM (Max Voting)",
            "allow_threshold": ALLOW_THRESHOLD,
            "block_threshold": BLOCK_THRESHOLD,
        },
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    return {"training_metrics": TRAINING_METRICS}


@app.get("/api/v1/session/{session_id}/timeline")
async def get_session_timeline(session_id: str):
    now = time.time()
    pts = session_timestamps.get(session_id, [])
    points = [{"offset": round(now - ts, 2), "ts": ts} for ts in pts]
    return {"session_id": session_id, "points": points}


@app.get("/api/v1/model-agreement")
async def get_model_agreement(limit: int = 10, db: Session = Depends(get_db)):
    rows = (
        db.query(TransactionLedger)
        .filter(TransactionLedger.model_votes.isnot(None))
        .filter(TransactionLedger.decision.in_(["ALLOW", "REVIEW", "BLOCK"]))
        .order_by(TransactionLedger.id.desc())
        .limit(limit)
        .all()
    )
    items = []
    for r in rows:
        votes = {}
        if r.model_votes:
            try:
                votes = json.loads(r.model_votes)
            except json.JSONDecodeError:
                pass
        if votes:
            items.append(
                {
                    "txn_uuid": r.txn_uuid,
                    "type": r.type,
                    "amount": r.amount,
                    "decision": r.decision,
                    "rf": votes.get("rf", 0),
                    "xgb": votes.get("xgb", 0),
                    "lgbm": votes.get("lgbm", 0),
                }
            )
    return {"count": len(items), "items": items}
