# Sentinel-XAI — Real-Time Credit Card Fraud Detection with Explainable AI

A high-concurrency fraud detection engine using a **Max-Vote Ensemble** of Random Forest, XGBoost, and LightGBM, with real-time explainability (XAI) via Euclidean distance to fraud centroid, velocity-based rate limiting, and SQLite persistence.

---

## Architecture

```
┌──────────────────────┐     ┌──────────────────────────────┐
│   React Frontend     │────▶│   FastAPI Backend (port 8000) │
│   (Vite + shadcn/ui) │◀────│   /api/v1/predict           │
│   port 5173          │     │   /api/v1/metrics            │
└──────────────────────┘     │   /api/v1/history            │
                             │   /api/v1/session/{id}/timeline
                             └──────────┬───────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
             ┌──────────┐     ┌──────────────┐     ┌──────────────┐
             │ SQLite   │     │ Ensemble     │     │ In-Memory    │
             │ Ledger   │     │ RF+XGB+LGBM  │     │ Velocity     │
             │ (persist)│     │ (joblib pkl) │     │ Cache        │
             └──────────┘     └──────────────┘     └──────────────┘
```

## Project Structure

```
├── backend/
│   ├── app.py                  # FastAPI engine — all API endpoints
│   ├── train_ensemble.py       # Model training pipeline
│   ├── database.py             # SQLAlchemy model + session factory
│   └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main dashboard
│   │   ├── ErrorBoundary.tsx    # React error boundary
│   │   ├── main.tsx             # Entry point
│   │   ├── index.css            # Tailwind styles
│   │   ├── components/ui/       # shadcn/ui primitives
│   │   │   ├── badge.tsx
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   ├── label.tsx
│   │   │   └── select.tsx
│   │   └── lib/
│   │       └── utils.ts
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── index.html
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

## Dataset

**PaySim mobile money simulator** — 6.3M rows of synthetic financial transactions.

- Download from [Kaggle: PaySim Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Place at `C:\AI-proj\dataset-fraud-detection.csv`
- Columns: `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`

## Setup

### Backend

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn sqlalchemy joblib numpy pandas scikit-learn xgboost lightgbm

# Train the ensemble (creates fraud_ensemble.pkl + fraud_centroid.npy + category_map.json)
cd backend
python train_ensemble.py
cd ..

# Start the API server
uvicorn backend.app:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in a browser.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Analyze a single transaction |
| `POST` | `/api/v1/analyze` | Alias for `/predict` |
| `POST` | `/api/v1/batch` | Batch analysis (max 500) |
| `GET` | `/api/v1/history` | Last 50 transactions from SQLite |
| `GET` | `/api/v1/metrics` | Training metrics (ROC, confusion matrix, precision/recall) |
| `GET` | `/api/v1/session/{id}/timeline` | Request timestamps for a session |
| `GET` | `/api/v1/health` | System health + uptime |
| `GET` | `/api/v1/config` | Current allow/block thresholds |
| `POST` | `/api/v1/config` | Update thresholds at runtime |
| `GET` | `/api/v1/feature-importance` | RandomForest feature importance |
| `GET` | `/api/v1/alerts` | Recent blocked/declined transactions |
| `GET` | `/api/v1/export/{uuid}` | JSON export of a transaction report |
| `GET` | `/api/v1/model-agreement` | Per-model vote breakdown |

### POST `/api/v1/predict`

```json
{
  "type": "TRANSFER",
  "amount": 80000,
  "oldbalanceOrg": 80000,
  "oldbalanceDest": 0,
  "session_id": "SESSION-99X"
}
```

## Detection Pipeline

1. **Velocity Firewall** — >3 requests in 10s per `session_id` → instant BLOCK (pre-ML)
2. **Heuristic Override** — `TRANSFER` + `amount == oldbalanceOrg` → `"Account Drain Detected"` → forced REVIEW (unless max vote > BLOCK_THRESHOLD)
3. **Feature Engineering** — 6 features (1 encoded categorical + 5 continuous, z-score scaled)
4. **Max-Vote Ensemble** — `final = max(rf_prob, xgb_prob, lgbm_prob)` instead of averaging
5. **State Routing** — `< 0.35` → ALLOW, `0.35–0.70` → REVIEW, `> 0.70` → BLOCK
6. **XAI** — Euclidean distance to fraud centroid per feature, displayed as "Feature Divergence from Fraud Pattern"
7. **Persistence** — Every request logged to SQLite `transaction_ledger` table

## Training Metrics (on test set: 1.27M samples)

| Model | ROC AUC |
|---|---|
| RandomForest | ~0.998 |
| XGBoost | ~0.999 |
| LightGBM | ~0.999 |
| **Ensemble (Max Vote)** | **~0.9995** |

## Frontend Features

- Real-time transaction analysis with per-feature XAI bar chart
- Radar chart for suspicion scores across features
- Live audit ledger table with decision/type/amount/risk
- Session timeline visualization
- Concurrency stress test (50 parallel requests)
- Batch CSV upload with drag-and-drop
- Threshold configurator with live sliders
- Alert feed for blocked/declined transactions
- Model agreement table (per-model vote breakdown)
- Feature importance chart (RandomForest)
- ML metrics dashboard (confusion matrix, ROC curves, precision/recall/F1)

## XAI: Feature Divergence Interpretation

The horizontal bar chart shows each feature's **Euclidean distance component** from the fraud centroid:

- **Green bars (long)** → Far from fraud pattern → SAFE
- **Yellow bars (medium)** → Moderate risk
- **Red bars (short)** → Close to fraud pattern → RISKY
