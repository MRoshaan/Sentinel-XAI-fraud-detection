import json
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATASET_PATH = "../dataset-fraud-detection.csv"
OUTPUT_DIR = "./"
ENSEMBLE_PATH = os.path.join(OUTPUT_DIR, "fraud_ensemble.pkl")
CATEGORY_MAP_PATH = os.path.join(OUTPUT_DIR, "category_map.json")
RANDOM_SEED = 42

LABEL_COL = "isFraud"
FEATURES = [
    "type_encoded",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
CONTINUOUS_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def load_dataset() -> pd.DataFrame:
    path = DATASET_PATH if os.path.exists(DATASET_PATH) else "./data/dataset.csv"
    raw = pd.read_csv(path)
    required = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        LABEL_COL,
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"Loaded {len(raw):,} rows from {path}")
    for col in CONTINUOUS_FEATURES:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
    raw[LABEL_COL] = (
        pd.to_numeric(raw[LABEL_COL], errors="coerce").fillna(0).astype(int)
    )
    return raw[required].copy()


def build_type_map(series: pd.Series) -> dict:
    mapping = {}
    for val in series.fillna("UNKNOWN").astype(str):
        key = val.strip().upper() if val.strip() else "UNKNOWN"
        mapping.setdefault(key, len(mapping))
    return mapping


def manual_balance(df: pd.DataFrame) -> pd.DataFrame:
    majority = df[df[LABEL_COL] == 0]
    minority = df[df[LABEL_COL] == 1]
    if len(majority) > 100_000:
        majority = majority.sample(n=100_000, random_state=RANDOM_SEED)
    target = len(majority)
    n_min = len(minority)
    if n_min >= target:
        return pd.concat([majority, minority.iloc[:target]], ignore_index=True)
    reps = target // n_min
    rem = target % n_min
    parts = [minority.copy() for _ in range(reps)]
    if rem > 0:
        parts.append(minority.iloc[:rem].copy())
    balanced = pd.concat([majority] + parts, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def compute_scaler_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in CONTINUOUS_FEATURES:
        m = float(df[col].mean())
        s = float(df[col].std(ddof=0))
        stats[col] = {"mean": m, "std": s if s != 0 else 1.0}
    return stats


def apply_scale(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    scaled = df.copy()
    for col in CONTINUOUS_FEATURES:
        scaled[col] = (scaled[col] - stats[col]["mean"]) / stats[col]["std"]
    return scaled


def main():
    df = load_dataset()
    category_map = build_type_map(df["type"])
    df["type_encoded"] = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(category_map)
        .fillna(-1)
        .astype(int)
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df[LABEL_COL]
    )

    scaler_stats = compute_scaler_stats(train_df)
    train_scaled = apply_scale(train_df, scaler_stats)
    test_scaled = apply_scale(test_df, scaler_stats)

    balanced = manual_balance(train_scaled)
    X_train = balanced[FEATURES].values.astype(np.float64)
    y_train = balanced[LABEL_COL].values
    X_test = test_scaled[FEATURES].values.astype(np.float64)
    y_test = test_scaled[LABEL_COL].values
    print(f"Balanced train: {len(balanced):,} | Test: {len(test_df):,}")

    print("\n--- Training RandomForest ---")
    rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=14,
        min_samples_leaf=4,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_importance = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURES, rf.feature_importances_)
    }

    print("--- Training XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=50,
        max_depth=8,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)

    print("--- Training LightGBM ---")
    lgbm = LGBMClassifier(
        n_estimators=50,
        max_depth=8,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)

    fraud_mask = train_scaled[train_scaled[LABEL_COL] == 1]
    fraud_centroid = (
        fraud_mask[CONTINUOUS_FEATURES].mean(axis=0).to_numpy(dtype=np.float64)
    )
    np.save(os.path.join(OUTPUT_DIR, "fraud_centroid.npy"), fraud_centroid)
    print(f"Fraud centroid shape: {fraud_centroid.shape}")

    with open(CATEGORY_MAP_PATH, "w") as f:
        json.dump(category_map, f, indent=2)
    print(f"Category map saved to {CATEGORY_MAP_PATH}")

    artifact = {
        "models": {"rf": rf, "xgb": xgb, "lgbm": lgbm},
        "feature_order": FEATURES,
        "continuous_feature_order": CONTINUOUS_FEATURES,
        "scaler_stats": scaler_stats,
        "category_map": category_map,
        "feature_importance": rf_importance,
    }
    with open(ENSEMBLE_PATH, "wb") as f:
        joblib.dump(artifact, f)
    print(f"Ensemble artifact saved to {ENSEMBLE_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
