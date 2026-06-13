import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


DATASET_PATH = "./data/dataset.csv"
MODEL_PATH = "./fraud_engine.pkl"
CATEGORY_MAP_PATH = "./category_map.json"
FRAUD_CENTROID_PATH = "./fraud_centroid.npy"
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
    dataset_path = (
        DATASET_PATH
        if os.path.exists(DATASET_PATH)
        else "./dataset-fraud-detection.csv"
    )
    raw = pd.read_csv(dataset_path)

    required = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        LABEL_COL,
    ]

    if LABEL_COL not in raw.columns and "is_fraud" in raw.columns:
        raw[LABEL_COL] = raw["is_fraud"]

    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw[required].copy()
    print(f"Loaded {len(df):,} rows from {dataset_path}")
    return df


def build_type_map(series: pd.Series) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for value in series.fillna("UNKNOWN").astype(str):
        key = value.strip().upper() if value.strip() else "UNKNOWN"
        if key not in mapping:
            mapping[key] = len(mapping)
    return mapping


def manual_balance(df: pd.DataFrame) -> pd.DataFrame:
    majority = df[df[LABEL_COL] == 0]
    minority = df[df[LABEL_COL] == 1]

    if majority.empty or minority.empty:
        raise ValueError("Both classes are required for balancing")

    target = len(majority)
    minority_count = len(minority)

    if minority_count >= target:
        return pd.concat([majority, minority.iloc[:target]], ignore_index=True)

    full_repeats = target // minority_count
    remainder = target % minority_count

    duplicated_parts: List[pd.DataFrame] = [
        minority.copy() for _ in range(full_repeats)
    ]
    if remainder > 0:
        duplicated_parts.append(minority.iloc[:remainder].copy())

    upsampled_minority = pd.concat(duplicated_parts, ignore_index=True)
    balanced = pd.concat([majority, upsampled_minority], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def compute_scaler_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in CONTINUOUS_FEATURES:
        mean_val = float(train_df[col].mean())
        std_val = float(train_df[col].std(ddof=0))
        if std_val == 0.0:
            std_val = 1.0
        stats[col] = {"mean": mean_val, "std": std_val}
    return stats


def apply_scaling(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    scaled = df.copy()
    for col in CONTINUOUS_FEATURES:
        scaled[col] = (scaled[col] - stats[col]["mean"]) / stats[col]["std"]
    return scaled


def train() -> None:
    df = load_dataset()

    for col in CONTINUOUS_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

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
    train_scaled = apply_scaling(train_df, scaler_stats)
    test_scaled = apply_scaling(test_df, scaler_stats)

    maj = train_scaled[train_scaled[LABEL_COL] == 0]
    mino = train_scaled[train_scaled[LABEL_COL] == 1]
    if len(maj) > 100_000:
        maj = maj.sample(n=100_000, random_state=RANDOM_SEED)
    combined = pd.concat([maj, mino], ignore_index=True)

    balanced_train = manual_balance(combined)

    X_train = balanced_train[FEATURES].values.astype(np.float64)
    y_train = balanced_train[LABEL_COL].values

    X_test = test_scaled[FEATURES].values.astype(np.float64)
    y_test = test_scaled[LABEL_COL].values

    print(f"Fraud rate before balance: {train_scaled[LABEL_COL].mean():.4%}")
    print(f"Balanced training size: {len(balanced_train):,}")

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_depth=14,
        min_samples_leaf=4,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\nHeld-out evaluation")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    fraud_train = train_scaled[train_scaled[LABEL_COL] == 1]
    fraud_centroid = (
        fraud_train[CONTINUOUS_FEATURES].mean(axis=0).to_numpy(dtype=np.float64)
    )
    np.save(FRAUD_CENTROID_PATH, fraud_centroid)
    print(f"Saved fraud centroid to {FRAUD_CENTROID_PATH}")

    with open(CATEGORY_MAP_PATH, "w", encoding="utf-8") as fp:
        json.dump(category_map, fp, indent=2)
    print(f"Saved category map to {CATEGORY_MAP_PATH}")

    artifact: Dict[str, Any] = {
        "model": model,
        "feature_order": FEATURES,
        "continuous_feature_order": CONTINUOUS_FEATURES,
        "scaler_stats": scaler_stats,
    }
    with open(MODEL_PATH, "wb") as fp:
        pickle.dump(artifact, fp)
    print(f"Saved model artifact to {MODEL_PATH}")


if __name__ == "__main__":
    train()
