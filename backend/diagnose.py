import pickle
import numpy as np
import pandas as pd

with open("fraud_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
meta = artifact["metadata"]
norm = meta["normalization_stats"]
type_map = meta["type_mapping"]

print("Type mapping:", type_map)
print()


def predict(amount, tx_type, old_orig, new_orig, old_dest, new_dest, step):
    eo = old_orig - new_orig - amount
    ed = old_dest + amount - new_dest
    te = float(type_map.get(tx_type.strip().upper(), -1))
    fv = np.array(
        [
            [
                (amount - norm["amount"]["mean"]) / max(norm["amount"]["std"], 1),
                te,
                (eo - norm["error_balance_orig"]["mean"])
                / max(norm["error_balance_orig"]["std"], 1),
                (ed - norm["error_balance_dest"]["mean"])
                / max(norm["error_balance_dest"]["std"], 1),
                (step - norm["step"]["mean"]) / max(norm["step"]["std"], 1),
            ]
        ],
        dtype=np.float64,
    )
    prob = float(model.predict_proba(fv)[0][1] * 100)
    return prob, eo, ed


# ----- hand-crafted tests -------------------------------------------------
tests = [
    # Normal: error_balance_orig != 0  (small rounding = legitimate)
    (100, "PAYMENT", 5000, 4895, 200, 295, 100),
    (50, "CASH_IN", 200, 250, 5000, 4950, 50),
    (75, "DEBIT", 1000, 920, 500, 575, 150),
    # Fraud: error_balance_orig == 0  (perfect bookkeeping = suspicious)
    (50000, "TRANSFER", 100000, 50000, 0, 50000, 400),
    (1_000_000, "TRANSFER", 0, 0, 0, 1_000_000, 200),
    (200, "CASH_OUT", 1000, 800, 1000, 1200, 300),
]

print("--- Hand-crafted tests ---")
print(
    f"{'Test':<10} {'Type':<12} {'Amount':>10} {'error_orig':>12} {'error_dest':>12} {'Fraud%':>8}  {'Decision'}"
)
print("-" * 80)
for i, (amt, typ, oo, no, od, nd, st) in enumerate(tests):
    prob, eo, ed = predict(amt, typ, oo, no, od, nd, st)
    dec = "BLOCK" if prob > 80 else ("REVIEW" if prob >= 40 else "ALLOW")
    print(
        f"{i:<10} {typ:<12} {amt:>10.2f} {eo:>12.2f} {ed:>12.2f} {prob:>7.2f}%  {dec}"
    )

# ----- test on actual dataset rows ----------------------------------------
print("\n--- Real data test (from test_subset.csv) ---")
raw = pd.read_csv("./data/test_subset.csv")
raw["is_fraud"] = raw["isFraud"]
raw["error_balance_orig"] = raw["oldbalanceOrg"] - raw["newbalanceOrig"] - raw["amount"]
raw["error_balance_dest"] = (
    raw["oldbalanceDest"] + raw["amount"] - raw["newbalanceDest"]
)

fraud_samples = raw[raw["is_fraud"] == 1].sample(n=10, random_state=42)
normal_samples = raw[raw["is_fraud"] == 0].sample(n=5, random_state=42)
real_tests = pd.concat([fraud_samples, normal_samples], ignore_index=True)

print(
    f"{'#':<4} {'True':<6} {'Type':<12} {'Amount':>10} {'error_orig':>12} {'error_dest':>12} {'Fraud%':>8}  {'Pred'}"
)
print("-" * 85)
for i, row in real_tests.iterrows():
    prob, eo, ed = predict(
        row["amount"],
        row["type"],
        row["oldbalanceOrg"],
        row["newbalanceOrig"],
        row["oldbalanceDest"],
        row["newbalanceDest"],
        row["step"],
    )
    dec = "BLOCK" if prob > 80 else ("REVIEW" if prob >= 40 else "ALLOW")
    print(
        f"{i:<4} {int(row['is_fraud']):<6} {row['type']:<12} {row['amount']:>10.2f} {eo:>12.2f} {ed:>12.2f} {prob:>7.2f}%  {dec}"
    )

print("\nDone.")
