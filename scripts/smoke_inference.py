"""Smoke-test the inference helpers over all saved models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.predict import predict_one

client = {
    "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 50000, "BILL_AMT2": 45000, "BILL_AMT3": 40000,
    "BILL_AMT4": 35000, "BILL_AMT5": 30000, "BILL_AMT6": 25000,
    "PAY_AMT1": 5000, "PAY_AMT2": 5000, "PAY_AMT3": 5000,
    "PAY_AMT4": 5000, "PAY_AMT5": 5000, "PAY_AMT6": 5000,
}

print("--- good client ---")
for m in ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]:
    r = predict_one(client, model_name=m)
    print(f"{m:22s} proba={r['default_proba']:.3f} pred={r['prediction']}")

risky = dict(client)
risky.update({"PAY_0": 4, "PAY_2": 3, "PAY_3": 2, "LIMIT_BAL": 50000, "AGE": 30})
print("--- risky client ---")
for m in ["lightgbm", "xgboost"]:
    r = predict_one(risky, model_name=m)
    print(f"{m:22s} proba={r['default_proba']:.3f} pred={r['prediction']}")
