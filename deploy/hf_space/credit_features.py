"""Feature engineering for the Hugging Face Space demo.

Kept in sync with ``src/features/engineering.py`` of the parent project
(credit-scoring-ml). Same recency weighting, same signed-log transforms and
ratios, so the Space can reconstruct the full model input purely from the
raw borrower fields the user provides in the form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

RECENCY_WEIGHTS = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

MONETARY_COLS = ["LIMIT_BAL", *BILL_COLS, *PAY_AMT_COLS]


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand raw borrower fields into the full 61-column model input."""
    df = df.copy()

    bill = df[BILL_COLS].to_numpy(dtype=float)
    pay = df[PAY_COLS].to_numpy(dtype=float)
    pay_amt = df[PAY_AMT_COLS].to_numpy(dtype=float)
    limit = df["LIMIT_BAL"].to_numpy(dtype=float)

    df["BILL_TOTAL"] = bill.sum(axis=1)
    df["PAY_TOTAL"] = pay_amt.sum(axis=1)
    df["BILL_MEAN"] = bill.mean(axis=1)
    df["PAY_MEAN"] = pay_amt.mean(axis=1)
    df["BILL_STD"] = bill.std(axis=1)
    df["PAY_STD"] = pay_amt.std(axis=1)
    df["BILL_MAX"] = bill.max(axis=1)
    df["PAY_MAX"] = pay_amt.max(axis=1)

    for col in MONETARY_COLS:
        df[f"LOG_{col}"] = _signed_log1p(df[col].to_numpy(dtype=float))

    eps = 1.0
    df["PAY_TO_BILL_RATIO"] = df["PAY_TOTAL"] / (df["BILL_TOTAL"].abs() + eps)
    df["UTILIZATION"] = df["BILL_MEAN"] / (df["LIMIT_BAL"] + eps)
    per_month_util = bill / (limit[:, None] + eps)
    df["UTILIZATION_MEAN"] = per_month_util.mean(axis=1)
    df["UTILIZATION_MAX"] = per_month_util.max(axis=1)
    df["UTILIZATION_STD"] = per_month_util.std(axis=1)
    per_month_cov = pay_amt / (np.abs(bill) + eps)
    df["COVERAGE_MEAN"] = per_month_cov.mean(axis=1)
    df["COVERAGE_MIN"] = per_month_cov.min(axis=1)

    df["N_MONTHS_DELAY"] = (pay > 0).sum(axis=1).astype(int)
    df["MAX_DELAY_SEVERITY"] = pay.max(axis=1).astype(int)
    df["N_MONTHS_PAID_DULY"] = (pay <= 0).sum(axis=1).astype(int)
    df["MEAN_DELAY"] = pay.mean(axis=1)
    df["DELAY_STD"] = pay.std(axis=1)

    weighted = (pay * np.array(RECENCY_WEIGHTS)).sum(axis=1)
    df["DELAY_WEIGHTED"] = weighted / sum(RECENCY_WEIGHTS)

    delayed = pay > 0
    streak = np.zeros(pay.shape, dtype=int)
    streak[:, 0] = delayed[:, 0].astype(int)
    for col in range(1, pay.shape[1]):
        streak[:, col] = (streak[:, col - 1] + 1) * delayed[:, col]
    df["LONGEST_DELAY_STREAK"] = streak.max(axis=1)

    df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["PAY_TREND"] = df["PAY_AMT1"] - df["PAY_AMT6"]
    df["DELAY_TREND"] = df["PAY_0"] - df["PAY_6"]

    df = df.replace([np.inf, -np.inf], 0.0)
    return df


def build_features_target(
    df: pd.DataFrame, target_col: str = "default_payment_next_month", id_col: str = "ID"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a df into features and target (same behaviour as the project)."""
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col, id_col], errors="ignore")
    return X, y


__all__ = ["add_behavioural_features", "build_features_target"]
