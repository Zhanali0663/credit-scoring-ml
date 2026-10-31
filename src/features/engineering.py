"""Feature engineering for credit scoring.

The dataset is the UCI Default of Credit Card Clients dataset with the
following columns (UCI naming convention):

- ``ID``                                  : client id
- ``LIMIT_BAL``                           : amount of given credit (NT$)
- ``SEX``                                 : 1 = male, 2 = female
- ``EDUCATION``                           : 1=grad school, 2=university, 3=high school, 4=others
- ``MARRIAGE``                            : 1=married, 2=single, 3=others
- ``AGE``                                 : years
- ``PAY_0`` ... ``PAY_6``                 : repayment status (-2, -1, 0..9) — most recent first
- ``BILL_AMT1`` ... ``BILL_AMT6``         : amount of bill statement (NT$), most recent first
- ``PAY_AMT1`` ... ``PAY_AMT6``           : amount of previous payment (NT$), most recent first
- ``default_payment_next_month``          : target (1=default, 0=non-default)

The engineered feature set mirrors signals used by professional risk teams:

* Volume aggregates (total / mean / std of bills and payments).
* Affordability & utilisation ratios (per-month and aggregate).
* Delinquency depth: counts, max severity, recency-weighted delay index.
* Trend features (recent vs older behaviour deltas).
* Signed-log transforms of heavy-tailed monetary columns (greatly improve
  logistic-regression and boosting performance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Recency weights: most recent month (PAY_0) is the strongest signal in
# industry scorecards, hence linearly decaying weights towards PAY_6.
RECENCY_WEIGHTS = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

MONETARY_COLS = [
    "LIMIT_BAL",
    *BILL_COLS,
    *PAY_AMT_COLS,
]


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    """Signed log1p: keeps the sign of monetary values, compresses heavy tails."""
    return np.sign(values) * np.log1p(np.abs(values))


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio/aggregate features that summarise billing behaviour.

    The engineered features are intentionally simple, monotonic and easy to
    reason about — a typical requirement in regulated credit risk models.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Volume aggregates
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Signed-log transforms of heavy-tailed monetary columns
    # ------------------------------------------------------------------
    for col in MONETARY_COLS:
        df[f"LOG_{col}"] = _signed_log1p(df[col].to_numpy(dtype=float))

    # ------------------------------------------------------------------
    # Affordability + utilisation ratios
    # ------------------------------------------------------------------
    eps = 1.0
    df["PAY_TO_BILL_RATIO"] = df["PAY_TOTAL"] / (df["BILL_TOTAL"].abs() + eps)
    df["UTILIZATION"] = df["BILL_MEAN"] / (df["LIMIT_BAL"] + eps)
    # Per-month utilisation: mean and max of BILL_AMT_i / LIMIT_BAL
    per_month_util = bill / (limit[:, None] + eps)
    df["UTILIZATION_MEAN"] = per_month_util.mean(axis=1)
    df["UTILIZATION_MAX"] = per_month_util.max(axis=1)
    df["UTILIZATION_STD"] = per_month_util.std(axis=1)

    # Per-month payment coverage ratio: PAY_AMT_i / |BILL_AMT_i|
    per_month_cov = pay_amt / (np.abs(bill) + eps)
    df["COVERAGE_MEAN"] = per_month_cov.mean(axis=1)
    df["COVERAGE_MIN"] = per_month_cov.min(axis=1)

    # ------------------------------------------------------------------
    # Delinquency depth + recency weighting
    # ------------------------------------------------------------------
    df["N_MONTHS_DELAY"] = (pay > 0).sum(axis=1).astype(int)
    df["MAX_DELAY_SEVERITY"] = pay.max(axis=1).astype(int)
    df["N_MONTHS_PAID_DULY"] = (pay <= 0).sum(axis=1).astype(int)
    df["MEAN_DELAY"] = pay.mean(axis=1)
    df["DELAY_STD"] = pay.std(axis=1)

    # Recency-weighted delay index (heavier weight on recent months)
    weighted = (pay * np.array(RECENCY_WEIGHTS)).sum(axis=1)
    df["DELAY_WEIGHTED"] = weighted / sum(RECENCY_WEIGHTS)

    # Longest consecutive "already delayed" streak (PAY_* > 0), most recent first
    delayed = pay > 0
    streak = np.zeros(pay.shape, dtype=int)
    streak[:, 0] = delayed[:, 0].astype(int)
    for col in range(1, pay.shape[1]):
        streak[:, col] = (streak[:, col - 1] + 1) * delayed[:, col]
    df["LONGEST_DELAY_STREAK"] = streak.max(axis=1)

    # ------------------------------------------------------------------
    # Trend features (recent vs older behaviour)
    # ------------------------------------------------------------------
    df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["PAY_TREND"] = df["PAY_AMT1"] - df["PAY_AMT6"]
    df["DELAY_TREND"] = df["PAY_0"] - df["PAY_6"]

    # Replace infinite values (e.g. when total bill == 0)
    df = df.replace([np.inf, -np.inf], 0.0)
    return df


def build_features_target(
    df: pd.DataFrame, target_col: str = "default_payment_next_month", id_col: str = "ID"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into features ``X`` and target ``y``.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe (post-cleaning).
    target_col : str, default ``"default_payment_next_month"``
        Target column name.
    id_col : str, default ``"ID"``
        Identifier column to drop from features.

    Returns
    -------
    X, y : Tuple[pandas.DataFrame, pandas.Series]
        Feature matrix and target vector.
    """
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col, id_col], errors="ignore")
    return X, y


__all__ = ["add_behavioural_features", "build_features_target", "PAY_COLS", "BILL_COLS", "PAY_AMT_COLS"]
