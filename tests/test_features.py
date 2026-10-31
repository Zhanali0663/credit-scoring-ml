"""Tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import (
    BILL_COLS,
    PAY_AMT_COLS,
    PAY_COLS,
    add_behavioural_features,
    build_features_target,
)


def test_add_behavioural_features_does_not_break_shape(sample_dataset: pd.DataFrame) -> None:
    df = add_behavioural_features(sample_dataset)
    expected_new_cols = {
        "BILL_TOTAL",
        "PAY_TOTAL",
        "BILL_MEAN",
        "PAY_MEAN",
        "PAY_TO_BILL_RATIO",
        "UTILIZATION",
        "N_MONTHS_DELAY",
        "MAX_DELAY_SEVERITY",
        "N_MONTHS_PAID_DULY",
        "BILL_TREND",
        "PAY_TREND",
        "DELAY_TREND",
    }
    assert expected_new_cols.issubset(df.columns)
    assert df.shape[0] == sample_dataset.shape[0]
    assert np.isfinite(df.select_dtypes("number").values).all()


def test_build_features_target_drops_id_and_target(sample_dataset: pd.DataFrame) -> None:
    X, y = build_features_target(sample_dataset)
    assert "ID" not in X.columns
    assert "default_payment_next_month" not in X.columns
    assert len(X) == len(y)
    assert y.dtype.kind == "i"


def test_engineered_features_use_correct_source_columns() -> None:
    """Confirm we use the documented column lists."""
    assert "PAY_0" in PAY_COLS
    assert "BILL_AMT1" in BILL_COLS
    assert "PAY_AMT6" in PAY_AMT_COLS
