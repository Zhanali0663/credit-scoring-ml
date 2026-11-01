"""Property-based invariants that guard against leakage and broken features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split


def _single_client(**overrides: float) -> pd.DataFrame:
    base = {
        "LIMIT_BAL": 200_000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 50_000,
        "BILL_AMT2": 45_000,
        "BILL_AMT3": 40_000,
        "BILL_AMT4": 35_000,
        "BILL_AMT5": 30_000,
        "BILL_AMT6": 25_000,
        "PAY_AMT1": 5_000,
        "PAY_AMT2": 5_000,
        "PAY_AMT3": 5_000,
        "PAY_AMT4": 5_000,
        "PAY_AMT5": 5_000,
        "PAY_AMT6": 5_000,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_higher_limit_lowers_utilisation() -> None:
    """With identical bills and payments, more credit cannot raise utilisation."""
    low = add_behavioural_features(_single_client(LIMIT_BAL=200_000))
    high = add_behavioural_features(_single_client(LIMIT_BAL=400_000))
    assert high["UTILIZATION"].iloc[0] < low["UTILIZATION"].iloc[0]
    assert high["UTILIZATION_MEAN"].iloc[0] < low["UTILIZATION_MEAN"].iloc[0]


def test_engineered_features_do_not_depend_on_target(sample_dataset: pd.DataFrame) -> None:
    """Flipping the target must leave every engineered feature unchanged."""
    first = add_behavioural_features(sample_dataset)
    flipped = sample_dataset.copy()
    flipped["default_payment_next_month"] = 1 - flipped["default_payment_next_month"]
    second = add_behavioural_features(flipped)

    engineered = [c for c in first.columns if c not in sample_dataset.columns]
    pd.testing.assert_frame_equal(first[engineered], second[engineered])


def test_feature_matrix_excludes_id_and_target(sample_dataset: pd.DataFrame) -> None:
    X, y = build_features_target(add_behavioural_features(sample_dataset))
    assert "ID" not in X.columns
    assert "default_payment_next_month" not in X.columns
    assert not any("default" in column.lower() for column in X.columns)
    assert len(X) == len(y)
    assert y.dtype.kind == "i"


def test_engineered_features_are_row_local() -> None:
    """Changing one borrower must not change another borrower's features."""
    first = _single_client()
    second = _single_client()
    combined = pd.concat([first, second], ignore_index=True)
    original = add_behavioural_features(combined)

    changed = combined.copy()
    changed.loc[0, "PAY_0"] = 6
    changed.loc[0, "BILL_AMT1"] = 999_999
    updated = add_behavioural_features(changed)

    engineered = [c for c in original.columns if c not in combined.columns]
    pd.testing.assert_series_equal(original.loc[1, engineered], updated.loc[1, engineered])


def test_no_infinities_after_engineering() -> None:
    edge = _single_client(LIMIT_BAL=0, BILL_AMT1=0, BILL_AMT2=0, PAY_AMT1=0)
    result = add_behavioural_features(edge)
    numeric = result.select_dtypes("number")
    assert np.isfinite(numeric.to_numpy()).all()


def test_stratified_split_is_deterministic_and_disjoint(sample_dataset: pd.DataFrame) -> None:
    X, y = build_features_target(sample_dataset)
    a = stratified_split(X, y, test_size=0.25, random_state=7)
    b = stratified_split(X, y, test_size=0.25, random_state=7)

    assert a[0].index.tolist() == b[0].index.tolist()
    assert a[1].index.tolist() == b[1].index.tolist()
    assert set(a[0].index).isdisjoint(set(a[1].index))
    assert len(a[0]) + len(a[1]) == len(X)
    # Stratification keeps the positive rate close on both sides.
    assert abs(a[2].mean() - a[3].mean()) < 0.1
