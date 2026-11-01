"""End-to-end test that runs the full pipeline on a synthetic dataset.

This test does **not** rely on a network download — it patches the loader to
return an in-memory dataframe so it can be run anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split
from src.models.compare import run_comparison


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 800
    data = {
        "ID": np.arange(n),
        "LIMIT_BAL": rng.integers(10_000, 500_000, size=n),
        "SEX": rng.integers(1, 3, size=n),
        "EDUCATION": rng.integers(1, 5, size=n),
        "MARRIAGE": rng.integers(1, 4, size=n),
        "AGE": rng.integers(21, 70, size=n),
    }
    pay_vals = rng.integers(-2, 6, size=(n, 6))
    bill_vals = rng.integers(0, 300_000, size=(n, 6))
    pay_amt_vals = rng.integers(0, 50_000, size=(n, 6))
    for i, val in enumerate(pay_vals.T):
        data["PAY_0" if i == 0 else f"PAY_{i + 1}"] = val
    for i, val in enumerate(bill_vals.T):
        data[f"BILL_AMT{i + 1}"] = val
    for i, val in enumerate(pay_amt_vals.T):
        data[f"PAY_AMT{i + 1}"] = val
    data["default_payment_next_month"] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


def test_full_pipeline_smoke(tiny_dataset: pd.DataFrame, tmp_path: Path) -> None:
    df = add_behavioural_features(tiny_dataset)
    X, y = build_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.25)

    num_features = X_train.select_dtypes(include="number").columns.tolist()
    summary = run_comparison(
        X_train,
        y_train,
        X_test,
        y_test,
        num_features=num_features,
        metrics_dir=tmp_path / "metrics",
        artifacts_dir=tmp_path / "artifacts",
        n_splits=2,  # keep test fast
    )

    table = summary["table"]
    assert not table.empty
    assert {"model", "roc_auc", "f1"}.issubset(table.columns)
    artifacts = list((tmp_path / "artifacts").glob("*.joblib"))
    assert len(artifacts) == len(table)
    assert (tmp_path / "metrics" / "comparison.csv").exists()
    assert (tmp_path / "metrics" / "comparison.json").exists()
