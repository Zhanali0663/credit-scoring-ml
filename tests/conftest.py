"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_dataset() -> pd.DataFrame:
    """Return a small synthetic dataset that mimics the UCI dataset columns."""
    rng = np.random.default_rng(0)
    n = 200
    data = {
        "ID": np.arange(n),
        "LIMIT_BAL": rng.integers(10_000, 500_000, size=n),
        "SEX": rng.integers(1, 3, size=n),
        "EDUCATION": rng.integers(1, 5, size=n),
        "MARRIAGE": rng.integers(1, 4, size=n),
        "AGE": rng.integers(21, 70, size=n),
    }
    for i in range(1, 7):
        data[f"PAY_{i if i > 1 else 0}"] = rng.integers(-2, 6, size=n)
        data[f"BILL_AMT{i}"] = rng.integers(0, 300_000, size=n)
        data[f"PAY_AMT{i}"] = rng.integers(0, 100_000, size=n)
    data["default_payment_next_month"] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)
