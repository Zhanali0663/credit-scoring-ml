"""Data loading utilities for the credit scoring project.

The default dataset is the UCI ``Default of Credit Card Clients`` dataset
(30,000 records, 23 features, binary target ``default_payment_next_month``).

The module supports two acquisition modes:

1. ``kagglehub`` - downloads from Kaggle (requires ``~/.kaggle/kaggle.json``).
2. Direct HTTP download - from a mirror of the UCI repository.

Both modes are wrapped with a fallback: if the dataset is already in
``data/raw/`` we just read it from disk.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical Kaggle dataset slug for the UCI credit card default dataset.
KAGGLE_DATASET = "uciml/default-of-credit-card-clients-dataset"
# Direct CSV URL (UCI mirror). Used as fallback if Kaggle credentials are missing.
DIRECT_URL = (
    "https://raw.githubusercontent.com/selva86/datasets/master/"
    "default%20of%20credit%20card%20clients.csv"
)
EXPECTED_FILENAME = "UCI_Credit_Card.csv"


def _default_raw_dir() -> Path:
    return get_project_root() / "data" / "raw"


def _find_existing(raw_dir: Path) -> Optional[Path]:
    """Return the existing dataset file in ``raw_dir`` if any."""
    if not raw_dir.exists():
        return None
    candidates = [
        raw_dir / EXPECTED_FILENAME,
        raw_dir / "default of credit card clients.csv",
        raw_dir / "default_of_credit_card_clients.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # last fallback: first CSV in the folder
    csvs = list(raw_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def download_from_kagglehub(target_dir: Path) -> Path:
    """Download the dataset via ``kagglehub`` (requires Kaggle credentials)."""
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled below
        raise RuntimeError("kagglehub is not installed") from exc

    logger.info("Downloading dataset via kagglehub: %s", KAGGLE_DATASET)
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src = _find_existing(cache_path)
    if src is None:
        raise FileNotFoundError(f"No CSV found in kagglehub cache: {cache_path}")
    target = target_dir / EXPECTED_FILENAME
    shutil.copy(src, target)
    logger.info("Saved dataset to %s", target)
    return target


def download_from_url(target_dir: Path, url: str = DIRECT_URL) -> Path:
    """Direct HTTP download of the CSV file."""
    logger.info("Downloading dataset from %s", url)
    target = target_dir / EXPECTED_FILENAME
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(url)
    df.to_csv(target, index=False)
    logger.info("Saved dataset to %s", target)
    return target


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """Load the raw credit card default dataset.

    Strategy:
        1. If ``force_download=False`` and a CSV already exists in ``data/raw``,
           read it.
        2. Otherwise try ``kagglehub`` (will fail silently if creds are missing).
        3. Fall back to direct URL download.

    Parameters
    ----------
    force_download : bool, default ``False``
        If ``True``, always re-download the dataset.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.
    """
    raw_dir = _default_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = None if force_download else _find_existing(raw_dir)
    if existing is not None:
        logger.info("Loading dataset from %s", existing)
        df = pd.read_csv(existing)
        return df

    # Try Kaggle first, then direct URL
    try:
        target = download_from_kagglehub(raw_dir)
    except Exception as exc:  # noqa: BLE001 - broad to allow fallback
        logger.warning("Kaggle download failed (%s), falling back to direct URL", exc)
        target = download_from_url(raw_dir)

    df = pd.read_csv(target)
    return df


_RENAME_MAP = {
    "ID": "ID",
    "LIMIT_BAL": "LIMIT_BAL",
    "SEX": "SEX",
    "EDUCATION": "EDUCATION",
    "MARRIAGE": "MARRIAGE",
    "AGE": "AGE",
    "PAY_0": "PAY_0",
    "PAY_2": "PAY_2",
    "PAY_3": "PAY_3",
    "PAY_4": "PAY_4",
    "PAY_5": "PAY_5",
    "PAY_6": "PAY_6",
    **{f"BILL_AMT{i}": f"BILL_AMT{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"PAY_AMT{i}" for i in range(1, 7)},
    "default.payment.next.month": "default_payment_next_month",
    # snake_case variants (some mirrors rename the columns)
    "default_payment_next_month": "default_payment_next_month",
}


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and dtypes.

    1. Strips whitespace from column names.
    2. Renames dotted variants to the canonical snake-case names used in the
       rest of the project.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename = {c: _RENAME_MAP.get(c, c) for c in df.columns if c in _RENAME_MAP}
    df = df.rename(columns=rename)

    # Downcast nullable float columns that should be int (when no NaNs)
    for col in df.columns[df.isna().sum() == 0]:
        if pd.api.types.is_float_dtype(df[col]):
            if (df[col] % 1 == 0).all():
                df[col] = df[col].astype("int64")
    return df


__all__ = ["load_raw_data", "basic_clean"]
