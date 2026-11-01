"""Standalone script that downloads and inspects the raw dataset.

Usage::

    python scripts/download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_data  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    df = load_raw_data()
    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns)[:10], )
    logger.info("Head:\n%s", df.head())
    logger.info("Dtypes:\n%s", df.dtypes)


if __name__ == "__main__":
    main()
