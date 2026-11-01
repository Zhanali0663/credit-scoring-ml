"""Backward-compatible entry point for the ablation study.

The maintained implementation lives in :mod:`src.models.ablation`::

    python -m src.models.ablation

This shim keeps older commands working without duplicating the flawed
temporary-file logic that used to live here.
"""
from __future__ import annotations

from src.models.ablation import main

if __name__ == "__main__":
    main()
