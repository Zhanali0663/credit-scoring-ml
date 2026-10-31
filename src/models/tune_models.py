"""CLI entry point: optimise boosted-tree hyperparameters within a time budget.

Usage::

    python -m src.models.tune_models --budget 240
    python -m src.models.tune_models --budget 90 --models xgboost

Progress is mirrored into ``reports/metrics/tuning_log.txt``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import basic_clean, load_raw_data
from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split
from src.models.tuning import _fmt, tune_model
from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(
    "tune_models",
    log_file=get_project_root() / "reports" / "metrics" / "tuning_log.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-boxed Optuna tuning")
    parser.add_argument(
        "--budget",
        type=float,
        default=210,
        help="Per-model time budget in seconds (default 210 = 3.5 min)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Maximum number of Optuna trials per model (hard cap). Default 25",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="xgboost,lightgbm",
        help="Comma-separated list of models to tune (xgboost,lightgbm)",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("TUNING RUN START | budget per model: %s | models: %s", _fmt(args.budget), args.models)

    # ------------------------------------------------------------------
    # Data preparation (mirrors the main pipeline)
    # ------------------------------------------------------------------
    df = basic_clean(load_raw_data())
    df = add_behavioural_features(df)
    X, y = build_features_target(df)
    X_tr, _, y_tr, _ = stratified_split(X, y, test_size=0.2, random_state=42)
    logger.info("Feature matrix: %s | target balance: %s", X_tr.shape, y_tr.value_counts().to_dict())

    # ------------------------------------------------------------------
    # Sequential tuning
    # ------------------------------------------------------------------
    report: dict = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        budget = args.budget
        remaining_total = args.budget * len(args.models.split(",")) - (time.perf_counter() - t_start)
        if remaining_total <= 0:
            logger.warning("Time budget exceeded — skipping %s", model)
            continue
        budget = min(budget, max(remaining_total, 30.0))
        logger.info(
            "Tuning %s | budget for this model: %s | max trials: %d",
            model,
            _fmt(budget),
            args.trials,
        )
        report[model] = tune_model(
            model,
            X_tr,
            y_tr,
            budget_seconds=budget,
            max_trials=args.trials,  # type: ignore[arg-type]
        )

    # Persist best params
    out = ROOT / "configs" / "best_params.yaml"
    payload = {m: r["best_params"] for m, r in report.items()}
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    for m, r in report.items():
        logger.info(
            "SUMMARY %s | trials=%d | best CV AUC=%.4f",
            m,
            r["trials"],
            r["best_cv_auc"],
        )
    logger.info("Saved best params to %s", out)
    logger.info("TUNING RUN DONE in %s", _fmt(time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
