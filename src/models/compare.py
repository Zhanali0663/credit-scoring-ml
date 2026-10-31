"""Compare multiple classifiers on the credit scoring dataset."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.pipe import build_pipeline
from src.models.registry import MODEL_REGISTRY
from src.models.trainer import cross_validate, fit_and_save
from src.utils.config import get_project_root
from src.utils.logging import get_logger
from src.utils.metrics import _fmt_seconds, evaluate_classifier

logger = get_logger(__name__)


def run_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: list[str],
    metrics_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train every model in :data:`MODEL_REGISTRY` and return a summary.

    The function performs cross-validation on the training set, fits each
    model on the full training set, evaluates on the held-out test set and
    writes ``reports/metrics/comparison.csv`` and ``.json`` files.
    """
    if metrics_dir is None:
        metrics_dir = get_project_root() / "reports" / "metrics"
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "models_artifacts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    full_report: dict[str, Any] = {}

    run_start = time.perf_counter()
    names = list(MODEL_REGISTRY)
    for i, name in enumerate(names, start=1):
        remaining_est = (time.perf_counter() - run_start) / i * (len(names) - i)
        logger.info("=" * 70)
        logger.info(
            "[%d/%d] Evaluating %s | elapsed %s | estimated remaining %s",
            i,
            len(names),
            name,
            _fmt_seconds(time.perf_counter() - run_start),
            _fmt_seconds(remaining_est),
        )
        pipeline = build_pipeline(name, num_features=num_features)
        cv_summary = cross_validate(
            pipeline, X_train, y_train, n_splits=n_splits, random_state=random_state
        )

        # Refit on full training set for the final test-set evaluation
        pipeline = build_pipeline(name, num_features=num_features)
        fit_and_save(pipeline, X_train, y_train, name, artifacts_dir=artifacts_dir)

        proba = pipeline.predict_proba(X_test)
        preds = pipeline.predict(X_test)
        test_metrics = evaluate_classifier(y_test.values, preds, proba)

        row = {"model": name, **test_metrics}
        for k, v in cv_summary["mean"].items():
            row[f"cv_mean_{k}"] = v
        for k, v in cv_summary["std"].items():
            row[f"cv_std_{k}"] = v
        rows.append(row)

        full_report[name] = {
            "test": test_metrics,
            "cv_summary": cv_summary,
        }
        logger.info(
            "[%d/%d] %s вњ“ | test ROC AUC %.4f | overall elapsed %s",
            i,
            len(names),
            name,
            test_metrics["roc_auc"],
            _fmt_seconds(time.perf_counter() - run_start),
        )

    df = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
    csv_path = metrics_dir / "comparison.csv"
    json_path = metrics_dir / "comparison.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Saved comparison table to %s", csv_path)
    logger.info("Saved detailed report to %s", json_path)
    return {"table": df, "report": full_report}
