"""Cross-validated training pipeline for credit scoring.

Each model is evaluated using stratified k-fold cross-validation on the
training set. We log ROC AUC, PR AUC, F1 and report the mean +/- std.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.utils.config import get_project_root
from src.utils.logging import get_logger
from src.utils.metrics import evaluate_classifier

logger = get_logger(__name__)


def cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run stratified k-fold cross-validation for a single pipeline.

    Returns a dictionary with the mean and std of each metric, and the per-fold
    detailed metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_val)
        preds = pipeline.predict(X_val)
        m = evaluate_classifier(y_val.values, preds, proba)
        m["fold"] = fold_idx
        fold_metrics.append(m)
        logger.info(
            "Fold %d | ROC AUC %.4f | PR AUC %.4f | F1 %.4f",
            fold_idx,
            m["roc_auc"],
            m["pr_auc"],
            m["f1"],
        )

    df = pd.DataFrame(fold_metrics).drop(columns=["fold"])
    summary = {
        "folds": n_splits,
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "per_fold": fold_metrics,
    }
    return summary


def fit_and_save(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    artifacts_dir: Path | None = None,
) -> Pipeline:
    """Fit the pipeline on the full training set and serialise it."""
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "models_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fitting %s on full training set...", model_name)
    pipeline.fit(X, y)
    out = artifacts_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, out)
    logger.info("Saved %s to %s", model_name, out)
    return pipeline


def load_pipeline(model_name: str, artifacts_dir: Path | None = None) -> Pipeline:
    """Load a previously saved pipeline."""
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "models_artifacts"
    path = artifacts_dir / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)
