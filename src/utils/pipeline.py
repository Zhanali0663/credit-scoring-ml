"""Top-level orchestrator that runs the full training pipeline.

Usage::

    python -m src.utils.pipeline
    # or
    from src.utils.pipeline import run
    run()
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from src.data.loader import basic_clean, load_raw_data
from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split
from src.models.compare import run_comparison
from src.models.trainer import load_pipeline
from src.utils.config import get_project_root, load_config
from src.utils.logging import get_logger
from src.utils.metrics import _fmt_seconds
from src.visualization import (
    plot_class_balance,
    plot_comparison_bar,
    plot_confusion,
    plot_feature_importance,
    plot_roc_pr,
    shap_bar_plot,
    shap_summary_plot,
)

logger = get_logger(__name__)


def _numerical_features(X: pd.DataFrame) -> List[str]:
    return X.select_dtypes(include="number").columns.tolist()


PHASES_TOTAL = 6


class _PhaseTracker:
    """Logs every pipeline phase with elapsed vs total elapsed time."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.phase = 0

    def begin(self, name: str) -> None:
        self.phase += 1
        logger.info(
            "PHASE %d/%d > %s | elapsed so far %s",
            self.phase,
            PHASES_TOTAL,
            name,
            _fmt_seconds(time.perf_counter() - self.start),
        )

    def done(self, extra: str = "") -> None:
        suffix = f" — {extra}" if extra else ""
        logger.info(
            "PHASE %d/%d ✓ done%s | total elapsed %s",
            self.phase,
            PHASES_TOTAL,
            suffix,
            _fmt_seconds(time.perf_counter() - self.start),
        )


def run() -> Dict[str, Any]:
    """Run the full pipeline: load → engineer → split → compare → visualise."""
    tracker = _PhaseTracker()
    config = load_config(get_project_root() / "configs" / "paths.yaml")

    # --- 1. Load ---
    tracker.begin("Load raw data")
    df = load_raw_data()
    df = basic_clean(df)
    logger.info("Raw shape: %s", df.shape)
    tracker.done()

    # --- 2. Feature engineering ---
    tracker.begin("Feature engineering")
    df = add_behavioural_features(df)
    X, y = build_features_target(df, target_col=config["data"]["target_col"])
    logger.info("Feature matrix: %s | target balance: %s", X.shape, y.value_counts().to_dict())
    tracker.done()

    figures_dir = get_project_root() / config["models"]["figures_dir"]
    plot_class_balance(y, save_to=figures_dir / "class_balance.png")

    # --- 3. Split ---
    tracker.begin("Stratified split")
    X_train, X_test, y_train, y_test = stratified_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    logger.info("Train: %s | Test: %s", X_train.shape, X_test.shape)
    tracker.done()

    # --- 4. Compare models ---
    tracker.begin("Cross-validated model comparison (5 models, 5 folds each)")
    num_features = _numerical_features(X_train)
    summary = run_comparison(
        X_train,
        y_train,
        X_test,
        y_test,
        num_features=num_features,
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"],
    )
    tracker.done()

    # --- 5. Visualisations ---
    tracker.begin("Generating figures and interpretability artifacts")
    table = summary["table"]
    metrics_dir = get_project_root() / config["models"]["metrics_dir"]
    artifacts_dir = get_project_root() / config["models"]["artifacts_dir"]

    plot_comparison_bar(table, metric="roc_auc", save_to=figures_dir / "comparison_roc_auc.png")
    plot_comparison_bar(table, metric="pr_auc", save_to=figures_dir / "comparison_pr_auc.png")
    plot_comparison_bar(table, metric="f1", save_to=figures_dir / "comparison_f1.png")

    # Use the best tree model (LightGBM by default) for interpretability
    best_model_name = table.iloc[0]["model"]
    logger.info("Best model on ROC AUC: %s", best_model_name)
    pipe = load_pipeline(best_model_name, artifacts_dir=artifacts_dir)
    estimator = pipe.named_steps["estimator"]

    # Test-set curves for the best model
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    plot_roc_pr(y_test.values, proba, best_model_name, save_to=figures_dir / f"{best_model_name}_roc_pr.png")
    plot_confusion(y_test.values, preds, best_model_name, save_to=figures_dir / f"{best_model_name}_confusion.png")

    # Feature importance + SHAP for tree-based models
    if hasattr(estimator, "feature_importances_"):
        plot_feature_importance(
            X.columns,
            estimator.feature_importances_,
            top_n=20,
            title=f"{best_model_name} — Feature importance",
            save_to=figures_dir / f"{best_model_name}_feature_importance.png",
        )

    if best_model_name in {"xgboost", "lightgbm", "catboost", "random_forest"}:
        try:
            shap_summary_plot(estimator, X_test, save_to=figures_dir / f"{best_model_name}_shap_summary.png")
            shap_bar_plot(estimator, X_test, save_to=figures_dir / f"{best_model_name}_shap_bar.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP computation failed for %s: %s", best_model_name, exc)

    # Persist the engineered dataset for downstream apps
    processed_dir = get_project_root() / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    full_df = X.copy()
    full_df[config["data"]["target_col"]] = y.values
    full_df.to_csv(processed_dir / "engineered.csv", index=False)
    logger.info("Saved processed dataset to %s", processed_dir / "engineered.csv")

    tracker.done()
    total = time.perf_counter() - tracker.start
    logger.info("Pipeline finished successfully in %s.", _fmt_seconds(total))
    if total > 600:
        logger.warning(
            "Pipeline exceeded the 10-minute target (%s) — consider reducing "
            "`n_estimators` of tree models or trimming the SHAP sample size.",
            _fmt_seconds(total),
        )
    return summary


if __name__ == "__main__":
    run()
