"""Cost-sensitive decision-threshold analysis.

The default 0.5 cut-off is arbitrary. This module selects a threshold on
*out-of-fold* training predictions (never on the test set), then evaluates the
frozen threshold once on the holdout.

Cost model (per client)::

    cost = 5 * false_negatives + 1 * false_positives

i.e. a missed default is assumed to be five times as expensive as a false
alarm. That ratio is an explicit assumption, not a fact about the business.

The search is run on isotonic-calibrated probabilities. Nested
cross-validation is used: the outer folds produce out-of-fold predictions,
while the calibrator is fitted on the inner folds only.

Usage::

    python -m src.models.threshold
    python -m src.models.threshold --cost-fn 8 --cost-fp 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data.loader import basic_clean, load_raw_data
from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split
from src.models.pipe import build_pipeline
from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)


def cumulative_cost(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, cost_fn: float, cost_fp: float
) -> float:
    """Total cost at a single threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return float(cost_fn * fn + cost_fp * fp)


def cost_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> np.ndarray:
    return np.array([cumulative_cost(y_true, proba, t, cost_fn, cost_fp) for t in thresholds])


def select_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    n_grid: int = 501,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the cost-minimising threshold and the evaluated grid."""
    thresholds = np.linspace(0.01, 0.99, n_grid)
    costs = cost_curve(y_true, proba, thresholds, cost_fn, cost_fp)
    best = float(thresholds[int(np.argmin(costs))])
    return best, thresholds, costs


def _threshold_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, cost_fn: float, cost_fp: float
) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "cost_total": float(cost_fn * fn + cost_fp * fp),
        "cost_per_client": float((cost_fn * fn + cost_fp * fp) / len(y_true)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _plot_cost_curve(
    thresholds: np.ndarray,
    oof_costs: np.ndarray,
    best_threshold: float,
    test_threshold: float,
    save_to: Path,
) -> None:
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, oof_costs / max(oof_costs.max(), 1.0), label="Out-of-fold cost")
    ax.axvline(best_threshold, color="#2ecc71", linestyle="--", label=f"Selected = {best_threshold:.2f}")
    ax.axvline(test_threshold, color="#e74c3c", linestyle=":", label=f"Baseline = {test_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Cost relative to maximum")
    ax.set_title("Cost-sensitive threshold selection (5:1 cost ratio)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_to, dpi=130)
    plt.close(fig)


def run_threshold_analysis(
    random_state: int = 42,
    outer_folds: int = 5,
    inner_folds: int = 5,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> dict[str, Any]:
    """Select a threshold out-of-fold and evaluate it on the holdout."""
    root = get_project_root()
    config = yaml.safe_load((root / "configs" / "paths.yaml").read_text("utf-8"))
    df = basic_clean(load_raw_data())
    df = add_behavioural_features(df)
    X, y = build_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["data"]["test_size"], random_state=random_state
    )
    num_features = X_train.select_dtypes(include="number").columns.tolist()
    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    calibrated = CalibratedClassifierCV(
        estimator=build_pipeline("lightgbm", num_features=num_features),
        method="isotonic",
        cv=inner_cv,
    )
    logger.info(
        "Generating out-of-fold calibrated predictions (%d outer x %d inner folds)",
        outer_folds,
        inner_folds,
    )
    oof_proba = cross_val_predict(
        calibrated, X_train, y_train, cv=outer_cv, method="predict_proba", n_jobs=1
    )[:, 1]

    best_threshold, thresholds, oof_costs = select_threshold(
        y_train_arr, oof_proba, cost_fn=cost_fn, cost_fp=cost_fp
    )

    logger.info("Refitting calibrated model on the full training set")
    calibrated.fit(X_train, y_train)
    test_proba = calibrated.predict_proba(X_test)[:, 1]

    oof_at_default = _threshold_metrics(
        y_train_arr, oof_proba, 0.5, cost_fn, cost_fp
    )
    oof_at_best = _threshold_metrics(
        y_train_arr, oof_proba, best_threshold, cost_fn, cost_fp
    )
    test_at_default = _threshold_metrics(
        y_test_arr, test_proba, 0.5, cost_fn, cost_fp
    )
    test_at_best = _threshold_metrics(
        y_test_arr, test_proba, best_threshold, cost_fn, cost_fp
    )

    report = {
        "model": "lightgbm_isotonic_calibrated",
        "random_state": random_state,
        "cost_matrix": {"false_negative": cost_fn, "false_positive": cost_fp},
        "selection": "out-of-fold training predictions",
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "selected_threshold": float(best_threshold),
        "baseline_threshold": 0.5,
        "oof_at_0.5": oof_at_default,
        "oof_at_selected": oof_at_best,
        "test_at_0.5": test_at_default,
        "test_at_selected": test_at_best,
        "test_cost_reduction_pct": float(
            100.0 * (test_at_default["cost_total"] - test_at_best["cost_total"])
            / max(test_at_default["cost_total"], 1.0)
        ),
    }

    metrics_dir = root / "reports" / "metrics"
    figures_dir = root / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "threshold.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _plot_cost_curve(
        thresholds,
        oof_costs,
        best_threshold,
        0.5,
        figures_dir / "threshold_cost_curve.png",
    )

    logger.info(
        "Selected threshold %.3f | test cost %.0f vs %.0f at 0.5 (%.1f%% lower)",
        best_threshold,
        test_at_best["cost_total"],
        test_at_default["cost_total"],
        report["test_cost_reduction_pct"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-sensitive threshold selection")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--cost-fn", type=float, default=5.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    args = parser.parse_args()
    run_threshold_analysis(
        random_state=args.random_state,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
    )


if __name__ == "__main__":
    main()
