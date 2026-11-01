"""Probability calibration study for the headline LightGBM model.

Class weighting makes the raw model rank well but shifts its probabilities
away from observed default rates. This module fits an isotonic
``CalibratedClassifierCV`` around the same pipeline (cross-validated *inside*
the training set) and reports the before/after Brier score, log loss and
reliability diagram.

Usage::

    python -m src.models.calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.loader import basic_clean, load_raw_data
from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split
from src.models.pipe import build_pipeline
from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Binned expected calibration error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        confidence = float(proba[mask].mean())
        observed = float(y_true[mask].mean())
        ece += mask.mean() * abs(confidence - observed)
    return float(ece)


def _probability_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "mean_predicted": float(proba.mean()),
        "observed_rate": float(y_true.mean()),
        "ece_10_bins": expected_calibration_error(y_true, proba),
    }


def _reliability_plot(
    y_true: np.ndarray,
    raw_proba: np.ndarray,
    calibrated_proba: np.ndarray,
    save_to: Path,
) -> None:
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for proba, label, color in (
        (raw_proba, "Raw (class-weighted)", "#e74c3c"),
        (calibrated_proba, "Isotonic calibration", "#2ecc71"),
    ):
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability diagram — LightGBM (test set)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_to, dpi=130)
    plt.close(fig)


def run_calibration(
    random_state: int = 42,
    n_splits: int = 5,
    save_artifact: bool = True,
) -> dict[str, Any]:
    """Fit and evaluate raw and isotonic-calibrated LightGBM pipelines."""
    root = get_project_root()
    config = yaml.safe_load((root / "configs" / "paths.yaml").read_text("utf-8"))
    df = basic_clean(load_raw_data())
    df = add_behavioural_features(df)
    X, y = build_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["data"]["test_size"], random_state=random_state
    )
    num_features = X_train.select_dtypes(include="number").columns.tolist()

    logger.info("Fitting raw LightGBM pipeline")
    raw_pipe = build_pipeline("lightgbm", num_features=num_features)
    raw_pipe.fit(X_train, y_train)
    raw_proba = raw_pipe.predict_proba(X_test)[:, 1]

    logger.info("Fitting isotonic calibration (%d internal folds)", n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    calibrated = CalibratedClassifierCV(
        estimator=build_pipeline("lightgbm", num_features=num_features),
        method="isotonic",
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    calibrated_proba = calibrated.predict_proba(X_test)[:, 1]

    y_test_arr = y_test.to_numpy()
    report = {
        "model": "lightgbm",
        "random_state": random_state,
        "n_splits": n_splits,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "raw": _probability_metrics(y_test_arr, raw_proba),
        "calibrated_isotonic": _probability_metrics(y_test_arr, calibrated_proba),
    }
    report["deltas"] = {
        key: report["calibrated_isotonic"][key] - report["raw"][key]
        for key in ("brier", "log_loss", "ece_10_bins")
    }

    metrics_dir = root / "reports" / "metrics"
    figures_dir = root / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _reliability_plot(
        y_test_arr,
        raw_proba,
        calibrated_proba,
        figures_dir / "calibration_reliability.png",
    )

    if save_artifact:
        artifacts_dir = root / "models_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated, artifacts_dir / "lightgbm_calibrated.joblib")

    logger.info(
        "Brier raw=%.4f calibrated=%.4f | log loss raw=%.4f calibrated=%.4f",
        report["raw"]["brier"],
        report["calibrated_isotonic"]["brier"],
        report["raw"]["log_loss"],
        report["calibrated_isotonic"]["log_loss"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration study (isotonic)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()
    run_calibration(
        random_state=args.random_state,
        n_splits=args.folds,
        save_artifact=not args.no_artifact,
    )


if __name__ == "__main__":
    main()
