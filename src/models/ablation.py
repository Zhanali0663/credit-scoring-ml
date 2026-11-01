"""Feature-engineering and hyperparameter ablation study.

The study answers two separate questions with the same model family
(LightGBM) and the same data:

1. Does the 38-feature engineered representation beat the 23 raw columns?
2. Do the Optuna hyperparameters from ``configs/best_params.yaml`` beat the
   registry defaults?

Four configurations are evaluated on five stratified 80/20 splits:

``raw_default`` / ``eng_default`` / ``raw_tuned`` / ``eng_tuned``.

Usage::

    python -m src.models.ablation
    python -m src.models.ablation --seeds 42,123,999
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import basic_clean, load_raw_data
from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split
from src.models.pipe import build_pipeline
from src.utils.config import get_project_root
from src.utils.logging import get_logger
from src.utils.metrics import evaluate_classifier

logger = get_logger(__name__)

#: The 23 columns present in the UCI dataset (minus ID and the target).
RAW_FEATURES: list[str] = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
]

DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 999, 2026, 7)
CONFIGS: tuple[tuple[str, bool, bool], ...] = (
    # label, use_engineered, use_tuned
    ("raw_default", False, False),
    ("eng_default", True, False),
    ("raw_tuned", False, True),
    ("eng_tuned", True, True),
)

PAIRED_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("feature_engineering_default", "eng_default", "raw_default"),
    ("feature_engineering_tuned", "eng_tuned", "raw_tuned"),
    ("tuning_raw", "raw_tuned", "raw_default"),
    ("tuning_engineered", "eng_tuned", "eng_default"),
    ("combined", "eng_tuned", "raw_default"),
)


def _prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return engineered features, target and the 23-column raw frame."""
    raw = basic_clean(load_raw_data())
    engineered = add_behavioural_features(raw)
    X_engineered, y = build_features_target(engineered)
    X_raw = raw[RAW_FEATURES].copy()
    missing = [c for c in RAW_FEATURES if c not in X_raw.columns]
    if missing:  # pragma: no cover - defensive, dataset schema is fixed
        raise ValueError(f"Raw dataset is missing columns: {missing}")
    return X_engineered, y, X_raw


def run_ablation(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    model_name: str = "lightgbm",
    test_size: float = 0.2,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Run the four-configuration ablation and persist the results."""
    seeds = list(seeds)
    X_engineered, y, X_raw = _prepare_data()
    logger.info(
        "Ablation data: %s engineered features, %s raw features, %s rows",
        X_engineered.shape[1],
        X_raw.shape[1],
        len(y),
    )

    rows: list[dict[str, Any]] = []
    feature_sets = {"raw": X_raw, "eng": X_engineered}

    for seed in seeds:
        # Split once per seed so all four configurations see exactly the same
        # train/test rows. Stratification makes the target folds identical for
        # the raw and engineered frames, which we assert defensively.
        split_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        targets: dict[str, tuple[pd.Series, pd.Series]] = {}
        for kind, frame in feature_sets.items():
            X_tr, X_te, y_tr, y_te = stratified_split(
                frame, y, test_size=test_size, random_state=seed
            )
            split_frames[kind] = (X_tr, X_te)
            targets[kind] = (y_tr, y_te)
        y_train, y_test = targets["eng"]
        pd.testing.assert_series_equal(y_train, targets["raw"][0])
        pd.testing.assert_series_equal(y_test, targets["raw"][1])

        for label, use_engineered, use_tuned in CONFIGS:
            kind = "eng" if use_engineered else "raw"
            X_tr, X_te = split_frames[kind]
            pipe = build_pipeline(model_name, num_features=list(X_tr.columns), tuned=use_tuned)
            # Vary the estimator seed as well so "5 seeds" is not only split noise.
            pipe.named_steps["estimator"].set_params(random_state=seed)
            pipe.fit(X_tr, y_train)
            proba = pipe.predict_proba(X_te)[:, 1]
            preds = pipe.predict(X_te)
            metrics = evaluate_classifier(y_test.to_numpy(), preds, proba)
            rows.append(
                {
                    "seed": seed,
                    "config": label,
                    "model": model_name,
                    "features": int(X_tr.shape[1]),
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1": metrics["f1"],
                }
            )
            logger.info(
                "seed=%s config=%-11s features=%d ROC AUC=%.4f",
                seed,
                label,
                X_tr.shape[1],
                metrics["roc_auc"],
            )

    results = pd.DataFrame(rows)
    summary_rows = []
    for label, _, _ in CONFIGS:
        values = results.loc[results["config"] == label, "roc_auc"]
        summary_rows.append(
            {
                "config": label,
                "features": int(results.loc[results["config"] == label, "features"].iloc[0]),
                "mean_roc_auc": float(values.mean()),
                "std_roc_auc": float(values.std(ddof=1)),
                "n_seeds": int(values.size),
            }
        )
    summary = pd.DataFrame(summary_rows)

    comparisons: dict[str, Any] = {}
    pivot = results.pivot(index="seed", columns="config", values="roc_auc")
    for name, treatment, control in PAIRED_COMPARISONS:
        delta = pivot[treatment] - pivot[control]
        comparisons[name] = {
            "treatment": treatment,
            "control": control,
            "mean_delta": float(delta.mean()),
            "std_delta": float(delta.std(ddof=1)),
            "positive_seeds": int((delta > 0).sum()),
            "n_seeds": int(delta.size),
        }

    root = get_project_root()
    output_csv = output_csv or root / "reports" / "metrics" / "ablation.csv"
    output_json = output_json or root / "reports" / "metrics" / "ablation_summary.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    payload = {
        "model": model_name,
        "seeds": seeds,
        "n_features_raw": int(X_raw.shape[1]),
        "n_features_engineered": int(X_engineered.shape[1]),
        "summary": summary.to_dict(orient="records"),
        "paired_comparisons": comparisons,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved per-seed ablation to %s", output_csv)
    logger.info("Saved ablation summary to %s", output_json)
    return {"results": results, "summary": summary, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-engineering / tuning ablation")
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated random seeds",
    )
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_ablation(seeds=seeds, model_name=args.model)


if __name__ == "__main__":
    main()
