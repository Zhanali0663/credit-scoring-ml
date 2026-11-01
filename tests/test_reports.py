"""Consistency checks for the committed metric artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"


def test_comparison_csv_matches_json() -> None:
    csv = pd.read_csv(METRICS / "comparison.csv").set_index("model")
    report = json.loads((METRICS / "comparison.json").read_text(encoding="utf-8"))

    assert set(csv.index) == set(report)
    for model, payload in report.items():
        for metric in ("roc_auc", "pr_auc", "f1", "recall", "precision"):
            assert abs(csv.loc[model, metric] - payload["test"][metric]) < 1e-9


def test_ablation_artifact_has_clean_feature_counts_and_five_seeds() -> None:
    df = pd.read_csv(METRICS / "ablation.csv")
    summary = json.loads((METRICS / "ablation_summary.json").read_text(encoding="utf-8"))

    assert set(df["config"]) == {"raw_default", "eng_default", "raw_tuned", "eng_tuned"}
    assert df["seed"].nunique() == 5
    assert len(df) == 20
    assert set(df.loc[df["config"].str.startswith("raw"), "features"]) == {23}
    assert set(df.loc[df["config"].str.startswith("eng"), "features"]) == {61}
    assert summary["n_features_raw"] == 23
    assert summary["n_features_engineered"] == 61

    # The old script silently made "tuned" identical to "default"; guard it.
    pivot = df.pivot(index="seed", columns="config", values="roc_auc")
    assert (pivot["raw_tuned"] != pivot["raw_default"]).any()
    assert (pivot["eng_tuned"] != pivot["eng_default"]).any()


def test_calibration_and_threshold_artifacts_are_consistent() -> None:
    calibration = json.loads((METRICS / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["calibrated_isotonic"]["brier"] < calibration["raw"]["brier"]
    assert calibration["calibrated_isotonic"]["ece_10_bins"] < calibration["raw"]["ece_10_bins"]
    assert abs(
        calibration["calibrated_isotonic"]["roc_auc"] - calibration["raw"]["roc_auc"]
    ) < 0.01

    threshold = json.loads((METRICS / "threshold.json").read_text(encoding="utf-8"))
    assert 0.0 < threshold["selected_threshold"] < 0.5
    assert threshold["test_at_selected"]["cost_total"] < threshold["test_at_0.5"]["cost_total"]
    assert threshold["selection"] == "out-of-fold training predictions"
