"""Standalone inference helper.

Loads a trained pipeline and predicts default probability for new clients
passed as a dict or a pandas DataFrame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features.engineering import add_behavioural_features
from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_artifacts(artifacts_dir: Path | None = None) -> dict[str, Any]:
    """Load all available model artifacts.

    Returns a dict of ``{model_name: pipeline}``.
    """
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "models_artifacts"
    models: dict[str, Any] = {}
    for path in sorted(artifacts_dir.glob("*.joblib")):
        models[path.stem] = joblib.load(path)
    if not models:
        raise FileNotFoundError(f"No model artifacts found in {artifacts_dir}")
    return models


def predict_one(
    client: dict[str, Any],
    model_name: str = "lightgbm",
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Predict default probability for a single client dict."""
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "models_artifacts"
    pipeline = joblib.load(artifacts_dir / f"{model_name}.joblib")

    df = pd.DataFrame([client])
    df = add_behavioural_features(df)

    # Make sure column order matches what the pipeline was trained on
    expected_cols = (
        list(pipeline.feature_names_in_) if hasattr(pipeline, "feature_names_in_") else df.columns.tolist()
    )
    df = df[expected_cols]

    proba = float(pipeline.predict_proba(df)[0, 1])
    pred = int(pipeline.predict(df)[0])

    return {"model": model_name, "default_proba": proba, "prediction": pred}


def predict_batch(
    df: pd.DataFrame,
    model_name: str = "lightgbm",
    artifacts_dir: Path | None = None,
) -> pd.DataFrame:
    """Predict default probability for a batch of clients."""
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "models_artifacts"
    pipeline = joblib.load(artifacts_dir / f"{model_name}.joblib")

    df_eng = add_behavioural_features(df.copy())
    proba = pipeline.predict_proba(df_eng)[:, 1]
    preds = pipeline.predict(df_eng)
    out = df.copy()
    out["default_proba"] = proba
    out["prediction"] = preds
    return out
