"""Shared inference helpers used by the Streamlit demo and smoke tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features.engineering import add_behavioural_features
from src.utils.config import get_project_root

MODEL_NAMES = [
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "logistic_regression",
]


def default_artifacts_dir() -> Path:
    return get_project_root() / "models_artifacts"


def available_models(artifacts_dir: Path | None = None) -> list[str]:
    """Return trained model names that exist on disk."""
    directory = Path(artifacts_dir) if artifacts_dir is not None else default_artifacts_dir()
    present = {path.stem for path in directory.glob("*.joblib")}
    ordered = [name for name in MODEL_NAMES if name in present]
    return ordered or sorted(present)


def load_model(model_name: str = "lightgbm", artifacts_dir: Path | None = None) -> Any:
    directory = Path(artifacts_dir) if artifacts_dir is not None else default_artifacts_dir()
    path = directory / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run `python -m src.utils.pipeline` first."
        )
    return joblib.load(path)


def predict_one(
    client: dict[str, int | float],
    model_name: str = "lightgbm",
    artifacts_dir: Path | None = None,
    threshold: float = 0.5,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Score a single borrower profile.

    Parameters
    ----------
    client : dict
        The 23 raw UCI fields for one borrower.
    model_name : str
        Artifact name in ``models_artifacts/``.
    artifacts_dir : Path, optional
        Override the artifact directory.
    threshold : float
        Decision cut-off applied to the default probability.
    pipeline : optional
        Pre-loaded pipeline; avoids repeated disk I/O in the demo.
    """
    pipe = pipeline if pipeline is not None else load_model(model_name, artifacts_dir)
    frame = pd.DataFrame([client])
    engineered = add_behavioural_features(frame)

    expected = getattr(pipe, "feature_names_in_", None)
    if expected is not None:
        engineered = engineered[list(expected)]

    proba = float(pipe.predict_proba(engineered)[0, 1])
    return {
        "model": model_name,
        "default_proba": proba,
        "prediction": int(proba >= threshold),
        "threshold": float(threshold),
        "features": engineered,
    }


__all__ = [
    "MODEL_NAMES",
    "available_models",
    "default_artifacts_dir",
    "load_model",
    "predict_one",
]
