"""Pipeline factory - builds a ColumnTransformer + estimator pipeline."""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.registry import build_model


def build_pipeline(model_name: str, num_features: list[str], tuned: bool = True) -> Pipeline:
    """Build a scaling + estimator pipeline.

    Numerical features are standardized (helps Logistic Regression and
    tree-based models with monotone constraints). The pipeline is deliberately
    simple — column types are pre-handled in the dataset (categorical cols are
    already integer-encoded).

    ``tuned=False`` keeps the registry defaults and ignores
    ``configs/best_params.yaml``; this is used by the ablation study.
    """
    estimator = build_model(model_name, tuned=tuned)
    steps = [
        ("scaler", StandardScaler()),
        ("estimator", estimator),
    ]
    return Pipeline(steps=steps)
