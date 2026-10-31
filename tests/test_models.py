"""Tests for model registry and pipelines."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.models.pipe import build_pipeline
from src.models.registry import MODEL_REGISTRY, build_model


def test_model_registry_is_not_empty() -> None:
    assert len(MODEL_REGISTRY) >= 3
    for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]:
        assert name in MODEL_REGISTRY


def test_build_model_returns_estimator() -> None:
    for name in MODEL_REGISTRY:
        est = build_model(name)
        assert est is not None


def test_build_pipeline_returns_pipeline() -> None:
    pipe = build_pipeline("logistic_regression", num_features=["LIMIT_BAL", "AGE"])
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert "estimator" in pipe.named_steps
