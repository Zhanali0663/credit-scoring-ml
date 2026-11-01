"""Regression tests for the SHAP scaling bug and the ablation feature list."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import add_behavioural_features, build_features_target
from src.models.ablation import RAW_FEATURES
from src.models.pipe import build_pipeline
from src.visualization.shap_utils import _scaled_frame, compute_shap_values


def test_ablation_uses_exactly_the_23_raw_columns(sample_dataset: pd.DataFrame) -> None:
    assert len(RAW_FEATURES) == 23
    assert set(RAW_FEATURES) == set(sample_dataset.columns) - {"ID", "default_payment_next_month"}


def test_shap_explains_the_scaled_matrix_but_plots_raw_values(
    sample_dataset: pd.DataFrame,
) -> None:
    df = add_behavioural_features(sample_dataset)
    X, y = build_features_target(df)
    X = X.head(30)
    y = y.head(30)

    pipe = build_pipeline("lightgbm", num_features=list(X.columns))
    pipe.named_steps["estimator"].set_params(n_estimators=20, min_child_samples=5)
    pipe.fit(X, y)

    scaled = _scaled_frame(pipe, X)
    assert not np.allclose(scaled.to_numpy(), X.to_numpy())
    assert np.allclose(scaled.mean(axis=0).to_numpy(), 0.0, atol=1e-8)

    values, display = compute_shap_values(pipe, X)
    assert values.shape == (len(X), X.shape[1])
    pd.testing.assert_frame_equal(display, X)
