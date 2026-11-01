"""SHAP-based interpretability for tree-based models.

The estimator inside our ``Pipeline`` is trained on **standardised** features,
so the explainer must receive the same scaled matrix. Plots, however, should
show the original feature values so that a reader can interpret "high
``LIMIT_BAL``" rather than "high z-score". These helpers therefore explain the
scaled matrix and plot against the raw frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _unwrap(pipeline: Any) -> tuple[Any, Any]:
    """Return ``(scaler, estimator)`` from a pipeline or a bare estimator."""
    if hasattr(pipeline, "named_steps"):
        scaler = pipeline.named_steps.get("scaler")
        estimator = pipeline.named_steps.get("estimator", pipeline)
        return scaler, estimator
    return None, pipeline


def _scaled_frame(pipeline: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the pipeline's scaler without touching the fitted estimator."""
    scaler, _ = _unwrap(pipeline)
    if scaler is None:
        return X
    values = scaler.transform(X)
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def _normalise_shap_values(shap_values: Any) -> Any:
    """Return a 2-D ``(n_samples, n_features)`` matrix for the positive class."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    if getattr(shap_values, "values", None) is not None:
        shap_values = shap_values.values
    if getattr(shap_values, "ndim", 2) == 3:
        return shap_values[:, :, 1]
    return shap_values


def compute_shap_values(
    pipeline: Any, X: pd.DataFrame
) -> tuple[Any, pd.DataFrame]:
    """Compute SHAP values on the model's scaled input.

    Returns
    -------
    shap_values : ndarray
        Matrix of shape ``(n_samples, n_features)``.
    X_display : pandas.DataFrame
        The original (unscaled) rows, suitable for plotting.
    """
    import shap  # type: ignore

    _, estimator = _unwrap(pipeline)
    X_model = _scaled_frame(pipeline, X)
    explainer = shap.TreeExplainer(estimator)
    values = _normalise_shap_values(explainer.shap_values(X_model))
    return values, X


def shap_summary_plot(
    pipeline: Any,
    X: pd.DataFrame,
    save_to: Path,
    max_display: int = 20,
) -> None:
    """Save a SHAP beeswarm summary plot."""
    try:
        import shap  # type: ignore
    except ImportError:
        return

    values, X_display = compute_shap_values(pipeline, X)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(values, X_display, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close()


def shap_bar_plot(
    pipeline: Any,
    X: pd.DataFrame,
    save_to: Path,
    max_display: int = 20,
) -> None:
    """Save a SHAP mean |value| bar plot."""
    try:
        import shap  # type: ignore
    except ImportError:
        return

    values, X_display = compute_shap_values(pipeline, X)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        values, X_display, max_display=max_display, show=False, plot_type="bar"
    )
    plt.tight_layout()
    plt.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close()
