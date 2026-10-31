"""SHAP-based interpretability for tree-based models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shap_summary_plot(
    model: Any,
    X: pd.DataFrame,
    save_to: Path,
    max_display: int = 20,
) -> None:
    """Save a SHAP beeswarm summary plot."""
    try:
        import shap  # type: ignore
    except ImportError:
        # silently skip if shap is not available
        return

    save_to.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification shap_values may be a list (legacy API)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
        # Newer SHAP returns (n_samples, n_features) for binary classifiers
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close()


def shap_bar_plot(
    model: Any,
    X: pd.DataFrame,
    save_to: Path,
    max_display: int = 20,
) -> None:
    """Save a SHAP mean |value| bar plot."""
    try:
        import shap  # type: ignore
    except ImportError:
        return

    save_to.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X, max_display=max_display, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close()
