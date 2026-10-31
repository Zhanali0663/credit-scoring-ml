"""Visualization helpers for credit scoring evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_class_balance(y: pd.Series, save_to: Path) -> None:
    """Bar plot of class balance."""
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Non-default (0)", "Default (1)"], counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylabel("Count")
    ax.set_title("Target class balance")
    fig.tight_layout()
    _ensure_dir(save_to)
    fig.savefig(save_to, dpi=120)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_to: Path) -> None:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-default", "Default"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(save_to)
    fig.savefig(save_to, dpi=120)
    plt.close(fig)


def plot_roc_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title_prefix: str,
    save_to: Path,
) -> None:
    """ROC + PR curves saved as a single PNG with two subplots."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=axes[0])
    axes[0].set_title(f"{title_prefix} — ROC curve")

    PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=axes[1])
    axes[1].set_title(f"{title_prefix} — Precision-Recall curve")

    fig.tight_layout()
    _ensure_dir(save_to)
    fig.savefig(save_to, dpi=120)
    plt.close(fig)


def plot_comparison_bar(df: pd.DataFrame, metric: str, save_to: Path) -> None:
    """Bar chart comparing models on a chosen metric."""
    df = df.sort_values(by=metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(df))]
    ax.barh(df["model"], df[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"Model comparison — {metric}")
    ax.invert_yaxis()
    for i, (m, v) in enumerate(zip(df["model"], df[metric])):
        ax.text(v, i, f" {v:.4f}", va="center")
    fig.tight_layout()
    _ensure_dir(save_to)
    fig.savefig(save_to, dpi=120)
    plt.close(fig)


def plot_feature_importance(
    names: Iterable[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature importance",
    save_to: Path | None = None,
) -> pd.DataFrame:
    """Horizontal bar chart for top features by importance."""
    df = pd.DataFrame({"feature": list(names), "importance": importances})
    df = df.sort_values(by="importance", ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * top_n)))
    ax.barh(df["feature"], df["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    if save_to is not None:
        _ensure_dir(save_to)
        fig.savefig(save_to, dpi=120)
    plt.close(fig)
    return df.sort_values(by="importance", ascending=False)
