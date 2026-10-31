"""Custom metrics utilities for credit scoring evaluation."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def _fmt_seconds(seconds: float) -> str:
    """Human readable seconds: 65.2 → ``1m05s``."""
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"0m{s:02d}s"


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute a set of binary classification metrics.

    Returns
    -------
    dict
        Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1`` and
        (if ``y_proba`` is provided) ``roc_auc`` and ``pr_auc``.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        # For binary classification use probability of positive class
        if y_proba.ndim == 2:
            scores = y_proba[:, 1]
        else:
            scores = y_proba
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    return metrics


def detailed_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return a sklearn classification report as a string."""
    return classification_report(y_true, y_pred, digits=4, zero_division=0)


def confusion_to_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return confusion matrix values in a dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
