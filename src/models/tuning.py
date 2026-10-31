"""Time-boxed Optuna hyperparameter tuning for boosted-tree models.

Progress is streamed to stderr/stdout with elapsed time and an ETA so the
operator always knows how much work is left, and epoch-by-epoch messages land
in the log file (``reports/metrics/tuning_log.txt`` when run via the CLI).
"""
from __future__ import annotations

import json
import time
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.utils.logging import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class _Timer:
    """Tracks elapsed and remaining seconds against a strict time budget."""

    def __init__(self, budget_seconds: float) -> None:
        self.budget = float(budget_seconds)
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def remaining(self) -> float:
        return max(self.budget - self.elapsed, 0.0)

    def out_of_time(self) -> bool:
        return self.remaining <= 0


def _fmt(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _xgb_objective(X: pd.DataFrame, y: pd.Series):
    """Return an Optuna objective for XGBoost on 3-fold CV ROC AUC."""
    from xgboost import XGBClassifier

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        }
        scores: list[float] = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for tr_idx, va_idx in skf.split(X, y):
            model = XGBClassifier(
                **params,
                scale_pos_weight=scale_pos,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = model.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[va_idx], proba))
        return float(np.mean(scores))

    return objective


def _lgb_objective(X: pd.DataFrame, y: pd.Series):
    """Return an Optuna objective for LightGBM on 3-fold CV ROC AUC."""
    from lightgbm import LGBMClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
        }
        scores: list[float] = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for tr_idx, va_idx in skf.split(X, y):
            model = LGBMClassifier(
                **params,
                objective="binary",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = model.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[va_idx], proba))
        return float(np.mean(scores))

    return objective


def tune_model(
    model_name: Literal["xgboost", "lightgbm"],
    X: pd.DataFrame,
    y: pd.Series,
    budget_seconds: float = 150,
    seed: int = 42,
    max_trials: int = 25,
) -> dict[str, Any]:
    """Run a time-boxed Optuna study for one model family.

    Prints human-readable progress after every trial: trial index, best CV
    ROC AUC so far, elapsed and remaining time against the budget. Stops
    either when the time budget is exhausted or ``max_trials`` is reached —
    whichever comes first.
    """
    assert model_name in {"xgboost", "lightgbm"}, "Tuning is only wired for boosted trees"

    study_name = f"study_{model_name}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    objective = _xgb_objective(X, y) if model_name == "xgboost" else _lgb_objective(X, y)
    timer = _Timer(budget_seconds)

    logger.info(
        "[%s] Starting Optuna tuning | budget=%s | max_trials=%d | trials will be logged one by one",
        model_name.upper(),
        _fmt(timer.budget),
        max_trials,
    )

    trial_no = 0
    while not timer.out_of_time() and trial_no < max_trials:
        remaining = timer.remaining
        trial_no += 1
        t0 = time.perf_counter()
        try:
            # Strict together with time budget: cap one trial to 1/3 of remaining time
            per_trial_timeout = min(remaining, max(remaining / 3.0, 5.0))
            study.optimize(
                objective,
                n_trials=1,
                timeout=per_trial_timeout,
                gc_after_trial=True,
            )
        except optuna.exceptions.TrialPruned:
            pass
        except Exception as exc:  # noqa: BLE001 - keep tuning alive when a trial fails
            logger.warning("[%s] Trial %d failed: %s", model_name.upper(), trial_no, exc)
            continue

        trial = study.best_trial
        if trial.value is not None and trial.value > float(study.best_value if study.best_trial.value is not None else float("-inf")):
            star = " *NEW BEST*"
        else:
            star = ""

        logger.info(
            "[%s] trial %3d/%d | best CV AUC %.4f%s | trial %s | elapsed %s | remaining %s",
            model_name.upper(),
            trial_no,
            max_trials,
            study.best_value if study.best_trial.value is not None else float("nan"),
            star,
            _fmt(time.perf_counter() - t0),
            _fmt(timer.elapsed),
            _fmt(timer.remaining),
        )

    if not study.trials:
        logger.warning("[%s] No completed trials within budget", model_name.upper())
        return {
            "model": model_name,
            "trials": 0,
            "best_cv_auc": float("nan"),
            "best_params": {},
        }

    best_params = dict(study.best_params)
    logger.info(
        "[%s] DONE | trials=%d | best CV AUC %.4f | best params=%s",
        model_name.upper(),
        len(study.trials),
        study.best_value,
        json.dumps(best_params),
    )
    return {
        "model": model_name,
        "trials": len(study.trials),
        "best_cv_auc": float(study.best_value),
        "best_params": best_params,
    }
