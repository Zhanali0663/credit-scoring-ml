"""Live Streamlit preview — Credit Scoring ML.

Run locally::

    streamlit run app/streamlit_app.py --server.port=8501

The demo reads the pipelines produced by ``python -m src.utils.pipeline`` from
``models_artifacts/``. Probabilities are the raw model scores; the calibrated
version and the cost-sensitive threshold study live in ``reports/metrics/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from app.predict import available_models, load_model, predict_one
from src.visualization.shap_utils import compute_shap_values

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring — Live Preview",
    page_icon="💳",
    layout="wide",
)

FEATURE_DEFAULTS = {
    "LIMIT_BAL": 200_000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 50_000,
    "BILL_AMT2": 45_000,
    "BILL_AMT3": 40_000,
    "BILL_AMT4": 35_000,
    "BILL_AMT5": 30_000,
    "BILL_AMT6": 25_000,
    "PAY_AMT1": 5_000,
    "PAY_AMT2": 5_000,
    "PAY_AMT3": 5_000,
    "PAY_AMT4": 5_000,
    "PAY_AMT5": 5_000,
    "PAY_AMT6": 5_000,
}

TREE_MODELS = {"lightgbm", "xgboost", "catboost", "random_forest"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, object]:
    models: dict[str, object] = {}
    for name in available_models():
        try:
            models[name] = load_model(name)
        except Exception as exc:  # pragma: no cover - defensive UI path
            st.warning(f"Failed to load {name}: {exc}")
    return models


def local_contributions(pipeline: object, features: pd.DataFrame, top_n: int = 8) -> pd.Series | None:
    """Return the largest absolute SHAP contributions for one borrower."""
    try:
        values, _ = compute_shap_values(pipeline, features)
    except Exception:  # SHAP is only available for tree models
        return None
    series = pd.Series(values[0], index=features.columns)
    return series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def main() -> None:
    st.title("💳 Credit Scoring — Live Preview")
    st.markdown(
        "End-to-end ML pipeline for predicting credit-card default (UCI dataset). "
        "Adjust borrower features, pick a model, and see the predicted probability."
    )

    models = load_models()
    if not models:
        st.error("No trained models found in `models_artifacts/`. Run `python -m src.utils.pipeline` first.")
        st.stop()

    available = list(models)
    col_input, col_result = st.columns([2, 1])

    with col_input:
        st.subheader("Borrower profile")
        with st.form("borrower_form"):
            cols = st.columns(3)
            user_input: dict[str, int | float] = {}
            for i, (feat, default) in enumerate(FEATURE_DEFAULTS.items()):
                with cols[i % 3]:
                    if isinstance(default, int):
                        user_input[feat] = st.number_input(feat, value=default, step=1)
                    else:
                        user_input[feat] = st.number_input(feat, value=float(default), step=1000.0)
            submitted = st.form_submit_button("Predict", type="primary")

    with col_result:
        st.subheader("Model")
        model_name = st.selectbox("Choose model", available, index=0)
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
        st.caption(
            "The 0.5 cut-off is illustrative. See `reports/metrics/threshold.json` "
            "for an out-of-fold, cost-sensitive threshold."
        )

    if not submitted:
        st.info("Fill the borrower form and press **Predict**.")
        return

    try:
        result = predict_one(
            user_input,
            model_name=model_name,
            threshold=threshold,
            pipeline=models[model_name],
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    proba = result["default_proba"]
    pred = result["prediction"]
    verdict_text = "HIGH RISK — likely to default" if pred == 1 else "LOW RISK — likely to pay on time"

    st.divider()
    st.subheader("Prediction result")
    metric_col, progress_col = st.columns([1, 2])
    with metric_col:
        st.metric("Default probability", f"{proba:.2%}")
    with progress_col:
        st.progress(min(max(proba, 0.0), 1.0))

    if pred == 1:
        st.error(f"⚠️ Verdict: {verdict_text}")
    else:
        st.success(f"✅ Verdict: {verdict_text}")

    contributions = None
    if model_name in TREE_MODELS:
        contributions = local_contributions(models[model_name], result["features"])

    if contributions is not None:
        with st.expander("Top SHAP contributions (this borrower)"):
            st.caption(
                "Positive values push the default probability up; negative values push it down. "
                "SHAP values are computed on the model's scaled input and shown per raw feature."
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "feature": contributions.index,
                        "SHAP value": contributions.to_numpy().round(4),
                    }
                )
            )

    with st.expander("Engineered behavioural features (used by model)"):
        engineered = result["features"].iloc[0]
        extra = engineered.drop(labels=list(FEATURE_DEFAULTS), errors="ignore").round(4)
        st.dataframe(pd.DataFrame({"feature": extra.index, "value": extra.to_numpy()}))
        st.json({k: float(v) for k, v in extra.to_dict().items()})


if __name__ == "__main__":
    main()
