"""Streamlit demo application for the credit scoring project.

Run locally::

    streamlit run app/streamlit_app.py

It expects the trained pipelines to be present in ``models_artifacts/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from app.predict import predict_one
from src.features.engineering import add_behavioural_features
from src.utils.config import get_project_root


FEATURE_DEFAULTS: Dict[str, int | float] = {
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

MODEL_NAMES = ["lightgbm", "xgboost", "catboost", "random_forest", "logistic_regression"]


def main() -> None:
    st.set_page_config(
        page_title="Credit Scoring Demo",
        page_icon="💳",
        layout="wide",
    )

    st.title("💳 Credit Scoring — Default Probability")
    st.markdown(
        "End-to-end ML pipeline for predicting whether a credit-card client "
        "will default on their next payment. Adjust the inputs on the left and "
        "the model on the right to see the predicted probability."
    )

    artifacts_dir = get_project_root() / "models_artifacts"
    available = [p.stem for p in artifacts_dir.glob("*.joblib")]
    if not available:
        st.error("No trained models found in models_artifacts/. Run the pipeline first.")
        st.stop()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Client features")
        with st.form("client_form"):
            cols = st.columns(3)
            user_input: Dict[str, float] = {}
            for i, (feat, default) in enumerate(FEATURE_DEFAULTS.items()):
                with cols[i % 3]:
                    if isinstance(default, int):
                        user_input[feat] = st.number_input(feat, value=default, step=1)
                    else:
                        user_input[feat] = st.number_input(feat, value=float(default), step=1000.0)
            submitted = st.form_submit_button("Predict", type="primary")

    with col_right:
        st.subheader("Model")
        model_name = st.selectbox("Choose a model", [m for m in MODEL_NAMES if m in available] or available)
        st.caption("Models are trained using the UCI Credit Card Default dataset.")

    if not submitted:
        st.info("Fill in the form and click **Predict** to see results.")
        return

    try:
        result = predict_one(user_input, model_name=model_name, artifacts_dir=artifacts_dir)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    proba = result["default_proba"]
    pred = result["prediction"]
    st.subheader("Prediction")
    st.metric("Default probability", f"{proba:.2%}")
    st.progress(min(max(proba, 0.0), 1.0))
    if pred == 1:
        st.error("Risk verdict: ⚠️ HIGH RISK — likely to default.")
    else:
        st.success("Risk verdict: ✅ LOW RISK — likely to pay on time.")

    with st.expander("Engineered behavioural features"):
        df = pd.DataFrame([user_input])
        eng = add_behavioural_features(df)
        extra = eng.drop(columns=list(FEATURE_DEFAULTS.keys())).iloc[0].to_dict()
        st.json({k: float(v) for k, v in extra.items()})


if __name__ == "__main__":
    main()
