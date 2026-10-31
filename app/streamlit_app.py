"""Live Streamlit preview — Credit Scoring ML.

Run locally:
    streamlit run app/streamlit_app.py --server.port=8501

Uses models from ``models_artifacts/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src.features.engineering import add_behavioural_features
from src.utils.config import get_project_root

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring — Live Preview",
    page_icon="💳",
    layout="wide",
)

# ------------------------------------------------------------------
# Optional LLM explanation (OpenRouter) — read from secrets/env
# ------------------------------------------------------------------
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "") if hasattr(st, "secrets") else ""

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

MODEL_NAMES = [
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "logistic_regression",
]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    artifacts_dir = get_project_root() / "models_artifacts"
    models = {}
    for path in sorted(artifacts_dir.glob("*.joblib")):
        try:
            models[path.stem] = joblib.load(path)
        except Exception as exc:
            st.warning(f"Failed to load {path.name}: {exc}")
    return models, artifacts_dir


def predict(client_dict: dict, model_name: str, artifacts_dir: Path):
    pipeline = joblib.load(artifacts_dir / f"{model_name}.joblib")
    df = pd.DataFrame([client_dict])
    df_eng = add_behavioural_features(df)

    expected_cols = (
        list(pipeline.feature_names_in_)
        if hasattr(pipeline, "feature_names_in_")
        else list(df_eng.columns)
    )
    df_eng = df_eng[expected_cols]

    proba = float(pipeline.predict_proba(df_eng)[0, 1])
    pred = int(pipeline.predict(df_eng)[0])
    return {"model": model_name, "default_proba": proba, "prediction": pred, "features": df_eng}


def explain_with_llm(proba: float, verdict: str, client_dict: dict) -> str:
    if not OPENROUTER_API_KEY:
        return ""
    try:
        import requests
        prompt = (
            f"A credit-scoring model predicted default probability {proba:.1%} "
            f"({verdict}). Client profile: {client_dict}. "
            f"Explain in 2-3 sentences why this risk level was assigned and what factors drive it."
        )
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"LLM explanation failed: {exc}"


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def main():
    st.title("💳 Credit Scoring — Live Preview")
    st.markdown(
        "End-to-end ML pipeline for predicting credit-card default (UCI dataset). "
        "Adjust borrower features, pick a model, and see the predicted probability."
    )

    models, artifacts_dir = load_models()
    if not models:
        st.error("No trained models found in `models_artifacts/`. Run the pipeline first.")
        st.stop()

    available_models = [m for m in MODEL_NAMES if m in models] or list(models.keys())

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
        model_name = st.selectbox("Choose model", available_models, index=0)
        st.caption(f"Trained on UCI Credit Card Default. ROC AUC ~0.78 (LightGBM).")

    if not submitted:
        st.info("Fill the borrower form and press **Predict**.")
        return

    try:
        result = predict(user_input, model_name, artifacts_dir)
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

    # Engineered features expander
    with st.expander("Engineered behavioural features (used by model)"):
        extra = result["features"].drop(columns=list(FEATURE_DEFAULTS.keys())).iloc[0].round(4)
        st.dataframe(pd.DataFrame({"feature": extra.index, "value": extra.values}))
        st.json({k: float(v) for k, v in extra.to_dict().items()})

    # Optional LLM explanation
    if st.checkbox("Show AI explanation (OpenRouter)"):
        with st.spinner("Generating explanation..."):
            explanation = explain_with_llm(proba, verdict_text, user_input)
        if explanation:
            st.info(explanation)
        else:
            st.warning("No API key configured. Add `OPENROUTER_API_KEY` to `.streamlit/secrets.toml`.")


if __name__ == "__main__":
    main()
