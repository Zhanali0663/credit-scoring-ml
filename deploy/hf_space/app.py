"""Gradio demo of the credit-scoring-ml LightGBM default model."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from credit_features import add_behavioural_features

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "lightgbm.joblib"

# label, default, min, max, step
RAW_FIELDS: dict[str, tuple] = {
    "LIMIT_BAL": ("Credit limit (NT$)", 200_000, 10_000, 800_000, 10_000),
    "AGE": ("Age", 35, 21, 75, 1),
    "SEX": ("Sex (1 = male, 2 = female)", 2, 1, 2, 1),
    "EDUCATION": ("Education (1 grad, 2 university, 3 high school, 4 other)", 2, 1, 4, 1),
    "MARRIAGE": ("Marriage (1 married, 2 single, 3 other)", 1, 1, 3, 1),
    "PAY_0": ("Repayment status, most recent (-1 duly, 0 revolving, 1-9 overdue)", 0, -1, 6, 1),
    "PAY_2": ("Repayment status, month -1", 0, -1, 6, 1),
    "PAY_3": ("Repayment status, month -2", 0, -1, 6, 1),
    "PAY_4": ("Repayment status, month -3", 0, -1, 6, 1),
    "PAY_5": ("Repayment status, month -4", 0, -1, 6, 1),
    "PAY_6": ("Repayment status, month -5", 0, -1, 6, 1),
    "BILL_AMT1": ("Bill amount, most recent (NT$)", 50_000, 0, 600_000, 5_000),
    "BILL_AMT2": ("Bill amount, month -1 (NT$)", 45_000, 0, 600_000, 5_000),
    "BILL_AMT3": ("Bill amount, month -2 (NT$)", 40_000, 0, 600_000, 5_000),
    "BILL_AMT4": ("Bill amount, month -3 (NT$)", 35_000, 0, 600_000, 5_000),
    "BILL_AMT5": ("Bill amount, month -4 (NT$)", 30_000, 0, 600_000, 5_000),
    "BILL_AMT6": ("Bill amount, month -5 (NT$)", 25_000, 0, 600_000, 5_000),
    "PAY_AMT1": ("Payment, most recent (NT$)", 5_000, 0, 200_000, 1_000),
    "PAY_AMT2": ("Payment, month -1 (NT$)", 5_000, 0, 200_000, 1_000),
    "PAY_AMT3": ("Payment, month -2 (NT$)", 5_000, 0, 200_000, 1_000),
    "PAY_AMT4": ("Payment, month -3 (NT$)", 5_000, 0, 200_000, 1_000),
    "PAY_AMT5": ("Payment, month -4 (NT$)", 5_000, 0, 200_000, 1_000),
    "PAY_AMT6": ("Payment, month -5 (NT$)", 5_000, 0, 200_000, 1_000),
}

pipeline = joblib.load(ARTIFACT)
EXPECTED_COLS = list(pipeline.feature_names_in_)


def predict(*values) -> tuple[str, str, pd.DataFrame]:
    """Compute the default probability from the form inputs."""
    client = dict(zip(RAW_FIELDS, values))
    df = add_behavioural_features(pd.DataFrame([client]))[EXPECTED_COLS]

    proba = float(pipeline.predict_proba(df)[0, 1])
    verdict = (
        "⚠️ Likely to default next month (threshold 0.5)"
        if proba >= 0.5
        else "✅ Likely to pay on time (threshold 0.5)"
    )

    derived = df.drop(columns=list(RAW_FIELDS)).iloc[0].round(4).reset_index()
    derived.columns = ["engineered feature", "value"]

    return f"{proba:.1%}", verdict, derived


with gr.Blocks(title="Credit Scoring Demo") as demo:
    gr.Markdown(
        "# 💳 Credit Scoring — default probability\n"
        "LightGBM model (Optuna-tuned) trained on the UCI *Default of Credit Card* "
        "dataset. Test ROC AUC **0.78**. Adjust the borrower profile and press **Predict**. "
        "Demo for methodology illustration only — not a lending decision."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Borrower profile")
            inputs = [
                gr.Number(label=label, value=default, minimum=lo, maximum=hi, step=step, precision=0)
                for label, default, lo, hi, step in RAW_FIELDS.values()
            ]
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            gr.Markdown("### Prediction")
            proba_out = gr.Textbox(label="Default probability")
            verdict_out = gr.Textbox(label="Verdict")
            feats_out = gr.DataFrame(label="Engineered features used by the model")

    btn.click(predict, inputs, [proba_out, verdict_out, feats_out])
    gr.Markdown(
        "Mirrors the [credit-scoring-ml](https://github.com/Zhanali0663/credit-scoring-ml) "
        "project; probabilities are not cost-calibrated. Educational demo only."
    )

if __name__ == "__main__":
    demo.launch()
