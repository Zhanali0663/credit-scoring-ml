---
title: Credit Scoring Demo
emoji: 💳
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Credit Scoring Demo

Predict the probability that a credit-card client defaults next month.

- Dataset: UCI *Default of Credit Card Clients* (30,000 borrowers, ~22% positives).
- Model: LightGBM, 61 features (23 raw + 38 engineered).
- Test ROC AUC: **0.783** on the fixed seed-42 holdout.
- Probabilities are the raw, class-weighted scores. Isotonic calibration and a
  cost-sensitive threshold are reported in the parent repository
  (`reports/metrics/calibration.json`, `reports/metrics/threshold.json`).

This Space is an educational demo, not a lending decision system. Free CPU
Spaces can pause or sleep; run the parent project's app locally with
`streamlit run app/streamlit_app.py`.
