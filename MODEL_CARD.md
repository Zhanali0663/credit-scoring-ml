# Model Card — Credit Scoring Pipeline

## 1. Model Overview
- **Task:** Binary classification — default on next-month credit card payment (target: `default_payment_next_month`)
- **Dataset:** UCI Default of Credit Card Clients (Taiwan, 2005; n=30,000; 23 raw features)
- **Headline model:** LightGBM (tuned via Optuna; test ROC AUC 0.783; recall 0.637)
- **Pipeline version:** 1.0.0 (see `pyproject.toml`)
- **Training date:** 2026-10-31
- **Code repository:** https://github.com/Zhanali0663/credit-scoring-ml

## 2. Intended Use
- **Purpose:** Educational / portfolio demonstration of end-to-end ML pipeline with interpretability.
- **Not intended for:** Real lending decisions without recalibration on recent data, fair-lending review, and regulatory compliance checks.
- **Users:** Data scientists, educators, interviewers evaluating methodology rigor.

## 3. Training Data
- **Source:** Kaggle / UCI ML Repository (`uciml/default-of-credit-card-clients-dataset`)
- **Size:** 30,000 records × 25 columns (23 raw + ID + target)
- **Split:** Stratified 80/20 fixed split (`random_state=42`); 5-fold CV on training set for stability
- **Target prevalence:** ~22% positive (`default_payment_next_month`)
- **Excluded from training:** No external macroeconomic covariates; demographics (`SEX`, `EDUCATION`, `MARRIAGE`) kept for dataset parity but noted as low-independent-signal features.

## 4. Model Architecture & Hyperparameters
- **Pipeline:** `Pipeline` with `StandardScaler` (fit inside each CV fold) + model.
- **Feature engineering:** 38 derived features (signed-log1p on monetary columns, recency-weighted delay index, utilisation ratios, delay streaks, trends). See `src/features/engineering.py`.
- **Tuning:** Optuna (`--budget 80 --trials 12`) with `scale_pos_weight` / `class_weight="balanced"` instead of SMOTE.
- **Best params:** `configs/best_params.yaml`

## 5. Performance Metrics (Test Set, n=6,000; Fixed Split)

| Model | ROC AUC | PR AUC | F1 | Recall | Precision | CV ROC AUC (±std) |
|---|---|---|---|---|---|---|
| XGBoost (tuned) | 0.783 | 0.563 | 0.545 | 0.604 | 0.495 | 0.789 ±0.006 |
| LightGBM (tuned) | **0.783** | 0.563 | 0.538 | **0.637** | 0.464 | 0.787 ±0.006 |
| CatBoost | 0.776 | 0.555 | 0.537 | 0.613 | 0.476 | 0.780 ±0.008 |
| Random Forest | 0.771 | 0.548 | 0.522 | 0.522 | 0.522 | 0.774 ±0.004 |
| Logistic Regression | 0.753 | 0.500 | 0.511 | 0.607 | 0.439 | 0.765 ±0.005 |

*Note:* Accuracy is reported for completeness (~78% for a naive "never default" classifier) but is not the primary metric due to class imbalance.

## 6. Interpretability
- **Method:** SHAP (`TreeExplainer` for tree-based models; `KernelExplainer` available for LR).
- **Key drivers:** `DELAY_WEIGHTED` (recency-weighted delinquency), `UTILIZATION_MEAN`, `N_MONTHS_DELAY`, `LIMIT_BAL` (signed-log transformed), `PAY_TO_BILL_RATIO`.
- **Artifacts:** `reports/figures/lightgbm_shap_summary.png`; `notebooks/04_interpretation.ipynb`.

## 7. Fairness & Ethics
- **Demographics included:** `SEX`, `EDUCATION`, `MARRIAGE` carried for dataset parity and subgroup analysis; independent predictive signal is dominated by behavioural history.
- **Limitations acknowledged:** No adverse-impact metrics (disparate impact, equalized odds) computed in v1. A production scorecard would exclude or treat demographics under fair-lending review.
- **Regulatory gap:** SHAP outputs are not mapped to formal adverse-action reason codes. This is noted explicitly and left out of scope.

## 8. Calibration & Probability Use
- **Status:** Probabilities are **not calibrated** (`README.md` line 104; `DOCUMENTATION.md` section 4).
- **Consequence:** The 0.5 threshold in the Streamlit demo is illustrative only, not business-optimal. Production deployment requires isotonic/sigmoid calibration (`CalibratedClassifierCV`) and a cost-sensitive threshold selection.
- **Brier score:** Not reported in v1.

## 9. Operational & Deployment Context
- **Containerisation:** `Dockerfile` + `docker-compose.yml` provided.
- **CI:** `.github/workflows/ci.yml` (ruff + pytest) — referenced in README badges.
- **Live demo:** `https://credit-scoring-ml-8qjuwktjdkyxrvcjv24uly.streamlit.app` (temporary Streamlit Community Cloud URL).
- **Monitoring / drift detection:** Not implemented in v1. Recommended next step: Evidently or custom PSI/KS script (`IMPROVEMENT_PLAN.md` item 3).

## 10. Limitations & Out-of-Scope
1. **Single portfolio / single year:** Taiwan 2005; no cyclical stress periods. Not deployable as a production scorecard without recalibration on recent lending data.
2. **No macroeconomic covariates:** Unemployment, interest rates, regional stress omitted.
3. **No cost-sensitive threshold:** 0.5 is arbitrary; business-optimal threshold requires a cost matrix (false alarm cost vs missed default cost).
4. **No post-deployment monitoring:** Drift, data quality checks, and model retraining triggers not implemented.
5. **No fairness metrics computed:** See section 7.
6. **Probabilities not calibrated:** See section 8.

## 11. Reproducibility Checklist
- [x] Fixed split (`random_state=42`) verified by test.
- [x] Deterministic seeds (`np.random.default_rng(42)` in synthetic tests; model seeds in registry).
- [x] `Dockerfile` pins runtime.
- [x] `models_artifacts/*.joblib` saved per comparison run.
- [x] `reports/metrics/comparison.csv` and `.json` saved automatically.
- [x] End-to-end smoke test runs without network (`tests/test_pipeline_end_to_end.py`).

## 12. References
- Yeh, I.-C., & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
- Dataset: [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- Code: https://github.com/Zhanali0663/credit-scoring-ml
