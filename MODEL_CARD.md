# Model Card — Credit Scoring Pipeline

## 1. Model Overview
- **Task:** Binary classification — default on next-month credit-card payment
  (`default_payment_next_month`).
- **Dataset:** UCI *Default of Credit Card Clients* (Taiwan, 2005; n = 30,000;
  23 raw features + ID + target).
- **Headline model:** LightGBM, Optuna-tuned (test ROC AUC **0.783**, recall
  **0.637** at the 0.5 threshold).
- **Pipeline version:** 1.0.0 (`pyproject.toml`).
- **Training date:** 2026-11-01.
- **Code repository:** https://github.com/Zhanali0663/credit-scoring-ml

## 2. Intended Use
- **Purpose:** Educational / portfolio demonstration of an end-to-end ML
  pipeline with ablation, calibration, threshold analysis and interpretability.
- **Not intended for:** Real lending decisions without recalibration on recent
  data, fair-lending review and regulatory compliance checks.
- **Users:** Data scientists, educators and interviewers evaluating methodology
  and reproducibility.

## 3. Training Data
- **Source:** Kaggle / UCI ML Repository
  (`uciml/default-of-credit-card-clients-dataset`), downloaded automatically by
  `src/data/loader.py`.
- **Size:** 30,000 records × 25 columns (23 raw features + `ID` + target).
- **Split:** Stratified 80/20 (`random_state=42`); 5-fold stratified CV on the
  24,000-row training part.
- **Target prevalence:** ~22.1% positive.
- **Demographics:** `SEX`, `EDUCATION`, `MARRIAGE` are kept for dataset parity
  and subgroup analysis; they are not treated as causal drivers.

## 4. Model Architecture & Hyperparameters
- **Pipeline:** `Pipeline(StandardScaler → estimator)`. The scaler is fitted
  inside each CV fold; no resampling is applied before the split.
- **Feature engineering:** 23 raw + 38 derived = 61 features. Groups: volume
  aggregates, signed `log1p` of monetary columns, utilisation and
  payment-to-bill ratios, delinquency counts/streaks, recency-weighted delay and
  trends. See `src/features/engineering.py`.
- **Class imbalance:** `class_weight="balanced"` (LightGBM/CatBoost/RF/LR) or
  `scale_pos_weight=3.0` (XGBoost). No SMOTE.
- **Tuning:** Optuna. The committed run used a hard budget of 12 trials / 80 s
  per model for **XGBoost and LightGBM only**. Best parameters are in
  `configs/best_params.yaml`. CatBoost, Random Forest and Logistic Regression
  use documented registry defaults.

## 5. Performance Metrics (Test Set, n = 6,000; raw, uncalibrated scores at 0.5)

| Model | ROC AUC | PR AUC | F1 | Recall | Precision | CV ROC AUC (± std) |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM (tuned) | **0.783** | 0.563 | 0.538 | **0.637** | 0.466 | 0.787 ± 0.006 |
| XGBoost (tuned) | 0.783 | 0.563 | 0.545 | 0.604 | 0.497 | 0.789 ± 0.006 |
| CatBoost | 0.776 | 0.555 | 0.537 | 0.613 | 0.478 | 0.780 ± 0.008 |
| Random Forest | 0.771 | 0.548 | 0.522 | 0.522 | 0.521 | 0.774 ± 0.004 |
| Logistic Regression | 0.753 | 0.500 | 0.511 | 0.607 | 0.442 | 0.765 ± 0.005 |

*Accuracy is reported in `reports/metrics/comparison.csv` for completeness
(~78% for a constant "never default" classifier) but is not the primary metric.

### Ablation (LightGBM, 5 seeds)

| Configuration | Features | Mean test AUC | Std |
|---|---:|---:|---:|
| raw + defaults | 23 | 0.7822 | 0.0077 |
| engineered + defaults | 61 | 0.7822 | 0.0068 |
| raw + tuned | 23 | 0.7870 | 0.0073 |
| engineered + tuned | 61 | 0.7914 | 0.0066 |

Feature engineering with defaults: +0.0001 ± 0.0032 (2/5 seeds). Feature
engineering with tuned params: +0.0044 ± 0.0018 (5/5). Tuning: +0.0049 ± 0.0015
(raw) and +0.0091 ± 0.0035 (engineered), 5/5 seeds. Source:
`reports/metrics/ablation.csv` and `ablation_summary.json`.

## 6. Interpretability
- **Method:** SHAP `TreeExplainer` for tree models. SHAP values are computed on
  the **standardized** matrix that the estimator was trained on; plots display
  the original feature values.
- **Key drivers:** `DELAY_WEIGHTED`, `MAX_DELAY_SEVERITY`, `PAY_0`,
  `UTILIZATION_MAX`, `COVERAGE_MIN`, `N_MONTHS_DELAY`, `LIMIT_BAL`.
- **Artifacts:** `reports/figures/lightgbm_shap_summary.png`,
  `lightgbm_shap_bar.png`; `notebooks/04_interpretation.ipynb`.

## 7. Fairness & Ethics
- **Demographics included:** `SEX`, `EDUCATION`, `MARRIAGE` carried for dataset
  parity. Behavioural history dominates their SHAP contribution.
- **Not computed:** disparate impact, equalized odds or any other fairness
  metric. A production scorecard would exclude or explicitly review these
  attributes under fair-lending rules.
- **Regulatory gap:** SHAP outputs are not mapped to formal adverse-action
  reason codes.

## 8. Calibration & Probability Use
- **Raw probabilities are not calibrated.** Class weighting inflates the mean
  predicted probability to 0.426 against an observed rate of 0.221 on the test
  set.
- **Isotonic calibration** (`CalibratedClassifierCV`, 5 internal folds) reduces
  Brier score from **0.1806 to 0.1342**, log loss from **0.5465 to 0.4273**, and
  ECE from **0.2047 to 0.0111**, while leaving ROC AUC unchanged (0.7828).
- **Cost-sensitive threshold:** with a 5:1 missed-default-to-false-alarm cost
  ratio, the threshold selected on out-of-fold predictions is **0.149**. On the
  calibrated holdout it reduces cost from 4,532 to 3,319 (−26.8%) and raises
  recall from 0.351 to 0.805. The 0.637 recall in section 5 is the raw,
  class-weighted model at 0.5; calibrated probabilities make 0.5 much stricter.
  The cost ratio is an explicit assumption.
- Sources: `reports/metrics/calibration.json`, `reports/metrics/threshold.json`.

## 9. Operational & Deployment Context
- **Artifacts:** `models_artifacts/*.joblib` are produced by
  `python -m src.utils.pipeline` and are gitignored (the Random Forest artifact
  is ~200 MB).
- **Local demo:** `streamlit run app/streamlit_app.py`; reads the saved
  pipelines and exposes the raw probability plus a configurable threshold.
- **Docker:** `Dockerfile` + `docker-compose.yml`; `docker compose up app`.
- **Optional Space:** `deploy/hf_space/` (Docker + Gradio). Free-tier Spaces can
  pause or sleep, so no public URL is advertised in the README.
- **Monitoring / drift detection:** not implemented. Recommended next step:
  Evidently or a PSI/KS script (`IMPROVEMENT_PLAN.md`).

## 10. Limitations & Out-of-Scope
1. **Single portfolio / single year:** Taiwan 2005; not deployable without
   recalibration on recent lending data.
2. **No macroeconomic covariates:** unemployment, interest rates and regional
   stress are omitted.
3. **Threshold assumptions:** the 5:1 cost ratio is illustrative.
4. **No post-deployment monitoring:** drift, data-quality checks and retraining
   triggers are not implemented.
5. **No fairness metrics computed:** see section 7.
6. **Partial tuning:** only XGBoost and LightGBM were tuned.
7. **SHAP is not a reason-code system:** see section 7.

## 11. Reproducibility Checklist
- [x] Fixed split (`random_state=42`), verified by tests.
- [x] Deterministic model seeds in the registry; ablation varies both split and
      estimator seeds.
- [x] `StandardScaler` fitted inside each pipeline/CV fold.
- [x] `Dockerfile` pins the Python runtime; dependency ranges are bounded.
- [x] `reports/metrics/comparison.csv` / `.json` regenerated automatically.
- [x] Ablation, calibration and threshold scripts write machine-readable JSON.
- [x] Notebooks 01–04 have executed outputs committed.
- [x] Unit, invariant and end-to-end tests run without network access.

## 12. References
- Yeh, I.-C., & Lien, C.-H. (2009). *The comparisons of data mining techniques
  for the predictive accuracy of probability of default of credit card clients.*
  Expert Systems with Applications, 36(2), 2473–2480.
- Dataset: [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- Code: https://github.com/Zhanali0663/credit-scoring-ml
