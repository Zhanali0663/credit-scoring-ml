# Credit Default Prediction

[![CI](https://github.com/Zhanali0663/credit-scoring-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhanali0663/credit-scoring-ml/actions)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/Live_demo-Streamlit-blue?logo=streamlit)](https://credit-scoring-ml-8qjuwktjdkyxrvcjv24uly.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, leakage-free ML pipeline that predicts credit-card default on the UCI *Default of Credit Card Clients* dataset (30k clients). Five model families are benchmarked under identical conditions, with SHAP interpretation, a calibration study, a cost-sensitive threshold analysis, tests, CI and Docker.

**[Live demo](https://credit-scoring-ml-8qjuwktjdkyxrvcjv24uly.streamlit.app)** · [Model card](MODEL_CARD.md) · [Documentation](DOCUMENTATION.md)

## TL;DR

- **Best test ROC AUC: 0.783** (LightGBM and XGBoost tie). Mean over 5 random splits: **0.791 ± 0.007**.
- **Realistic ceiling.** This dataset tops out around 0.79–0.80 AUC on a genuine holdout. Public results of 0.9+ typically come from applying SMOTE before the split or from reporting train-fold metrics.
- **Feature engineering gives a small, stable gain** (+0.005 AUC, positive on 5/5 seeds). Most of the lift over a plain baseline comes from the model choice, not from heavy feature work.
- **Boosted trees beat logistic regression by about 0.03 AUC** (0.783 vs 0.753).

## Results

Stratified 80/20 split (seed 42, test n = 6,000); 5-fold stratified CV on the training part. Per-fold numbers: `reports/metrics/comparison.csv`.

| Model               | CV ROC AUC | Test ROC AUC | PR AUC | F1    | Recall |
|---------------------|-----------:|-------------:|-------:|------:|-------:|
| XGBoost (tuned)     | 0.789 ± 0.006 | 0.783 | 0.563 | 0.545 | 0.604 |
| LightGBM (tuned)    | 0.787 ± 0.006 | 0.783 | 0.563 | 0.538 | 0.637 |
| CatBoost            | 0.780 ± 0.008 | 0.776 | 0.555 | 0.537 | 0.613 |
| Random Forest       | 0.774 ± 0.004 | 0.771 | 0.548 | 0.522 | 0.522 |
| Logistic Regression | 0.765 ± 0.005 | 0.753 | 0.500 | 0.511 | 0.607 |

> **Reading the table.** With n = 6,000 the standard error of test AUC is roughly 0.007, so the top two (and arguably the top three) models are statistically indistinguishable. LightGBM is used as the headline model because of its higher recall and its direct fit with SHAP `TreeExplainer`, not because it "wins".

![Model comparison — test ROC AUC](https://raw.githubusercontent.com/Zhanali0663/credit-scoring-ml/main/reports/figures/comparison_roc_auc.png)

![LightGBM ROC and PR curves on the holdout](https://raw.githubusercontent.com/Zhanali0663/credit-scoring-ml/main/reports/figures/lightgbm_roc_pr.png)

![SHAP summary for the headline model](https://raw.githubusercontent.com/Zhanali0663/credit-scoring-ml/main/reports/figures/lightgbm_shap_summary.png)

### Ablation: what actually helped?

LightGBM only, 5 seeds (`42, 123, 999, 2026, 7`), mean ± std of test AUC. Code: `ablation_full.py`, results: `reports/metrics/ablation.csv`.

| Configuration          | Features | Mean AUC | Std |
|------------------------|---------:|---------:|----:|
| raw + default params   | 23 | 0.7808 | 0.0092 |
| engineered + default   | 61 | 0.7859 | 0.0065 |
| raw + tuned            | 23 | 0.7871 | 0.0096 |
| engineered + tuned     | 61 | 0.7911 | 0.0071 |

Paired differences across seeds: feature engineering **+0.005 ± 0.002**, tuning **+0.006 ± 0.003**; both positive on 5/5 seeds. The effects are small but consistent. Caveat: Optuna ran on seed 42, so tuned results are slightly optimistic for the other seeds.

## Methodology

- **No leakage.** `StandardScaler` lives inside a `Pipeline` and is fitted per CV fold. No resampling happens before the split.
- **Imbalance (~22 % positives)** is handled with `class_weight="balanced"` / `scale_pos_weight`. SMOTE is available as an experiment but excluded from the benchmark.
- **Reproducibility.** Fixed seed, stratified split, shared `random_state` across registry, split and tuning. Verified by an end-to-end test.
- **Tuning.** Optuna with a hard time budget (12 trials, 80 s) for XGBoost and LightGBM; best parameters are stored in `configs/best_params.yaml`. This is a deliberately light search, not an exhaustive one.
- **Features.** 23 raw columns + 38 derived = 61. Groups: volume aggregates, signed `log1p` of money columns, utilisation and payment-to-bill ratios, delinquency depth (counts, streaks, recency-weighted delay), and trends.

## Calibration and decision threshold

Both analyses are **demonstrations**, not production validation.

- **Calibration** (`calibration_check.py`, isotonic via `CalibratedClassifierCV`): Brier score 0.1744 → 0.1289. Note that a large part of this gain simply undoes the probability shift caused by `class_weight="balanced"`, so it should not be read as pure calibration skill. The live demo shows raw, uncalibrated probabilities.
- **Cost-sensitive threshold** (`threshold_optimization.py`): with an assumed 5:1 cost ratio (a missed default costs 5× a false alarm), the best threshold on the test split is about 0.35 instead of 0.50, reducing cost by about 3.6 % (1173 vs 1217). This was selected on the test set and the cost ratio is an assumption; a real deployment needs out-of-fold or validation-set selection and real cost figures.

## Repository layout

```
├── src/
│   ├── data/            # loader (kagglehub + direct-URL fallback)
│   ├── features/        # feature engineering, stratified split
│   ├── models/          # registry, pipelines, CV trainer, Optuna tuning
│   ├── visualization/   # ROC/PR, confusion matrix, importance, SHAP
│   └── utils/           # config, logging, metrics, pipeline orchestrator
├── app/                 # Streamlit demo
├── notebooks/           # EDA, features, modeling, SHAP (outputs committed)
├── tests/               # unit, end-to-end and property-based tests
├── configs/             # paths, tuned hyperparameters
├── reports/             # metrics (csv/json) and figures
└── .github/workflows/   # CI: ruff + pytest, Docker build
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.models.tune_models --budget 80 --trials 12 --models xgboost,lightgbm  # optional, ~2.5 min
python -m src.utils.pipeline          # full benchmark, figures, SHAP (~5 min)
streamlit run app/streamlit_app.py    # demo
docker compose up app                 # demo in a container
pytest -v                             # tests, no network needed
```

The dataset is downloaded automatically on first use; no Kaggle credentials required.

## Limitations

- **One portfolio, one country, 2005.** This is a methodology demonstration, not a deployable scorecard. It would need recalibration on recent lending data and macro covariates.
- **Protected attributes.** `SEX`, `EDUCATION` and `MARRIAGE` are kept for dataset parity and subgroup analysis. A production fair-lending model would exclude them or put them under adverse-impact review.
- **SHAP is not a set of regulatory reason codes.** Adverse-action notices need a mapped, human-readable reason set.
- **Not production infrastructure.** There is no serving API, monitoring, experiment tracking or model registry in this repo.

## References

- Yeh, I.-C. & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
- [UCI ML Repository: Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) · [Kaggle mirror](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

## License

MIT, see [LICENSE](LICENSE).
