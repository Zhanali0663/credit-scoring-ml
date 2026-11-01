# Model Card — Credit Scoring Default Prediction

> This is a short template kept next to the pipeline utilities. The maintained,
> experiment-specific card is [`MODEL_CARD.md`](../../MODEL_CARD.md) at the
> repository root.

## Model Details

| Field | Value |
|---|---|
| Model type | Auto-evaluated: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost |
| Training objective | Binary classification (default next month) |
| Input | 23 UCI credit-card features + 38 engineered behavioural features = 61 |
| Output | Probability between 0 (`pay on time`) and 1 (`default`) |
| Version | 1.0.0 |
| Language | Python 3.11+ |

## Intended Use

- Educational / portfolio demonstration of an end-to-end tabular ML pipeline.
- Studying the relative strengths of interpretable vs boosted-tree models on
  imbalanced tabular data.
- Template for applying the same pipeline to a real lending dataset.

## Out-of-Scope Use Cases

- Production decisions for real borrowers without retraining on domain data.
- Substitution for a regulated scorecard.

## Training Data

- Source: UCI ML Repository — *Default of Credit Card Clients* (Taiwan, 2005),
  mirrored on Kaggle as `uciml/default-of-credit-card-clients-dataset`.
- Size: 30,000 records × 23 raw features (plus ID and the target).
- Target distribution: ~22% positives (defaults).

## Evaluation Data

- Stratified held-out test split (20%, seed 42) from the same dataset.
- Cross-validation: 5-fold stratified on the training portion only.

## Metrics (final test set)

See `reports/metrics/comparison.csv` for the exact values of each trained model
for ROC AUC, PR AUC, F1, precision, recall, accuracy, and CV summary
(`cv_mean_*`, `cv_std_*` columns). See `reports/metrics/ablation_summary.json`,
`calibration.json` and `threshold.json` for the ablation, calibration and
cost-sensitive threshold results.

## Limitations

- Dataset is country and epoch specific; not calibrated for today's populations.
- Raw probabilities are class-weighted and **not** calibrated. Isotonic
  calibration is available in `src/models/calibration.py`; it reduces the Brier
  score from 0.1806 to 0.1342 on the fixed holdout.
- SHAP values identify main drivers, not regulatory reason codes.

## Ethical Considerations

- The dataset includes demographic attributes (`SEX`, `EDUCATION`, `MARRIAGE`).
  Using them directly in production raises fairness and legal concerns; this
  project keeps them only to reproduce the historical dataset and to enable
  subgroup experiments.
- No fairness metrics are computed. Run subgroup audits before any real
  deployment.
