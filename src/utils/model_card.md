# Model Card — Credit Scoring Default Prediction

## Model Details

| Field | Value |
|---|---|
| Model type | Auto-evaluated: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost |
| Training objective | Binary classification (default next month) |
| Input | 23 UCI credit-card features + 12 engineered behavioural features |
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
- Size: 30,000 records × 23 raw features.
- Target distribution: ~22% positives (defaults).

## Evaluation Data

- Stratified held-out test split (20%) from the same dataset.
- Cross-validation: 5-fold stratified on the training portion only.

## Metrics (final test set)

See `reports/metrics/comparison.csv` for the exact values of each trained model
for ROC AUC, PR AUC, F1, precision, recall, accuracy, and CV summary
(`cv_mean_*`, `cv_std_*` columns).

## Limitations

- Dataset is country & epoch specific; not calibrated for today’s populations.
- Probabilities are not calibrated for quoting to end users.
- SHAP values identify main drivers, not regulatory reason codes.

## Ethical Considerations

- The dataset includes demographic attributes (`SEX`, `EDUCATION`, `MARRIAGE`).
  Using them directly in production raise fairness and legal concerns;
  this project keeps them only to reproduce the historical dataset and to
  enable subgroup experiments.
- Users should run subgroup-fairness audits before any real deployment.
