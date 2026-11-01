# Method Notes — Metrics, Ablation, Calibration and Limitations

How to read `reports/metrics/`, what each experiment measures, and where the
methodology has honest gaps.

## 1. Metric glossary

| Metric | Purpose | Why it matters for credit scoring |
|---|---|---|
| ROC AUC | Threshold-free ranking quality | Headline scorecard metric: how well a random defaulter ranks above a random payer |
| PR AUC | Precision vs recall area | More informative than ROC AUC at ~22% positives |
| F1 | Balance of precision and recall | Single number at a chosen threshold |
| Precision | Spurious flags | Every false alarm creates friction and review cost |
| Recall | Caught defaulters | Every miss is principal at risk |
| Brier score | Mean squared probability error | Measures calibration, not just ranking |
| Log loss | Penalised probability error | Sensitive to confident mistakes |
| ECE | Expected calibration error | Average gap between predicted and observed rates |

Accuracy is reported for completeness but is the least informative metric here:
a constant "never default" classifier scores ~78% accuracy.

## 2. How to read the numbers

Results are written per model with two views:

- `roc_auc`, `pr_auc`, `f1`, ... — held-out test set (6,000 clients).
- `cv_mean_*`, `cv_std_*` — 5-fold stratified cross-validation on the 24,000-row
  training portion, showing stability rather than a lucky single split.

What to trust:

- `cv_std` below ~0.01 means the model is stable across folds.
- A CV mean slightly above test ROC AUC with the same seed is normal. Gaps above
  ~0.015 would suggest overfitting or a buggy split.

Reference points on this dataset (reproducible from this repo):

| Setup | ROC AUC |
|---|---|
| Logistic regression, 61 features | ~0.753 test |
| Random forest, 61 features | ~0.771 test |
| Boosting, defaults + 38 features | ~0.782 mean over 5 seeds |
| Boosting, tuned + engineered | **~0.791 mean / 0.783 test** |
| Claims of 0.90+ | often reflect resampling before the split or train-fold metrics; verify the protocol |

## 3. Design decisions and trade-offs

- **Class weighting over SMOTE.** Balanced `class_weight` /
  `scale_pos_weight` are deterministic and keep validation bounds honest. SMOTE
  applied before the split leaks synthetic neighbours across the boundary;
  applied inside each fold it is valid, but it is not used anywhere in this
  benchmark. Clone-and-resample is deliberately out of scope.
- **Signed log1p for money columns.** Raw amounts span several orders of
  magnitude. The transform tames outliers for logistic regression and gives
  boosting more stable split points.
- **Recency-weighted delay index.** Payment status of the newest month is the
  strongest single predictor; `DELAY_WEIGHTED` encodes this grading directly
  rather than hoping the model learns it. The dataset has six observed months
  (`PAY_0`, `PAY_2` … `PAY_6`); there is no seventh month in the source data.
- **Optuna with a hard budget.** Tuning time is capped (`--budget` / `--trials`),
  so the search cannot run away. Only XGBoost and LightGBM are tuned.
- **Calibration as a separate step.** The raw model is deliberately
  class-weighted, which improves ranking at the 0.5 cut-off but distorts
  probabilities. Isotonic calibration is fitted inside the training set and
  evaluated once on the holdout.
- **Threshold selected out-of-fold.** The cost-sensitive threshold is chosen on
  out-of-fold calibrated predictions, not on the test set, then frozen for one
  holdout evaluation.

## 4. Ablation: feature engineering vs tuning

`src/models/ablation.py` runs four configurations on five stratified 80/20
splits with LightGBM:

| Configuration | Features | Mean test AUC | Std |
|---|---:|---:|---:|
| raw + defaults | 23 | 0.7822 | 0.0077 |
| engineered + defaults | 61 | 0.7822 | 0.0068 |
| raw + tuned | 23 | 0.7870 | 0.0073 |
| engineered + tuned | 61 | 0.7914 | 0.0066 |

Paired differences:

| Comparison | Mean Δ | Std | Seeds improved |
|---|---:|---:|---:|
| feature engineering, defaults | +0.0001 | 0.0032 | 2/5 |
| feature engineering, tuned | +0.0044 | 0.0018 | 5/5 |
| tuning, raw | +0.0049 | 0.0015 | 5/5 |
| tuning, engineered | +0.0091 | 0.0035 | 5/5 |

The engineered representation is **not** a free win: with defaults it is a wash.
It pays off only after tuning. Tuning is the more reliable lever, and the tuned
parameters were selected on seed 42, so the tuned rows are slightly optimistic
on the other seeds. This is why the README does not claim a large feature-
engineering effect.

## 5. Calibration and threshold analysis

### Calibration

`src/models/calibration.py` fits:

1. a raw LightGBM pipeline on the training set;
2. an isotonic `CalibratedClassifierCV` around the same pipeline, with 5
   internal folds.

Both are evaluated on the 6,000-row holdout:

| Metric | Raw | Isotonic |
|---|---:|---:|
| Brier | 0.1806 | **0.1342** |
| Log loss | 0.5465 | **0.4273** |
| ECE (10 bins) | 0.2047 | **0.0111** |
| Mean predicted | 0.4259 | 0.2221 |
| Observed rate | 0.2212 | 0.2212 |
| ROC AUC | 0.7828 | 0.7828 |

The large Brier/ECE gain mostly undoes the probability shift caused by
`class_weight="balanced"`. It is a real calibration correction, but it should
not be read as extra ranking skill: ROC AUC is unchanged.

### Cost-sensitive threshold

`src/models/threshold.py` uses nested cross-validation:

- outer 5 folds generate out-of-fold calibrated probabilities on the training
  set;
- inner 5 folds fit the calibrator inside each outer training fold;
- the threshold minimising `5 × FN + FP` is selected on those out-of-fold
  predictions;
- the calibrated model is refit on the full training set and evaluated once on
  the holdout.

| Threshold | Recall | Precision | F1 | Test cost |
|---:|---:|---:|---:|---:|
| 0.50 | 0.351 | 0.672 | 0.461 | 4,532 |
| **0.149** | **0.805** | 0.345 | **0.483** | **3,319 (−26.8%)** |

The selected threshold is low because calibrated probabilities are small and a
miss is five times as expensive as a false alarm. After calibration, 0.5 is much
stricter than it was for the raw class-weighted scores, which is why the recall
at 0.5 drops from 0.637 (raw) to 0.351 (calibrated). Change the cost ratio and
the threshold changes. This is a method demonstration, not a validated policy.

## 6. What limited the final score

1. **Information ceiling of the dataset.** Six months of behaviour is less than
   real scorecards have (usually 24–36 months). Some delinquency signals are
   simply not in the data.
2. **No macro covariates.** Unemployment, interest rates and regional stress
   shift default rates; adding them could lift AUC, but they are not in this
   dataset.
3. **Demographics carry little independent signal.** Their SHAP weights are
   dominated by behavioural history; a fair-lending pipeline would drop them and
   lose little ranking performance.
4. **A single-year portfolio.** Taiwan 2005 does not contain cyclical stagnation
   periods that stress-test scorecards.

## 7. Pitfalls checked in this repo

- **No scaler leakage:** `StandardScaler` is fit inside each CV fold via the
  `Pipeline`, never on the whole dataset.
- **No target leakage:** engineered features are row-local functions of the raw
  borrower fields. `tests/test_invariants.py` flips the target and asserts that
  every engineered feature is unchanged.
- **No resampling before the split:** there is no SMOTE call anywhere in the
  codebase.
- **SHAP input matches training input:** the explainer receives the standardized
  matrix; the plot uses raw feature values only for readability.
- **Deterministic seeds** for the split, CV and models; Docker pins the Python
  runtime; dependency ranges are bounded.

## 8. How to reproduce

```bash
python -m src.utils.pipeline        # comparison metrics + figures + artifacts
python -m src.models.ablation       # ablation CSV/JSON
python -m src.models.calibration    # calibration JSON + reliability figure
python -m src.models.threshold      # threshold JSON + cost curve
python -m scripts.regenerate_shap   # SHAP figures from the saved model
pytest -q
```

All commands are network-free after the first dataset download.
