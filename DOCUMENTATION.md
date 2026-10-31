# Method Notes — Metrics, Tuning and Limitations

How to read `reports/metrics/comparison.csv`, what each model does well, and
where the methodology has honest gaps.

## 1. Metric glossary

| Metric | Purpose | Why it matters for credit scoring |
|---|---|---|
| ROC AUC | Threshold-free ranking quality | The headline metric for scorecards; how well a random defaulter ranks above a random payer |
| PR AUC | Precision vs recall area | More informative than ROC AUC at ~22 % positives; sensitive to the rare-class end |
| F1 | Balance of precision & recall | Single number for thresholded decisions |
| Precision | Spurious flags | Every false alarm is a lost sale (compliance burden, customer friction) |
| Recall | Caught defaulters | Every miss is a write-off (principal at risk) |
| KS statistic | max(TPR - FPR) | Classic scorecard separation measure; implied by the ROC curve |

Accuracy is reported for completeness but is the least informative metric
here: a constant "never default" classifier scores ~78 % accuracy.

## 2. How to read the numbers

Results are written per model with two views:

- `roc_auc`, `pr_auc`, `f1`, ... — held-out test set (6000 clients).
- `cv_mean_*`, `cv_std_*` — 5-fold stratified cross-validation on the
  training portion, which shows stability rather than a one-split lucky draw.

What to trust:

- `cv_std` lower than ~0.01 means the model is stable across folds.
- A CV mean noticeably above test ROC AUC with the same seed is normal
  (train folds are smaller, validation folds rotate); gaps above ~0.015 would
  suggest overfitting or a buggy split.

Reference points on this dataset (publicly reproducible ones):

| Setup | ROC AUC |
|---|---|
| Logistic regression, raw features (many tutorials) | ~0.72–0.75 |
| Random forest, defaults | ~0.77 |
| Boosting, defaults + 12 behavioural features | ~0.776–0.78 |
| Boosting, tuned + signed-log + recency-weighted delinquency | **~0.787–0.789 CV / 0.783 test** |
| Whatever claims 0.90+ | almost always leakage (SMOTE pre-split, train-fold metrics) |

## 3. Design decisions and trade-offs

- **Class weighting over SMOTE.** Balanced `class_weight` /
  `scale_pos_weight` are deterministic and keep validation bounds honest.
  SMOTE applied before split leaks synthetic neighbours across the boundary;
  applied inside each fold it is valid, but the class-weight route is simpler
  and achieves comparable recall in this dataset.
- **Signed log1p for money columns.** Raw amounts span 3-4 orders of
  magnitude. Without the transform logistic regression weights outliers
  heavily and wastes capacity; with it, LR goes from ~0.745 to ~0.753 test
  AUC and boosting gains a little as well (stable split points).
- **Recency-weighted delay index.** Payment status of the newest month is the
  strongest single predictor; the `DELAY_WEIGHTED` feature encodes this
  grading directly rather than hoping the model learns it.
- **Optuna with a hard budget.** Tuning time is capped (--budget / --trials),
  so the tuning stage cannot run away. Every trial is logged with running
  best AUC and remaining time; no silent long run.
- **Calibration is out of scope.** Probabilities are good for ranking, not
  yet for quoting "your default probability is 7 %" to a customer. Production
  would add isotonic/sigmoid calibration and report Brier score.

## 4. What limited the final score

1. **Information ceiling of the dataset.** Six months of behaviour is less
   than real scorecards have (usually 24-36 months of history). Some
   delinquency signals simply are not in the data.
2. **No macro covariates.** Unemployment, interest rates, regional stress
   shift default rates; adding them typically lifts AUC by another notch but
   is impossible on this dataset.
3. **Demographics carry little independent signal.** Their SHAP weights are
   dominated by behavioural history; a fair-lending pipeline would drop them
   and lose almost nothing in AUC.
4. **A single-year portfolio.** Taiwan 2005 does not contain cyclical stagnation
   periods that stress-test scorecards.

## 5. Hovering pitfalls (checked in this repo)

- No `std` leakage: `StandardScaler` is fit inside each CV fold via the
  `Pipeline` object, never on the whole dataset.
- No target leakage: aggregates use the *past* five months only relative to
  the last observed period; the target is next-month default.
- Deterministic seeds everywhere for reproducibility; Docker pins the runtime.
