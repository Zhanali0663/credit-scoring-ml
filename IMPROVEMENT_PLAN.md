# Improvement Plan

The original v1 plan (model card, calibration, ablation, README) is now
implemented. Current state and next steps:

## Done

- [x] Model card aligned with measured metrics (`MODEL_CARD.md`).
- [x] Isotonic calibration with Brier, log loss and ECE (`src/models/calibration.py`).
- [x] Out-of-fold, cost-sensitive threshold selection (`src/models/threshold.py`).
- [x] Clean 23-vs-61 feature / tuning ablation (`src/models/ablation.py`).
- [x] Real invariant tests, green `ruff`, GitHub Actions CI.
- [x] Executed notebooks 01–04 with committed outputs.
- [x] SHAP explains the scaled model input and plots raw feature values.
- [x] Local/Docker demo; no dead public links.

## Next

1. **Fairness audit** — disparate impact and equalized odds for `SEX`,
   `EDUCATION`, `MARRIAGE`, with confidence intervals.
2. **Drift monitoring** — PSI/KS on the 23 raw fields plus calibration drift;
   alert when the observed default rate leaves the training interval.
3. **Serving API** — a small FastAPI service around `app/predict.py` with a
   versioned model artifact and request logging.
4. **Experiment tracking** — MLflow or Weights & Biases for seeds, parameters
   and metrics (currently CSV/JSON artifacts).
5. **Cost validation** — replace the assumed 5:1 cost ratio with real loss-given-
   default and collection-cost estimates before using the threshold.
6. **Model registry** — content-hash artifacts, promotion rules and rollback.
