# Notebooks

All notebooks have been executed; outputs are committed so they can be read
without re-running the code.

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Target balance, demographics, payment status vs default, signed-log transforms, correlation. |
| `02_feature_engineering.ipynb` | The 38 engineered behavioural features and how they differ across target classes. |
| `03_modeling.ipynb` | 5-fold cross-validation of all five models, a single LightGBM fit, confusion matrix and ROC/PR curves. |
| `04_interpretation.ipynb` | SHAP beeswarm and mean-|SHAP| bar chart for LightGBM. Explains the scaled model input and plots raw feature values. |

> **Note** — Notebooks are self-contained demos; the reproducible pipeline
> lives in `src/`. Notebook 03 trains its own models and takes a few minutes on
> CPU. Notebook 04 also trains a LightGBM model, so no prior `make train` is
> required. Figures are written to `reports/figures/`.
