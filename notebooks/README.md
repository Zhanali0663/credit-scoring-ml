# Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory analysis: target balance, demographic distribution, monthly bill/payment trends, correlation matrix. |
| `02_feature_engineering.ipynb` | Walkthrough of the 12 engineered behavioural features and how they differ across target classes. |
| `03_modeling.ipynb` | Short demo of training one model locally; the full 5-model benchmark lives in `src/utils/pipeline.py`. |
| `04_interpretation.ipynb` | SHAP beeswarm for the overall decisions and top-1 local explanation. |

> **Note** — Notebooks are self-contained demos; the reproducible, production
> pipeline lives in `src/` so that everything runs easily in Docker. Run
> `make train` before opening notebooks 3–4 to have artifacts/models available.
