"""One-shot generator for portfolio notebooks from percent-format sources.

Correct percent format used here:
  - lines NOT preceded by `# %%` at the very top = intro markdown heading,
    until a `# %%` marker switches to code
  - `# %%`       -> new code cell
  - `# %% [markdown]` -> new markdown cell (body lines start with `# `)
"""
import json
import pathlib

notebooks = {}

notebooks["01_eda"] = r"""# 01 — Exploratory Data Analysis: Credit Card Default
#
# Dataset: UCI *Default of Credit Card Clients* (Taiwan, 2005) — 30 000 records,
# 23 features, binary target `default_payment_next_month` (~22% positives).

# %%
import sys, warnings
sys.path.append('..')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style='whitegrid')

from src.data.loader import load_raw_data, basic_clean
from src.features.engineering import add_behavioural_features

df = basic_clean(load_raw_data())
print(f'Shape: {df.shape}')
df.head()

# %% [markdown]
# ## Target balance
#
# The dataset is moderately imbalanced: ~22% of clients default. This guides
# the choice of metrics (F1, PR AUC) and class weighting in the models.

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))
counts = df['default_payment_next_month'].value_counts().sort_index()
bars = ax.bar(['Paid (0)', 'Defaulted (1)'], counts.values, color=['#2ecc71', '#e74c3c'])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v, f'{v:,} ({v/len(df):.1%})', ha='center', va='bottom')
ax.set_title('Target class balance')
plt.tight_layout()
plt.savefig('../reports/figures/01_class_balance.png', dpi=110)
plt.show()

# %% [markdown]
# ## Demographics
#
# Credit limit, age and the social attributes. Numeric attributes are aggregate
# risk proxies rather than causal features.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df['LIMIT_BAL'], bins=50, color='#3498db')
axes[0].set_title('Credit limit distribution')
axes[1].hist(df['AGE'], bins=40, color='#9b59b6')
axes[1].set_title('Age distribution')
sns.countplot(x='EDUCATION', data=df, ax=axes[2], color='#34495e')
axes[2].set_title('Education levels')
plt.tight_layout()
plt.savefig('../reports/figures/01_demographics.png', dpi=110)
plt.show()

# %% [markdown]
# ## Payment history — the core signal
#
# `PAY_0..PAY_6` record repayment status for the last six months
# (-1 = paid duly, 0 = revolving, 1..9 = months overdue). A client with
# `PAY_0 >= 2` is already behind on the newest bill — historically the
# strongest default predictor.

# %%
fig, ax = plt.subplots(figsize=(9, 4))
sns.countplot(x='PAY_0', hue='default_payment_next_month', data=df, ax=ax)
ax.set_title('Most recent repayment status vs default')
ax.legend(title='Defaulted')
plt.tight_layout()
plt.savefig('../reports/figures/01_pay0_vs_default.png', dpi=110)
plt.show()

delay_rates = df.groupby('PAY_0')['default_payment_next_month'].mean()
print(delay_rates.to_string())

# %% [markdown]
# ## Signed-log transforms tame heavy tails
#
# Raw bill/payment amounts span several orders of magnitude. A signed log1p
# transform lets linear models use the full range and improves tree split
# behaviour slightly.

# %%
df = add_behavioural_features(df)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df['BILL_AMT1'].clip(-5e5, 5e5), bins=50, color='#3498db')
axes[0].set_title('BILL_AMT1 (raw)')
axes[1].hist(df['LOG_BILL_AMT1'], bins=50, color='#9b59b6')
axes[1].set_title('LOG_BILL_AMT1 (signed log1p)')
axes[2].hist(df['DELAY_WEIGHTED'], bins=30, color='#e67e22')
axes[2].set_title('Recency-weighted delay index')
plt.tight_layout()
plt.savefig('../reports/figures/01_log_transforms.png', dpi=110)
plt.show()

# %% [markdown]
# ## Billing dynamics + correlation heatmap

# %%
fig, ax = plt.subplots(figsize=(7, 5))
corr_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'BILL_AMT1', 'PAY_AMT1',
             'default_payment_next_month']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation of numeric features with the target')
plt.tight_layout()
plt.savefig('../reports/figures/01_correlation.png', dpi=110)
plt.show()

eng = df
eng.groupby('default_payment_next_month')[['UTILIZATION', 'N_MONTHS_DELAY', 'DELAY_WEIGHTED']].mean()
"""

notebooks["02_feature_engineering"] = r"""# 02 — Feature Engineering Walkthrough
#
# 38 behavioural features derived from the raw 23-column payment history:
# volume aggregates, signed-log transforms of heavy-tailed monetary columns,
# affordability / utilisation ratios, recency-weighted delinquency depth and
# trend features. All of them are self-evident for business audiences — a
# requirement in regulated credit risk models.

# %%
import sys, warnings
sys.path.append('..')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style='whitegrid')

from src.data.loader import load_raw_data, basic_clean
from src.features.engineering import add_behavioural_features

df = basic_clean(load_raw_data())
df = add_behavioural_features(df)
df.head()

# %% [markdown]
# ## Engineered features summary
#
# The model uses all 38 derived features; the table below describes a
# representative subset (aggregates, ratios, delay signals and trends).

# %%
new_cols = ['BILL_TOTAL', 'BILL_STD', 'BILL_MAX', 'PAY_TOTAL', 'PAY_STD', 'PAY_MAX',
            'PAY_TO_BILL_RATIO', 'UTILIZATION',
            'UTILIZATION_MEAN', 'UTILIZATION_MAX', 'UTILIZATION_STD',
            'COVERAGE_MEAN', 'COVERAGE_MIN', 'LOG_LIMIT_BAL',
            'LOG_BILL_AMT1', 'LOG_PAY_AMT1',
            'N_MONTHS_DELAY', 'MAX_DELAY_SEVERITY', 'MEAN_DELAY', 'DELAY_STD',
            'DELAY_WEIGHTED', 'LONGEST_DELAY_STREAK', 'N_MONTHS_PAID_DULY',
            'BILL_TREND', 'PAY_TREND', 'DELAY_TREND']
df[new_cols].describe().T

# %% [markdown]
# ## Behavioural split by target class
#
# `N_MONTHS_DELAY`, `MAX_DELAY_SEVERITY`, `DELAY_WEIGHTED` and `UTILIZATION`
# separate defaulters from payers nicely — they aggregate raw payment history
# into monotonic business signals.

# %%
group = df.groupby('default_payment_next_month')[new_cols].mean().T
group.columns = ['Payers (0)', 'Defaulters (1)']
print(group.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
diff = (group['Defaulters (1)'] - group['Payers (0)']).sort_values()
colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in diff]
ax.barh(list(diff.index), diff.values, color=colors)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Mean difference (Defaulters − Payers) for engineered features')
plt.tight_layout()
plt.savefig('../reports/figures/02_feature_split.png', dpi=110)
plt.show()
"""

notebooks["03_modeling"] = r"""# 03 — Quick Modeling Demo
#
# This notebook is a **compact toy demo** — the reproducible, production
# pipeline (all 5 models with cross-validation) lives in `src/utils/pipeline.py`.
# Run `make train` before this notebook to build artifacts — or execute the
# local loading / feature-engineering cells as below.

# %%
import sys, warnings
sys.path.append('..')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from src.data.loader import load_raw_data, basic_clean
from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split
from src.models.pipe import build_pipeline
from src.models.trainer import cross_validate
from src.models.registry import MODEL_REGISTRY
from src.utils.metrics import evaluate_classifier

df = basic_clean(load_raw_data())
df = add_behavioural_features(df)
X, y = build_features_target(df)
X_train, X_test, y_train, y_test = stratified_split(X, y)
print(f'train {X_train.shape} | test {X_test.shape}')

# %% [markdown]
# ## Cross-validate all 5 models (compact leaderboard)
#
# Same `cross_validate` helper as the production pipeline (5-fold stratified
# CV on the 24k-row training set). It takes a few minutes on CPU. Reduce
# `n_splits` only for a quick local smoke run, and report the setting used.

# %%
num_features = X_train.select_dtypes(include='number').columns.tolist()
rows = []
for name in MODEL_REGISTRY:
    pipe = build_pipeline(name, num_features=num_features)
    summary = cross_validate(pipe, X_train, y_train, n_splits=5)
    rows.append({'model': name, **summary['mean']})
result_cv = pd.DataFrame(rows).sort_values(by='roc_auc', ascending=False).set_index('model')
result_cv.round(4)

# %% [markdown]
# ## Train a single LightGBM model and inspect test scores

# %%
pipe = build_pipeline('lightgbm', num_features=num_features)
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
preds = pipe.predict(X_test)
pd.Series(evaluate_classifier(y_test.values, preds, proba)).round(4)

# %% [markdown]
# ## Confusion matrix + ROC / PR curves

# %%
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('LightGBM confusion matrix')
RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
axes[1].set_title('ROC')
PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[2])
axes[2].set_title('Precision-Recall')
plt.tight_layout()
plt.savefig('../reports/figures/03_lightgbm_curves.png', dpi=110)
plt.show()
"""

notebooks["04_interpretation"] = r"""# 04 — Model Interpretation with SHAP
#
# SHAP values quantify how much each feature pushes a specific prediction
# higher or lower, relative to the expected prediction. They are the modern
# standard for explaining boosted-tree models to non-expert stakeholders.

# %%
import sys, warnings
sys.path.append('..')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import shap

from src.data.loader import load_raw_data, basic_clean
from src.features.engineering import add_behavioural_features, build_features_target
from src.features.splitting import stratified_split
from src.models.pipe import build_pipeline
from src.visualization.shap_utils import compute_shap_values
%matplotlib inline

df = basic_clean(load_raw_data())
df = add_behavioural_features(df)
X, y = build_features_target(df)
X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=42)

pipe = build_pipeline('lightgbm', num_features=X_train.columns.tolist())
pipe.fit(X_train, y_train)

# The estimator was trained on StandardScaler output, so TreeExplainer must
# receive the scaled matrix. `compute_shap_values` does that and returns the
# original (unscaled) rows for readable plots.
X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
sv, X_sample = compute_shap_values(pipe, X_sample)
print('Explainer ready, shap matrix shape:', sv.shape)

# %% [markdown]
# ## Beeswarm — global feature importance
#
# Each dot is a client. Red = high feature value, blue = low; x-axis = impact
# on the predicted default probability.

# %%
plt.figure(figsize=(9, 6))
shap.summary_plot(sv, X_sample, show=False)
plt.tight_layout()
plt.savefig('../reports/figures/04_shap_summary.png', dpi=110)
plt.show()

# %% [markdown]
# ## Bar summary — mean absolute SHAP values

# %%
plt.figure(figsize=(8, 6))
shap.summary_plot(sv, X_sample, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('../reports/figures/04_shap_bar.png', dpi=110)
plt.show()
"""


def parse_blocks(source: str):
    """Split a percent-format source into (type, lines) blocks."""
    blocks = []
    cur_type: str | None = None
    cur_lines: list[str] = []

    for line in source.split("\n"):
        if line.startswith("# %%"):
            if cur_lines and cur_type is not None:
                blocks.append((cur_type, cur_lines))
            cur_type = "markdown" if line.startswith("# %% [markdown]") else "code"
            cur_lines = []
        else:
            cur_lines.append(line)

    blocks.append((cur_type, cur_lines))

    out = []
    for typ, lines in blocks:
        while lines and not lines[0].strip():
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        if not lines or all(not line.strip() for line in lines):
            continue
        out.append((typ, lines))
    return out


def _split_intro(lines: list[str]):
    """Split the leading (unmarked) block into (markdown_heading, code)."""
    heading: list[str] = []
    idx = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            heading.append(line[2:])
            idx = i + 1
        elif not line.strip():
            continue
        else:
            idx = i
            break
    return heading, lines[idx:]


def to_ipynb(source: str):
    cells = []
    blocks = parse_blocks(source)

    if blocks and blocks[0][0] is None:
        heading, code = _split_intro(blocks[0][1])
        if heading:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(heading).strip()})
        if any(line.strip() for line in code):
            cells.append({
                "cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": "\n".join(code).strip(),
            })
        blocks = blocks[1:]

    for typ, lines in blocks:
        if typ == "markdown":
            text = [
                line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line)
                for line in lines
            ]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(text).strip()})
        else:
            cells.append({
                "cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": "\n".join(lines).strip(),
            })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


if __name__ == "__main__":
    base = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
    base.mkdir(parents=True, exist_ok=True)
    for name, source in notebooks.items():
        nb = to_ipynb(source)
        path = base / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        n_cells = len(nb["cells"])
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"created {path.name}: {n_cells} cells ({n_code} code)")
