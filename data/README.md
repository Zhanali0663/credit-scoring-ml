# Data folder layout

```
data/
├── raw/          # original dataset (auto-downloaded, git-ignored)
├── processed/    # engineered features (auto-generated, git-ignored)
└── external/     # external data sources (git-ignored)
```

The dataset is **Default of Credit Card Clients** (UCI ML Repository, also
available on Kaggle as `uciml/default-of-credit-card-clients-dataset`).

`src/data/loader.py` downloads the CSV to `data/raw/` the first time it runs.
All three folders are git-ignored to keep the repository lightweight; the
download is reproducible via `make data`.
