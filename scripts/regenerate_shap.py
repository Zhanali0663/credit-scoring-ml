"""Regenerate the committed SHAP figures for the headline LightGBM model.

The notebook ``04_interpretation.ipynb`` produces the same figures; this script
is a fast, headless alternative for CI or for refreshing artifacts after a
model retrain.

Usage::

    python -m scripts.regenerate_shap
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import basic_clean, load_raw_data  # noqa: E402
from src.features.engineering import add_behavioural_features, build_features_target  # noqa: E402
from src.features.splitting import stratified_split  # noqa: E402
from src.models.trainer import load_pipeline  # noqa: E402
from src.utils.config import get_project_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.visualization.shap_utils import shap_bar_plot, shap_summary_plot  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    root = get_project_root()
    config = yaml.safe_load((root / "configs" / "paths.yaml").read_text("utf-8"))
    df = add_behavioural_features(basic_clean(load_raw_data()))
    X, y = build_features_target(df)
    _, X_test, _, _ = stratified_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)

    pipe = load_pipeline("lightgbm")
    figures = root / "reports" / "figures"
    shap_summary_plot(pipe, sample, save_to=figures / "lightgbm_shap_summary.png")
    shap_bar_plot(pipe, sample, save_to=figures / "lightgbm_shap_bar.png")
    logger.info("Regenerated SHAP figures in %s", figures)


if __name__ == "__main__":
    main()
