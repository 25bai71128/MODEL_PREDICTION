"""Evaluate the persisted meta-model on the saved meta-dataset."""

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from meta_recommender.config import META_DATASET_PATH, META_FEATURE_ORDER
from meta_recommender.logging_utils import setup_logging
from meta_recommender.predictor import MetaModelPredictor


def main() -> None:
    """Print held-out performance diagnostics for the trained meta-model."""
    setup_logging()
    predictor = MetaModelPredictor.load()
    meta_df = pd.read_csv(META_DATASET_PATH)
    meta_X = meta_df[[f"meta_{feature}" for feature in META_FEATURE_ORDER]].rename(
        columns={f"meta_{feature}": feature for feature in META_FEATURE_ORDER}
    )
    meta_y = meta_df["best_model"]
    metrics = MetaModelPredictor.evaluate_holdout(meta_X, meta_y)

    print(f"Accuracy: {float(metrics['accuracy']):.4f}")
    print(f"Top-3 Accuracy: {float(metrics['top_3_accuracy']):.4f}")


if __name__ == "__main__":
    main()
