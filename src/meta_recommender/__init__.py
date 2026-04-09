"""Meta-learning model recommender package."""

from .pipeline import recommend_for_csv, recommend_for_dataframe, run_training_pipeline
from .predictor import MetaModelPredictor

__all__ = [
    "MetaModelPredictor",
    "recommend_for_csv",
    "recommend_for_dataframe",
    "run_training_pipeline",
]
