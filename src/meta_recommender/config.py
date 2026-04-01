"""Configuration and constants."""

from pathlib import Path

MAX_DATASET_ROWS = 100_000
MIN_ROWS_FOR_EVAL = 50
DEFAULT_OPENML_SIZE = 30
RANDOM_STATE = 42
CV_FOLDS = 5
TIMEOUT_PER_MODEL_SECONDS = 90

META_FEATURE_ORDER = [
    "n_samples",
    "n_features",
    "missing_ratio",
    "n_numeric",
    "n_categorical",
    "mean_variance",
    "mean_skewness",
    "mean_abs_correlation",
    "pca_first_component_var",
    "class_imbalance_ratio",
]

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "meta_model.joblib"
