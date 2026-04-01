"""Configuration and constants for the meta-learning system."""

from pathlib import Path

MAX_DATASET_ROWS = 100_000
MIN_ROWS_FOR_EVAL = 50
DEFAULT_OPENML_SIZE = 30
RANDOM_STATE = 42
CV_FOLDS = 5
TIMEOUT_PER_MODEL_SECONDS = 90
DEFAULT_TOP_K = 3
DEFAULT_N_JOBS = 2
Z_SCORE_THRESHOLD = 3.0

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "meta_model.joblib"
SCALER_PATH = MODEL_DIR / "meta_scaler.joblib"
META_DATASET_PATH = ARTIFACTS_DIR / "meta_dataset.csv"
EVALUATION_CACHE_PATH = ARTIFACTS_DIR / "evaluation_cache.joblib"
META_EVALUATION_PATH = ARTIFACTS_DIR / "meta_model_metrics.json"
LOG_PATH = Path("logs.txt")

META_FEATURE_ORDER = [
    "n_samples",
    "n_features",
    "missing_ratio",
    "n_numeric",
    "n_categorical",
    "mean_variance",
    "mean_skewness",
    "mean_kurtosis",
    "mean_entropy",
    "mean_abs_correlation",
    "pca_component_1_var",
    "pca_component_2_var",
    "feature_sparsity",
    "outlier_percentage",
    "class_imbalance_ratio",
]
