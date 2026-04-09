"""Configuration and constants for the meta-learning system."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MAX_DATASET_ROWS = 100_000
MIN_ROWS_FOR_EVAL = 50
DEFAULT_OPENML_SIZE = 100
RANDOM_STATE = 42
CV_FOLDS = 5
TIMEOUT_PER_MODEL_SECONDS = 90
DATASET_LOAD_TIMEOUT_SECONDS = 120
DATASET_PROCESS_TIMEOUT_SECONDS = 420
DATASET_RETRY_COUNT = 2
MAX_DATASET_SCAN_MULTIPLIER = 25
DEBUG_DATASET_LIMIT = 3
DEBUG_MODEL_LIMIT = 1
DEFAULT_TOP_K = 3
DEFAULT_N_JOBS = 2
Z_SCORE_THRESHOLD = 3.0
AUTOML_TIME_BUDGET_SECONDS = 30
INFERENCE_WEIGHT = 5.0
RANK_RELEVANCE_SCALE = 20

ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"

MODEL_PATH = MODEL_DIR / "meta_model.joblib"
SCALER_PATH = MODEL_DIR / "meta_scaler.joblib"
META_DATASET_PATH = ARTIFACTS_DIR / "meta_dataset.csv"
EVALUATION_CACHE_PATH = ARTIFACTS_DIR / "evaluation_cache.joblib"
META_EVALUATION_PATH = ARTIFACTS_DIR / "meta_model_metrics.json"
TRAINING_REPORT_PATH = REPORT_DIR / "training_report.json"
AUTOML_BENCHMARK_PATH = REPORT_DIR / "automl_benchmark.json"
LOG_PATH = ROOT / "logs.txt"

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
    "feature_importance_var",
    "class_imbalance_ratio",
]
