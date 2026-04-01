"""Preprocessing and meta-feature extraction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import META_FEATURE_ORDER

logger = logging.getLogger(__name__)


def detect_task_type(y: pd.Series) -> str:
    """Infer classification/regression from target properties."""
    y_non_null = y.dropna()
    if y_non_null.empty:
        return "classification"

    if pd.api.types.is_object_dtype(y_non_null) or pd.api.types.is_categorical_dtype(y_non_null):
        return "classification"

    unique_count = y_non_null.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y_non_null), 1)

    # Typical heuristic: few unique values likely class labels.
    if unique_count <= 20 and unique_ratio < 0.2:
        return "classification"

    return "regression"


def clean_X(X: pd.DataFrame) -> pd.DataFrame:
    """Drop constant and fully missing columns; keep tabular DataFrame."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    X = X.copy()
    full_missing_cols = [c for c in X.columns if X[c].isna().all()]
    if full_missing_cols:
        X = X.drop(columns=full_missing_cols)

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    if X.empty:
        raise ValueError("No usable columns after cleaning features.")

    return X


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )

    return preprocessor, num_cols, cat_cols


def extract_meta_features(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute robust meta-features in a fixed order."""
    X = clean_X(X)

    n_samples, n_features = X.shape
    n_total_cells = n_samples * n_features
    missing_ratio = float(X.isna().sum().sum() / max(n_total_cells, 1))

    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_df = X[num_cols].copy() if num_cols else pd.DataFrame(index=X.index)

    if not num_df.empty:
        num_df = num_df.fillna(num_df.mean(numeric_only=True))
        variances = num_df.var(axis=0)
        variances = variances[variances > 0]
        mean_variance = float(variances.mean()) if not variances.empty else 0.0

        skewness = num_df.skew(axis=0).replace([np.inf, -np.inf], np.nan).dropna()
        mean_skewness = float(skewness.mean()) if not skewness.empty else 0.0

        corr = num_df.corr().replace([np.inf, -np.inf], np.nan)
        if corr.shape[0] > 1:
            upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            mean_abs_corr = float(upper_tri.abs().mean()) if not upper_tri.empty else 0.0
        else:
            mean_abs_corr = 0.0

        pca_var = 0.0
        try:
            pca = PCA(n_components=1, random_state=42)
            pca.fit(num_df)
            pca_var = float(pca.explained_variance_ratio_[0])
        except Exception:  # noqa: BLE001
            pca_var = 0.0
    else:
        mean_variance = 0.0
        mean_skewness = 0.0
        mean_abs_corr = 0.0
        pca_var = 0.0

    class_imbalance_ratio = 1.0
    if detect_task_type(y) == "classification":
        counts = y.value_counts(dropna=True)
        if not counts.empty and counts.max() > 0:
            class_imbalance_ratio = float(counts.min() / counts.max())

    features = {
        "n_samples": float(n_samples),
        "n_features": float(n_features),
        "missing_ratio": float(missing_ratio),
        "n_numeric": float(len(num_cols)),
        "n_categorical": float(len(cat_cols)),
        "mean_variance": float(mean_variance),
        "mean_skewness": float(mean_skewness),
        "mean_abs_correlation": float(mean_abs_corr),
        "pca_first_component_var": float(pca_var),
        "class_imbalance_ratio": float(class_imbalance_ratio),
    }

    # Enforce consistent ordering contract.
    return {k: features.get(k, 0.0) for k in META_FEATURE_ORDER}
