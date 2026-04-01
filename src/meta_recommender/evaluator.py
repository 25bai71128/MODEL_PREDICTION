"""Model evaluation utilities for candidate base learners."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from .config import CV_FOLDS, MIN_ROWS_FOR_EVAL, RANDOM_STATE, TIMEOUT_PER_MODEL_SECONDS
from .features import detect_task_type

logger = logging.getLogger(__name__)
EVAL_CACHE: dict[tuple, tuple[str | None, dict[str, float], str, dict[str, dict[str, float]]]] = {}


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _common_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="mean"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def _candidate_models(task_type: str) -> dict[str, Pipeline]:
    is_cls = task_type == "classification"

    if is_cls:
        return {
            "RandomForest": Pipeline(
                [("prep", "passthrough"), ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=150))]
            ),
            "SVM": Pipeline(
                [("prep", "passthrough"), ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]
            ),
            "LogisticRegression": Pipeline(
                [("prep", "passthrough"), ("model", LogisticRegression(max_iter=800, class_weight="balanced"))]
            ),
            "KNN": Pipeline([("prep", "passthrough"), ("model", KNeighborsClassifier(n_neighbors=5))]),
            "XGBoost": Pipeline(
                [
                    ("prep", "passthrough"),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=200,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.8,
                            random_state=RANDOM_STATE,
                            eval_metric="logloss",
                        ),
                    ),
                ]
            ),
        }

    return {
        "RandomForest": Pipeline(
            [("prep", "passthrough"), ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=150))]
        ),
        "SVM": Pipeline([("prep", "passthrough"), ("model", SVR(kernel="rbf"))]),
        "LogisticRegression": Pipeline([("prep", "passthrough"), ("model", LinearRegression())]),
        "KNN": Pipeline([("prep", "passthrough"), ("model", KNeighborsRegressor(n_neighbors=5))]),
        "XGBoost": Pipeline(
            [
                ("prep", "passthrough"),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        objective="reg:squarederror",
                    ),
                ),
            ]
        ),
    }


def _attach_preprocessors(models: dict[str, Pipeline], X: pd.DataFrame) -> dict[str, Pipeline]:
    scaled = {"SVM", "LogisticRegression", "KNN"}
    with_prep = {}
    for name, pipe in models.items():
        prep = _common_preprocessor(X, scale_numeric=name in scaled)
        pipe.set_params(prep=prep)
        with_prep[name] = pipe
    return with_prep


def _cross_val(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipe, X, y, scoring="f1_weighted", cv=cv, n_jobs=1)
        return float(np.mean(scores))

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scorer = make_scorer(_rmse, greater_is_better=False)
    scores = cross_val_score(pipe, X, y, scoring=rmse_scorer, cv=cv, n_jobs=1)
    return float(np.mean(scores))


def _run_with_timeout(func, timeout_sec: int):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        return future.result(timeout=timeout_sec)


def _dataset_signature(X: pd.DataFrame, y: pd.Series) -> tuple:
    return (len(X), X.shape[1], tuple(map(str, X.dtypes.tolist())), str(y.dtype), int(y.nunique(dropna=True)))


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> tuple[str | None, dict[str, float], str, dict[str, dict[str, float]]]:
    """Evaluate required model set and return best model + per-model scores."""
    task_type = detect_task_type(y)
    cache_key = _dataset_signature(X, y)
    if cache_key in EVAL_CACHE:
        logger.info("Using cached evaluation results.")
        return EVAL_CACHE[cache_key]

    if len(X) < MIN_ROWS_FOR_EVAL:
        logger.warning("Dataset too small for stable CV (%s rows). Skipping evaluation.", len(X))
        result = (None, {}, task_type, {})
        EVAL_CACHE[cache_key] = result
        return result

    models = _attach_preprocessors(_candidate_models(task_type), X)

    scores: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, pipe in models.items():
        start = time.time()
        try:
            score = _run_with_timeout(lambda: _cross_val(pipe, X, y, task_type), TIMEOUT_PER_MODEL_SECONDS)
            scores[name] = score
            fit_time = time.time() - start
            pred_start = time.time()
            pipe.fit(X, y)
            _ = pipe.predict(X.iloc[: min(10, len(X))])
            infer_time = time.time() - pred_start
            metrics[name] = {"training_time_sec": float(fit_time), "inference_time_sec": float(infer_time)}
            logger.info("Model %s score=%.5f (%s)", name, score, task_type)
        except TimeoutError:
            logger.warning("Model %s timed out (> %ss).", name, TIMEOUT_PER_MODEL_SECONDS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model %s failed: %s", name, exc)
        finally:
            logger.debug("%s runtime: %.2fs", name, time.time() - start)

    if not scores:
        result = (None, {}, task_type, metrics)
        EVAL_CACHE[cache_key] = result
        return result

    reverse = task_type == "classification"  # maximize F1, maximize (less negative) RMSE score.
    best_model = sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[0][0]
    result = (best_model, scores, task_type, metrics)
    EVAL_CACHE[cache_key] = result
    return result
