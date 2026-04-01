"""Model evaluation utilities for candidate base learners."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
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

from .config import CV_FOLDS, EVALUATION_CACHE_PATH, MIN_ROWS_FOR_EVAL, RANDOM_STATE, TIMEOUT_PER_MODEL_SECONDS
from .features import detect_task_type
from .logging_utils import log_exception

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluation:
    """Evaluation details for one candidate learner."""

    score: float
    training_time: float
    inference_time: float
    skipped: bool = False
    reason: str = ""


@dataclass
class EvaluationResult:
    """Aggregate evaluation result for one dataset."""

    best_model: str | None
    task_type: str
    scores: dict[str, float]
    timings: dict[str, dict[str, float | bool | str]]


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
        "LinearRegression": Pipeline([("prep", "passthrough"), ("model", LinearRegression())]),
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
    scaled = {"SVM", "LogisticRegression", "LinearRegression", "KNN"}
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


def _fit_predict_timing(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    start_train = time.perf_counter()
    pipe.fit(X, y)
    training_time = time.perf_counter() - start_train

    sample = X.head(min(20, len(X)))
    start_infer = time.perf_counter()
    pipe.predict(sample)
    inference_time = time.perf_counter() - start_infer
    return training_time, inference_time


def _run_with_timeout(func, timeout_sec: int):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        return future.result(timeout=timeout_sec)


def _load_cache(path: Path = EVALUATION_CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return joblib.load(path)
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(cache: dict[str, dict], path: Path = EVALUATION_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, path)


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_key: str | None = None,
    cache_path: Path = EVALUATION_CACHE_PATH,
) -> EvaluationResult:
    """Evaluate candidate models and return scores plus timing metadata."""
    task_type = detect_task_type(y)
    if len(X) < MIN_ROWS_FOR_EVAL:
        logger.warning("Dataset too small for stable CV (%s rows). Skipping evaluation.", len(X))
        return EvaluationResult(best_model=None, task_type=task_type, scores={}, timings={})

    models = _attach_preprocessors(_candidate_models(task_type), X)
    cache = _load_cache(cache_path)
    if dataset_key and dataset_key in cache:
        cached = cache[dataset_key]
        return EvaluationResult(
            best_model=cached.get("best_model"),
            task_type=cached.get("task_type", task_type),
            scores=cached.get("scores", {}),
            timings=cached.get("timings", {}),
        )

    scores: dict[str, float] = {}
    timings: dict[str, dict[str, float | bool | str]] = {}

    for name, pipe in models.items():
        try:
            score = _run_with_timeout(lambda: _cross_val(pipe, X, y, task_type), TIMEOUT_PER_MODEL_SECONDS)
            training_time, inference_time = _run_with_timeout(
                lambda: _fit_predict_timing(pipe, X, y),
                TIMEOUT_PER_MODEL_SECONDS,
            )
            scores[name] = score
            timings[name] = asdict(
                CandidateEvaluation(score=score, training_time=training_time, inference_time=inference_time)
            )
            logger.info("Model %s score=%.5f (%s)", name, score, task_type)
        except TimeoutError:
            logger.warning("Model %s timed out (> %ss).", name, TIMEOUT_PER_MODEL_SECONDS)
            timings[name] = asdict(
                CandidateEvaluation(
                    score=float("-inf"),
                    training_time=float(TIMEOUT_PER_MODEL_SECONDS),
                    inference_time=0.0,
                    skipped=True,
                    reason="timeout",
                )
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(logger, "model training", name, exc)
            timings[name] = asdict(
                CandidateEvaluation(
                    score=float("-inf"),
                    training_time=0.0,
                    inference_time=0.0,
                    skipped=True,
                    reason=str(exc),
                )
            )

    if not scores:
        return EvaluationResult(best_model=None, task_type=task_type, scores={}, timings=timings)

    best_model = max(scores.items(), key=lambda item: item[1])[0]
    result = EvaluationResult(best_model=best_model, task_type=task_type, scores=scores, timings=timings)
    if dataset_key:
        cache[dataset_key] = {
            "best_model": result.best_model,
            "task_type": result.task_type,
            "scores": result.scores,
            "timings": result.timings,
        }
        _save_cache(cache, cache_path)
    return result
