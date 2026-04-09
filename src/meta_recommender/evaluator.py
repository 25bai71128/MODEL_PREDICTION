"""Model evaluation utilities for candidate base learners and AutoML baselines."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import f1_score, make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

try:
    from flaml import AutoML

    HAS_FLAML = True
except ImportError:  # pragma: no cover
    HAS_FLAML = False

try:
    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.regression import AutoSklearnRegressor

    HAS_AUTOSKLEARN = True
except ImportError:  # pragma: no cover
    HAS_AUTOSKLEARN = False

try:
    from tpot import TPOTClassifier, TPOTRegressor

    HAS_TPOT = True
except ImportError:  # pragma: no cover
    HAS_TPOT = False

from .config import (
    AUTOML_BENCHMARK_PATH,
    AUTOML_TIME_BUDGET_SECONDS,
    CV_FOLDS,
    DEBUG_MODEL_LIMIT,
    EVALUATION_CACHE_PATH,
    MIN_ROWS_FOR_EVAL,
    RANDOM_STATE,
    TIMEOUT_PER_MODEL_SECONDS,
)
from .features import detect_task_type
from .logging_utils import log_exception
from .runtime_utils import HardTimeoutError, run_with_hard_timeout

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


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _common_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="mean"))]
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


def _candidate_models(task_type: str, *, debug: bool = False) -> dict[str, BaseEstimator]:
    is_classification = task_type == "classification"
    if is_classification:
        models: dict[str, BaseEstimator] = {
            "LogisticRegression": LogisticRegression(max_iter=1200, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(
                n_estimators=220,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=220,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.06,
                random_state=RANDOM_STATE,
            ),
            "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "XGBoost": XGBClassifier(
                n_estimators=220,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=1,
            ),
        }
    else:
        models = {
            "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=RANDOM_STATE),
            "RandomForest": RandomForestRegressor(
                n_estimators=220,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=220,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.06,
                random_state=RANDOM_STATE,
            ),
            "SVM": SVR(kernel="rbf"),
            "KNN": KNeighborsRegressor(n_neighbors=7),
            "XGBoost": XGBRegressor(
                n_estimators=260,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                n_jobs=1,
            ),
        }

    if debug:
        first_name = next(iter(models))
        logger.debug("Debug mode enabled: restricting evaluator to model %s.", first_name)
        return {first_name: models[first_name]}
    return models


def _attach_preprocessors(models: dict[str, BaseEstimator], X: pd.DataFrame) -> dict[str, Pipeline]:
    scaled = {"SVM", "LogisticRegression", "ElasticNet", "KNN"}
    pipelines: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        prep = _common_preprocessor(X, scale_numeric=name in scaled)
        pipelines[name] = Pipeline([("prep", prep), ("model", clone(estimator))])
    return pipelines


def _cross_val(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
    if task_type == "classification":
        min_class_count = int(y.value_counts(dropna=True).min()) if not y.empty else 0
        if min_class_count < 2 or len(X) < 4:
            pipe.fit(X, y)
            preds = pipe.predict(X)
            return float(f1_score(y, preds, average="weighted"))
        n_splits = min(CV_FOLDS, max(2, min_class_count))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipe, X, y, scoring="f1_weighted", cv=cv, n_jobs=1)
        return float(np.mean(scores))

    if len(X) < 4:
        pipe.fit(X, y)
        preds = pipe.predict(X)
        return float(r2_score(y, preds))
    n_splits = min(CV_FOLDS, max(2, len(X) // 5))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmse_scorer = make_scorer(_rmse, greater_is_better=False)
    scores = cross_val_score(pipe, X, y, scoring=rmse_scorer, cv=cv, n_jobs=1)
    return float(np.mean(scores))


def _fit_predict_timing(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    start_train = time.perf_counter()
    pipe.fit(X, y)
    training_time = time.perf_counter() - start_train

    sample = X.head(min(32, len(X)))
    start_infer = time.perf_counter()
    pipe.predict(sample)
    inference_time = time.perf_counter() - start_infer
    return training_time, inference_time


def _evaluate_single_model_worker(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    debug: bool = False,
) -> dict[str, float | bool | str]:
    models = _candidate_models(task_type, debug=debug)
    if model_name not in models:
        raise KeyError(f"Unknown model: {model_name}")

    pipe = _attach_preprocessors({model_name: models[model_name]}, X)[model_name]
    score = _cross_val(pipe, X, y, task_type)
    training_time, inference_time = _fit_predict_timing(pipe, X, y)
    return asdict(
        CandidateEvaluation(
            score=float(score),
            training_time=float(training_time),
            inference_time=float(inference_time),
        )
    )


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
    *,
    debug: bool = False,
    dataset_name: str | None = None,
) -> EvaluationResult:
    """Evaluate candidate models and return scores plus timing metadata."""
    task_type = detect_task_type(y)
    dataset_label = dataset_name or dataset_key or "dataset"
    logger.info("Evaluator start for %s (%s).", dataset_label, task_type)

    if len(X) < MIN_ROWS_FOR_EVAL and not debug:
        logger.warning("Dataset %s too small for stable CV (%s rows). Skipping evaluation.", dataset_label, len(X))
        return EvaluationResult(best_model=None, task_type=task_type, scores={}, timings={})
    if task_type == "classification" and y.nunique(dropna=True) < 2:
        logger.warning("Dataset %s has fewer than 2 target classes. Skipping evaluation.", dataset_label)
        return EvaluationResult(best_model=None, task_type=task_type, scores={}, timings={})
    if task_type == "regression" and y.nunique(dropna=True) < 2:
        logger.warning("Dataset %s has a constant regression target. Skipping evaluation.", dataset_label)
        return EvaluationResult(best_model=None, task_type=task_type, scores={}, timings={})

    cache = _load_cache(cache_path)
    cache_key = f"{dataset_key}::debug={debug}"
    if dataset_key and cache_key in cache:
        cached = cache[cache_key]
        logger.info("Evaluator cache hit for %s.", dataset_label)
        return EvaluationResult(
            best_model=cached.get("best_model"),
            task_type=cached.get("task_type", task_type),
            scores=cached.get("scores", {}),
            timings=cached.get("timings", {}),
        )

    model_names = list(_candidate_models(task_type, debug=debug).keys())
    if debug:
        model_names = model_names[:DEBUG_MODEL_LIMIT]

    scores: dict[str, float] = {}
    timings: dict[str, dict[str, float | bool | str]] = {}

    for model_name in model_names:
        logger.info("Dataset %s | model start: %s", dataset_label, model_name)
        try:
            timed = run_with_hard_timeout(
                _evaluate_single_model_worker,
                kwargs={
                    "model_name": model_name,
                    "X": X,
                    "y": y,
                    "task_type": task_type,
                    "debug": debug,
                },
                timeout_seconds=TIMEOUT_PER_MODEL_SECONDS,
                stage_name=f"model evaluation {model_name}",
            )
            payload = timed.value
            scores[model_name] = float(payload["score"])
            timings[model_name] = payload
            logger.info(
                "Dataset %s | model done: %s score=%.5f train=%.2fs infer=%.4fs total=%.2fs",
                dataset_label,
                model_name,
                float(payload["score"]),
                float(payload["training_time"]),
                float(payload["inference_time"]),
                timed.elapsed_seconds,
            )
        except HardTimeoutError:
            logger.warning("Dataset %s | model timeout: %s", dataset_label, model_name)
            timings[model_name] = asdict(
                CandidateEvaluation(
                    score=float("-inf"),
                    training_time=float(TIMEOUT_PER_MODEL_SECONDS),
                    inference_time=0.0,
                    skipped=True,
                    reason="timeout",
                )
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(logger, "model training", model_name, exc)
            timings[model_name] = asdict(
                CandidateEvaluation(
                    score=float("-inf"),
                    training_time=0.0,
                    inference_time=0.0,
                    skipped=True,
                    reason=str(exc),
                )
            )

    if not scores:
        logger.warning("Evaluator produced no valid scores for %s.", dataset_label)
        return EvaluationResult(best_model=None, task_type=task_type, scores={}, timings=timings)

    best_model = max(scores.items(), key=lambda item: item[1])[0]
    result = EvaluationResult(best_model=best_model, task_type=task_type, scores=scores, timings=timings)
    if dataset_key:
        cache[cache_key] = {
            "best_model": result.best_model,
            "task_type": result.task_type,
            "scores": result.scores,
            "timings": result.timings,
        }
        _save_cache(cache, cache_path)

    logger.info("Evaluator finished for %s. Best model: %s", dataset_label, best_model)
    return result


def _fit_automl_tool(name: str, X: pd.DataFrame, y: pd.Series, task_type: str) -> dict[str, Any]:
    start = time.perf_counter()
    if name == "FLAML":
        if not HAS_FLAML:
            raise ImportError("FLAML is not installed.")
        automl = AutoML()
        automl.fit(
            X_train=X,
            y_train=y,
            task="classification" if task_type == "classification" else "regression",
            time_budget=AUTOML_TIME_BUDGET_SECONDS,
            verbose=0,
        )
        predictor = automl
    elif name == "AutoSklearn":
        if not HAS_AUTOSKLEARN:
            raise ImportError("Auto-sklearn is not installed.")
        if task_type == "classification":
            predictor = AutoSklearnClassifier(
                time_left_for_this_task=AUTOML_TIME_BUDGET_SECONDS,
                per_run_time_limit=max(10, AUTOML_TIME_BUDGET_SECONDS // 3),
                seed=RANDOM_STATE,
            )
        else:
            predictor = AutoSklearnRegressor(
                time_left_for_this_task=AUTOML_TIME_BUDGET_SECONDS,
                per_run_time_limit=max(10, AUTOML_TIME_BUDGET_SECONDS // 3),
                seed=RANDOM_STATE,
            )
        predictor.fit(X.copy(), y.copy())
    elif name == "TPOT":
        if not HAS_TPOT:
            raise ImportError("TPOT is not installed.")
        if task_type == "classification":
            predictor = TPOTClassifier(
                generations=3,
                population_size=12,
                cv=3,
                max_time_mins=max(1, AUTOML_TIME_BUDGET_SECONDS // 60),
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=1,
            )
        else:
            predictor = TPOTRegressor(
                generations=3,
                population_size=12,
                cv=3,
                max_time_mins=max(1, AUTOML_TIME_BUDGET_SECONDS // 60),
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=1,
            )
        predictor.fit(X.copy(), y.copy())
    else:  # pragma: no cover
        raise ValueError(f"Unsupported AutoML tool: {name}")

    elapsed = time.perf_counter() - start
    predictions = predictor.predict(X)
    if task_type == "classification":
        score = float(f1_score(y, predictions, average="weighted"))
    else:
        score = float(r2_score(y, predictions))
    return {"tool": name, "score": score, "elapsed_seconds": elapsed}


def benchmark_against_automl(
    X: pd.DataFrame,
    y: pd.Series,
    recommended_model: str | None,
    recommender_score: float | None = None,
    output_path: Path = AUTOML_BENCHMARK_PATH,
) -> dict[str, Any]:
    """Benchmark the current recommendation against optional AutoML systems."""
    task_type = detect_task_type(y)
    tools = ["FLAML", "AutoSklearn", "TPOT"]
    results: list[dict[str, Any]] = []

    for tool_name in tools:
        logger.info("AutoML benchmark start: %s", tool_name)
        try:
            results.append(_fit_automl_tool(tool_name, X, y, task_type))
        except Exception as exc:  # noqa: BLE001
            results.append({"tool": tool_name, "error": str(exc)})

    valid_scores = [entry["score"] for entry in results if "score" in entry]
    best_automl_score = max(valid_scores) if valid_scores else None
    gap = None
    if recommender_score is not None and best_automl_score is not None:
        gap = float(best_automl_score - recommender_score)

    summary = {
        "task_type": task_type,
        "recommended_model": recommended_model,
        "recommender_score": recommender_score,
        "best_automl_score": best_automl_score,
        "score_gap_vs_best_automl": gap,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
