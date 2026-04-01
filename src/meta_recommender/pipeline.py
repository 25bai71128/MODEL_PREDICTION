"""End-to-end pipeline to construct, train, and serve the meta-learning system."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import pandas as pd
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from .config import DEFAULT_N_JOBS, DEFAULT_OPENML_SIZE, DEFAULT_TOP_K, META_DATASET_PATH, META_FEATURE_ORDER
from .features import clean_X, detect_task_type, extract_meta_features
from .logging_utils import log_exception, setup_logging
from .predictor import MetaModelPredictor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .data_loader import DatasetBundle


@dataclass
class MetaRecord:
    """Single meta-learning training record built from one source dataset."""

    dataset_id: int
    dataset_name: str
    best_model: str
    task_type: str
    n_rows: int
    n_cols: int
    model_scores: dict[str, float]
    model_timings: dict[str, dict[str, float | bool | str]]
    meta_features: dict[str, float]


def _summarize_dataset(df: pd.DataFrame) -> dict[str, int | str]:
    """Return a lightweight user-facing summary for a tabular dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
    }


def process_dataset_bundle(bundle: "DatasetBundle") -> MetaRecord | None:
    """Process one OpenML dataset into a meta-learning training record."""
    try:
        from .evaluator import evaluate_models

        X = clean_X(bundle.X)
        y = bundle.y
        meta = extract_meta_features(X, y)
        eval_result = evaluate_models(X, y, dataset_key=str(bundle.dataset_id))
        if not eval_result.best_model:
            logger.info("Skipping dataset %s due to missing valid model scores.", bundle.name)
            return None

        logger.info("Best model for %s: %s", bundle.name, eval_result.best_model)
        return MetaRecord(
            dataset_id=bundle.dataset_id,
            dataset_name=bundle.name,
            best_model=eval_result.best_model,
            task_type=eval_result.task_type,
            n_rows=len(X),
            n_cols=X.shape[1],
            model_scores=eval_result.scores,
            model_timings=eval_result.timings,
            meta_features=meta,
        )
    except Exception as exc:  # noqa: BLE001
        log_exception(logger, "dataset processing", bundle.name, exc)
        return None


def _meta_records_to_frame(records: list[MetaRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **asdict(record),
                **{f"meta_{key}": value for key, value in record.meta_features.items()},
            }
            for record in records
        ]
    )


def run_training_pipeline(
    openml_limit: int = DEFAULT_OPENML_SIZE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> tuple[MetaModelPredictor | None, pd.DataFrame]:
    """Build meta-dataset from OpenML datasets and train the persisted meta-model."""
    setup_logging()
    from .data_loader import load_openml_datasets

    bundles = list(load_openml_datasets(limit=openml_limit))
    records: list[MetaRecord] = []

    if n_jobs > 1 and len(bundles) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_dataset_bundle, bundle) for bundle in bundles]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
                record = future.result()
                if record is not None:
                    records.append(record)
    else:
        for bundle in tqdm(bundles, desc="Processing datasets"):
            record = process_dataset_bundle(bundle)
            if record is not None:
                records.append(record)

    if not records:
        logger.error("No valid datasets produced a meta-record.")
        return None, pd.DataFrame()

    records.sort(key=lambda record: record.dataset_id)
    df = _meta_records_to_frame(records)
    META_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(META_DATASET_PATH, index=False)

    meta_X = pd.DataFrame([record.meta_features for record in records])[META_FEATURE_ORDER]
    meta_y = pd.Series([record.best_model for record in records], name="best_model")

    logger.info("Meta label distribution: %s", meta_y.value_counts().to_dict())
    predictor = MetaModelPredictor.train(meta_X, meta_y)
    predictor.save()
    logger.info("Training complete using %s datasets.", len(records))
    return predictor, df


def recommend_for_dataframe(df: pd.DataFrame, predictor: MetaModelPredictor, target_column: str | None = None) -> dict:
    """Generate recommendations for an in-memory DataFrame."""
    if df.empty:
        raise ValueError("Input dataset is empty.")

    working_df = df.copy()
    if target_column:
        if target_column not in working_df.columns:
            raise ValueError(f"Target column '{target_column}' was not found.")
        y = working_df.pop(target_column)
    else:
        y = pd.Series([0] * len(working_df), name="target")

    meta_features = extract_meta_features(working_df, y)
    best = predictor.predict_best_model(meta_features)
    top_k = predictor.predict_top_k_models(meta_features, k=DEFAULT_TOP_K)
    problem_type = detect_task_type(y)

    return {
        "best_model": best,
        "top_3": [{"model": model, "probability": probability} for model, probability in top_k],
        "meta_features": meta_features,
        "dataset_summary": _summarize_dataset(df),
        "problem_type": problem_type,
        "target_column": target_column,
    }


def recommend_for_csv(csv_path: str, predictor: MetaModelPredictor, target_column: str | None = None) -> dict:
    """Run recommendation on a user-provided CSV file."""
    df = pd.read_csv(csv_path)
    return recommend_for_dataframe(df, predictor, target_column=target_column)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training and prediction."""
    parser = argparse.ArgumentParser(description="Meta-learning model recommender")
    parser.add_argument("--train", action="store_true", help="Train the meta-model on OpenML datasets.")
    parser.add_argument("--openml-limit", type=int, default=DEFAULT_OPENML_SIZE, help="Number of OpenML datasets to process.")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS, help="Number of worker processes for dataset processing.")
    parser.add_argument("--predict-csv", type=str, default=None, help="Path to CSV file for prediction.")
    parser.add_argument("--target", type=str, default=None, help="Optional target column name for prediction.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for both training and batch recommendation."""
    setup_logging()
    args = parse_args()

    if args.train:
        predictor, summary = run_training_pipeline(openml_limit=args.openml_limit, n_jobs=args.n_jobs)
        if predictor is not None and not summary.empty:
            print("Training summary rows:", len(summary))
            print("Accuracy:", round(float(predictor.metrics.get("accuracy", 0.0)), 4))
            print("Top-3 Accuracy:", round(float(predictor.metrics.get("top_3_accuracy", 0.0)), 4))

    if args.predict_csv:
        predictor = MetaModelPredictor.load()
        result = recommend_for_csv(args.predict_csv, predictor, target_column=args.target)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
