"""End-to-end pipeline to construct and train the meta-learning system."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .config import DEFAULT_OPENML_SIZE, META_DATASET_PATH, META_FEATURE_ORDER
from .data_loader import load_openml_datasets
from .evaluator import evaluate_models
from .features import clean_X, extract_meta_features
from .logging_utils import setup_logging
from .predictor import MetaModelPredictor

logger = logging.getLogger(__name__)


@dataclass
class MetaRecord:
    dataset_id: int
    dataset_name: str
    best_model: str
    task_type: str
    n_rows: int
    n_cols: int
    model_scores: dict[str, float]
    model_metrics: dict[str, dict[str, float]]
    meta_features: dict[str, float]

def _process_bundle(bundle) -> MetaRecord | None:
    X = clean_X(bundle.X)
    y = bundle.y
    meta = extract_meta_features(X, y)
    best_model, scores, task_type, metrics = evaluate_models(X, y)
    if not best_model:
        return None
    return MetaRecord(
        dataset_id=bundle.dataset_id,
        dataset_name=bundle.name,
        best_model=best_model,
        task_type=task_type,
        n_rows=len(X),
        n_cols=X.shape[1],
        model_scores=scores,
        model_metrics=metrics,
        meta_features=meta,
    )


def run_training_pipeline(openml_limit: int = DEFAULT_OPENML_SIZE, n_jobs: int = 2) -> tuple[MetaModelPredictor | None, pd.DataFrame]:
    """Build meta-dataset from OpenML datasets and train meta-model."""
    records: list[MetaRecord] = []

    bundles = list(load_openml_datasets(limit=openml_limit))
    with ProcessPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        futures = {pool.submit(_process_bundle, b): b for b in bundles}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
            bundle = futures[future]
            try:
                rec = future.result()
                if rec is None:
                    logger.info("Skipping dataset %s due to missing valid model scores.", bundle.name)
                    continue
                records.append(rec)
                logger.info("Best model for %s: %s", bundle.name, rec.best_model)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error processing dataset %s: %s", bundle.name, exc)

    if not records:
        logger.error("No valid datasets produced a meta-record.")
        return None, pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                **asdict(r),
                **{f"meta_{k}": v for k, v in r.meta_features.items()},
            }
            for r in records
        ]
    )

    meta_X = pd.DataFrame([r.meta_features for r in records])[META_FEATURE_ORDER]
    meta_y = pd.Series([r.best_model for r in records], name="best_model")

    logger.info("Meta label distribution: %s", meta_y.value_counts().to_dict())

    predictor, meta_eval = MetaModelPredictor.train(meta_X, meta_y)
    predictor.save()
    logger.info("Meta-model evaluation: %s", meta_eval)
    df.to_csv(META_DATASET_PATH, index=False)
    logger.info("Saved intermediate meta-dataset to %s", META_DATASET_PATH)
    logger.info("Training complete using %s datasets.", len(records))

    return predictor, df


def recommend_for_csv(csv_path: str, predictor: MetaModelPredictor) -> dict:
    """Run recommendation on a user-provided CSV file."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    # Fallback when target is unknown for online recommendation.
    pseudo_target = pd.Series([0] * len(df), name="target")
    meta_features = extract_meta_features(df, pseudo_target)

    best = predictor.predict_best_model(meta_features)
    top3 = predictor.predict_top_k_models(meta_features, k=3)

    return {
        "best_model": best,
        "top_3": [{"model": m, "probability": p} for m, p in top3],
        "meta_features": meta_features,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-learning model recommender")
    parser.add_argument("--train", action="store_true", help="Train the meta-model on OpenML datasets.")
    parser.add_argument("--openml-limit", type=int, default=DEFAULT_OPENML_SIZE, help="Number of OpenML datasets to process.")
    parser.add_argument("--n-jobs", type=int, default=2, help="Parallel workers for dataset processing.")
    parser.add_argument("--predict-csv", type=str, default=None, help="Path to CSV file for prediction.")
    return parser.parse_args()


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()

    if args.train:
        predictor, summary = run_training_pipeline(openml_limit=args.openml_limit, n_jobs=args.n_jobs)
        if predictor is not None and not summary.empty:
            print("Training summary rows:", len(summary))

    if args.predict_csv:
        predictor = MetaModelPredictor.load()
        result = recommend_for_csv(args.predict_csv, predictor)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
