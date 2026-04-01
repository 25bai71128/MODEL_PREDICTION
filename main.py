"""User-facing CLI for model recommendation."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from meta_recommender.features import detect_task_type, extract_meta_features
from meta_recommender.logging_utils import setup_logging
from meta_recommender.predictor import MetaModelPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend best ML model for a CSV dataset.")
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=False, default=None, help="Target column name")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    df = pd.read_csv(args.file)
    if args.target and args.target in df.columns:
        y = df[args.target]
        X = df.drop(columns=[args.target])
    else:
        y = pd.Series([0] * len(df), name="target")
        X = df

    task = detect_task_type(y)
    meta = extract_meta_features(X, y)
    predictor = MetaModelPredictor.load()

    result = {
        "problem_type": task,
        "best_model": predictor.predict_best_model(meta),
        "top_3": predictor.predict_top_k_models(meta, 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
