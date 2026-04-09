"""Simple CLI wrapper for recommending models from a CSV dataset."""

from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from meta_recommender.logging_utils import setup_logging
from meta_recommender.pipeline import recommend_for_csv
from meta_recommender.predictor import MetaModelPredictor


def parse_args() -> argparse.Namespace:
    """Parse main.py CLI arguments."""
    parser = argparse.ArgumentParser(description="Recommend the best ML model for a CSV dataset.")
    parser.add_argument("--file", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--mode", choices=["accurate", "fast"], default="accurate", help="Recommendation mode.")
    parser.add_argument("--benchmark-automl", action="store_true", help="Benchmark against installed AutoML tools.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging for debugging.")
    return parser.parse_args()


def main() -> None:
    """Load trained artifacts and print ranked recommendations."""
    args = parse_args()
    setup_logging(debug=args.debug)
    start = perf_counter()

    input_path = Path(args.file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found: {input_path}")

    print(f"Loading predictor and scoring dataset: {input_path}")
    print(f"Target column: {args.target}")
    print(f"Mode: {args.mode}")

    predictor = MetaModelPredictor.load()
    result = recommend_for_csv(
        str(input_path),
        predictor,
        target_column=args.target,
        mode=args.mode,
        benchmark_automl=args.benchmark_automl,
    )

    elapsed = perf_counter() - start
    print(f"Finished in {elapsed:.2f}s")
    print(f"Best model: {result['best_model']}")
    print("Top 3 models:")
    for entry in result["top_3"]:
        print(f"- {entry['model']}: {entry['probability']:.4f}")
    print(json.dumps(result["dataset_summary"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Prediction interrupted by user.")
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Prediction failed: {exc}")
        raise
