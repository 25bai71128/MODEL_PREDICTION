"""Simple CLI wrapper for recommending models from a CSV dataset."""

from __future__ import annotations

import argparse
import json
import sys
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
    return parser.parse_args()


def main() -> None:
    """Load trained artifacts and print ranked recommendations."""
    setup_logging()
    args = parse_args()
    predictor = MetaModelPredictor.load()
    result = recommend_for_csv(args.file, predictor, target_column=args.target)
    print(f"Best model: {result['best_model']}")
    print("Top 3 models:")
    for entry in result["top_3"]:
        print(f"- {entry['model']}: {entry['probability']:.4f}")
    print(json.dumps(result["dataset_summary"], indent=2))


if __name__ == "__main__":
    main()
