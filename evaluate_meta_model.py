"""Print persisted meta-model evaluation diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from meta_recommender.logging_utils import setup_logging
from meta_recommender.predictor import MetaModelPredictor


def main() -> None:
    """Print the saved ranking and baseline comparison metrics."""
    setup_logging()
    predictor = MetaModelPredictor.load()
    print(json.dumps(predictor.metrics, indent=2))


if __name__ == "__main__":
    main()
