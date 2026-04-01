"""Shared logging and error handling helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import LOG_PATH


def setup_logging(log_path: Path = LOG_PATH) -> None:
    """Configure console and file logging once for the process."""
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def log_exception(logger: logging.Logger, stage: str, dataset_name: str, exc: Exception) -> None:
    """Persist a structured exception message for centralized troubleshooting."""
    logger.exception("%s failure for %s: %s", stage, dataset_name, exc)
