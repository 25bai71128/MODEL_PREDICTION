"""Centralized logging setup."""

from __future__ import annotations

import logging

from .config import LOG_FILE


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
    )
