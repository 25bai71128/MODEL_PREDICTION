"""OpenML dataset loading utilities with robust validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from .config import DEFAULT_OPENML_SIZE, MAX_DATASET_ROWS, RANDOM_STATE
from .logging_utils import log_exception

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Container for an individual dataset."""

    dataset_id: int
    name: str
    X: pd.DataFrame
    y: pd.Series


def _safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Features are not a DataFrame.")
    if df.empty:
        raise ValueError("Dataset has zero rows or columns.")

    df = df.copy()
    full_missing_cols = [c for c in df.columns if df[c].isna().all()]
    if full_missing_cols:
        df = df.drop(columns=full_missing_cols)
        logger.debug("Dropped fully missing columns: %s", full_missing_cols)

    if df.empty:
        raise ValueError("Dataset became empty after dropping fully missing columns.")

    return df


def _safe_target(y: pd.Series | pd.DataFrame | None, target_name: str | None) -> pd.Series:
    if y is None:
        raise ValueError("Target is missing.")

    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("Target DataFrame is empty.")
        y = y.iloc[:, 0]

    if not isinstance(y, pd.Series):
        try:
            y = pd.Series(y, name=target_name or "target")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Target could not be converted to Series.") from exc

    if y.empty:
        raise ValueError("Target column is empty.")

    return y


def _drop_duplicates_align(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    joined = X.copy()
    joined["__target__"] = y.values
    before = len(joined)
    joined = joined.drop_duplicates()
    removed = before - len(joined)
    if removed:
        logger.debug("Removed %s duplicate rows.", removed)
    y_out = joined.pop("__target__")
    return joined, y_out


def _sample_if_large(X: pd.DataFrame, y: pd.Series, max_rows: int = MAX_DATASET_ROWS) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y
    sampled = X.sample(n=max_rows, random_state=RANDOM_STATE)
    y = y.loc[sampled.index]
    logger.info("Sampled dataset from %s to %s rows.", len(X), len(sampled))
    return sampled.reset_index(drop=True), y.reset_index(drop=True)


def load_openml_datasets(limit: int = DEFAULT_OPENML_SIZE) -> Iterator[DatasetBundle]:
    """Yield valid dataset bundles from OpenML, skipping invalid records safely."""
    import openml

    datasets = openml.datasets.list_datasets(output_format="dataframe")

    if "NumberOfInstances" in datasets.columns:
        datasets = datasets.sort_values("NumberOfInstances", ascending=True)

    yielded = 0
    for _, row in datasets.iterrows():
        if yielded >= limit:
            break

        did = int(row["did"])
        name = str(row.get("name", f"dataset_{did}"))
        try:
            ds = openml.datasets.get_dataset(did)
            target_attr = ds.default_target_attribute
            X, y, _, _ = ds.get_data(target=target_attr, dataset_format="dataframe")

            X = _safe_dataframe(X)
            y = _safe_target(y, target_attr)

            min_len = min(len(X), len(y))
            X = X.iloc[:min_len].reset_index(drop=True)
            y = y.iloc[:min_len].reset_index(drop=True)

            X, y = _drop_duplicates_align(X, y)
            X, y = _sample_if_large(X, y)

            if X.empty or y.empty:
                raise ValueError("Dataset empty after cleaning.")

            yielded += 1
            logger.info("Loaded dataset %s (%s) with shape %s", did, name, X.shape)
            yield DatasetBundle(dataset_id=did, name=name, X=X, y=y)
        except Exception as exc:  # noqa: BLE001
            log_exception(logger, "dataset loading", name, exc)
            continue
