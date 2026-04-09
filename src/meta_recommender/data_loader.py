"""OpenML dataset loading utilities with robust validation and hard timeouts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from .config import (
    DATASET_LOAD_TIMEOUT_SECONDS,
    DATASET_RETRY_COUNT,
    DEBUG_DATASET_LIMIT,
    DEFAULT_OPENML_SIZE,
    MAX_DATASET_ROWS,
    MAX_DATASET_SCAN_MULTIPLIER,
    MIN_ROWS_FOR_EVAL,
    RANDOM_STATE,
)
from .logging_utils import log_exception
from .runtime_utils import HardTimeoutError, run_with_hard_timeout

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
    joined = joined.drop_duplicates()
    y_out = joined.pop("__target__")
    return joined, y_out


def _sample_if_large(X: pd.DataFrame, y: pd.Series, max_rows: int = MAX_DATASET_ROWS) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y
    sampled = X.sample(n=max_rows, random_state=RANDOM_STATE)
    y = y.loc[sampled.index]
    return sampled.reset_index(drop=True), y.reset_index(drop=True)


def _normalize_target_attribute(target_attr: str | list[str] | tuple[str, ...] | None) -> str | None:
    """Return a single supported target attribute, or None if the dataset should be skipped."""
    if target_attr is None:
        return None

    if isinstance(target_attr, str):
        stripped = target_attr.strip()
        if not stripped:
            return None
        if "," in stripped:
            parts = [part.strip() for part in stripped.split(",") if part.strip()]
            return parts[0] if len(parts) == 1 else None
        return stripped

    if isinstance(target_attr, (list, tuple)):
        if len(target_attr) != 1:
            return None
        return str(target_attr[0]).strip() or None

    return str(target_attr).strip() or None


def _load_single_dataset(did: int, name: str) -> dict[str, object]:
    """Load, validate, and return one OpenML dataset payload."""
    import openml

    try:
        ds = openml.datasets.get_dataset(did)
        target_attr = _normalize_target_attribute(ds.default_target_attribute)
        if target_attr is None:
            return {"status": "skip", "reason": "no supported single target attribute"}

        X, y, _, _ = ds.get_data(target=target_attr, dataset_format="dataframe")
        X = _safe_dataframe(X)
        y = _safe_target(y, target_attr)

        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].reset_index(drop=True)
        y = y.iloc[:min_len].reset_index(drop=True)

        X, y = _drop_duplicates_align(X, y)
        X, y = _sample_if_large(X, y)

        if X.empty or y.empty:
            return {"status": "skip", "reason": "dataset empty after cleaning"}

        return {
            "status": "ok",
            "bundle": DatasetBundle(dataset_id=did, name=name, X=X, y=y),
        }
    except NotImplementedError as exc:
        return {"status": "skip", "reason": str(exc)}
    except ValueError as exc:
        if "Target is missing" in str(exc) or "Target column is empty" in str(exc):
            return {"status": "skip", "reason": str(exc)}
        raise
    except TypeError as exc:
        if "factorize requires" in str(exc) and "got list" in str(exc):
            return {"status": "skip", "reason": "incompatible OpenML sparse/list format"}
        raise
    except PermissionError as exc:
        if "[WinError 32]" in str(exc):
            return {"status": "skip", "reason": "OpenML cache file is locked"}
        raise


def _load_dataset_with_retries(
    did: int,
    name: str,
    *,
    timeout_seconds: int,
    retries: int,
) -> DatasetBundle | None:
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            timed = run_with_hard_timeout(
                _load_single_dataset,
                kwargs={"did": did, "name": name},
                timeout_seconds=timeout_seconds,
                stage_name=f"dataset load {did}",
            )
            payload = timed.value
            status = payload.get("status")
            if status == "ok":
                bundle = payload["bundle"]
                logger.info(
                    "Loaded dataset %s (%s) with shape %s in %.2fs",
                    did,
                    name,
                    bundle.X.shape,
                    timed.elapsed_seconds,
                )
                return bundle

            reason = str(payload.get("reason", "skipped"))
            logger.info("Skipping dataset %s (%s): %s.", did, name, reason)
            return None
        except HardTimeoutError:
            logger.warning(
                "Dataset load timed out for %s (%s) on attempt %s/%s after %ss.",
                did,
                name,
                attempt,
                attempts,
                timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt < attempts:
                logger.warning(
                    "Dataset load retry for %s (%s) after failure on attempt %s/%s: %s",
                    did,
                    name,
                    attempt,
                    attempts,
                    exc,
                )
            else:
                log_exception(logger, "dataset loading", name, exc)
    return None


def load_openml_datasets(
    limit: int = DEFAULT_OPENML_SIZE,
    *,
    debug: bool = False,
    retries: int = DATASET_RETRY_COUNT,
    timeout_seconds: int = DATASET_LOAD_TIMEOUT_SECONDS,
    max_scan_multiplier: int = MAX_DATASET_SCAN_MULTIPLIER,
) -> Iterator[DatasetBundle]:
    """Yield valid dataset bundles from OpenML with bounded runtime and retries."""
    import openml

    effective_limit = min(limit, DEBUG_DATASET_LIMIT) if debug else limit
    logger.info(
        "Starting OpenML ingestion for up to %s datasets (debug=%s, load_timeout=%ss, retries=%s).",
        effective_limit,
        debug,
        timeout_seconds,
        retries,
    )

    datasets = openml.datasets.list_datasets(output_format="dataframe")
    if "NumberOfInstances" in datasets.columns:
        datasets = datasets.sort_values("NumberOfInstances", ascending=True)
        datasets = datasets[datasets["NumberOfInstances"].fillna(0) >= MIN_ROWS_FOR_EVAL]
    if "default_target_attribute" in datasets.columns:
        datasets = datasets[
            datasets["default_target_attribute"].apply(lambda value: _normalize_target_attribute(value) is not None)
        ]

    max_attempts = min(len(datasets), max(1, effective_limit) * max_scan_multiplier)
    yielded = 0
    scanned = 0
    logger.info("Scanning at most %s OpenML candidates to obtain %s usable datasets.", max_attempts, effective_limit)

    for _, row in datasets.iterrows():
        if yielded >= effective_limit or scanned >= max_attempts:
            break

        scanned += 1
        did = int(row["did"])
        name = str(row.get("name", f"dataset_{did}"))
        logger.info("Dataset scan %s/%s: %s (%s)", scanned, max_attempts, name, did)

        bundle = _load_dataset_with_retries(
            did,
            name,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        if bundle is None:
            continue

        yielded += 1
        yield bundle

    logger.info(
        "OpenML ingestion finished with %s usable datasets after scanning %s candidates.",
        yielded,
        scanned,
    )
