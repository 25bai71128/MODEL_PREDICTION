"""FastAPI service for the meta-learning recommender."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import recommend_for_dataframe
from .predictor import MetaModelPredictor

app = FastAPI(title="Meta Recommender API", version="2.0.0")
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="Tabular dataset rows as JSON objects.")
    target_column: str | None = Field(default=None, description="Optional target column name.")
    mode: str = Field(default="accurate", pattern="^(accurate|fast)$")
    benchmark_automl: bool = Field(default=False, description="Benchmark against installed AutoML tools.")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/meta-model/metrics")
def meta_model_metrics() -> dict[str, Any]:
    try:
        predictor = MetaModelPredictor.load()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load meta-model metrics: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return predictor.metrics


@app.post("/recommend")
def recommend(request: PredictionRequest) -> dict[str, Any]:
    if not request.records:
        raise HTTPException(status_code=400, detail="No records were provided.")

    try:
        start = perf_counter()
        predictor = MetaModelPredictor.load()
        df = pd.DataFrame(request.records)
        result = recommend_for_dataframe(
            df,
            predictor,
            target_column=request.target_column,
            mode=request.mode,
            benchmark_automl=request.benchmark_automl,
        )
        logger.info(
            "API recommendation complete in %.2fs for %s rows.",
            perf_counter() - start,
            len(df),
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("API recommendation failure: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
