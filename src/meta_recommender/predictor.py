"""Meta-model training and inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import META_FEATURE_ORDER, MODEL_PATH, RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class MetaModelPredictor:
    model: RandomForestClassifier
    feature_order: list[str]

    @classmethod
    def train(cls, meta_X: pd.DataFrame, meta_y: pd.Series) -> "MetaModelPredictor":
        meta_X = meta_X.copy()
        for col in META_FEATURE_ORDER:
            if col not in meta_X.columns:
                meta_X[col] = 0.0
        meta_X = meta_X[META_FEATURE_ORDER].fillna(meta_X.median(numeric_only=True)).fillna(0.0)

        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
        )
        rf.fit(meta_X.values, meta_y.values)
        return cls(model=rf, feature_order=META_FEATURE_ORDER.copy())

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model, "feature_order": self.feature_order}
        joblib.dump(payload, path)
        logger.info("Saved meta-model to %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "MetaModelPredictor":
        payload = joblib.load(path)
        return cls(model=payload["model"], feature_order=payload["feature_order"])

    def _prepare_vector(self, meta_features: dict[str, float]) -> np.ndarray:
        row = pd.DataFrame([meta_features])
        for col in self.feature_order:
            if col not in row.columns:
                row[col] = np.nan
        row = row[self.feature_order]
        row = row.fillna(row.median(numeric_only=True)).fillna(0.0)
        return row.values

    def predict_best_model(self, meta_features: dict[str, float]) -> str:
        vec = self._prepare_vector(meta_features)
        return str(self.model.predict(vec)[0])

    def predict_top_k_models(self, meta_features: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
        vec = self._prepare_vector(meta_features)
        proba = self.model.predict_proba(vec)[0]
        classes = self.model.classes_
        sorted_idx = np.argsort(proba)[::-1][:k]
        return [(str(classes[i]), float(proba[i])) for i in sorted_idx]
