"""Meta-model training and inference."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_TOP_K, META_EVALUATION_PATH, META_FEATURE_ORDER, MODEL_PATH, RANDOM_STATE, SCALER_PATH

logger = logging.getLogger(__name__)


@dataclass
class MetaModelPredictor:
    """Wrapper for the trained meta-model and associated artifacts."""

    model: RandomForestClassifier
    scaler: StandardScaler
    feature_order: list[str]
    fill_values: dict[str, float]
    metrics: dict[str, object]

    @staticmethod
    def _prepare_training_frame(meta_X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        frame = meta_X.copy()
        for col in META_FEATURE_ORDER:
            if col not in frame.columns:
                frame[col] = 0.0
        frame = frame[META_FEATURE_ORDER]
        fill_values = frame.median(numeric_only=True).fillna(0.0).to_dict()
        frame = frame.fillna(fill_values).fillna(0.0)
        return frame, {k: float(v) for k, v in fill_values.items()}

    @staticmethod
    def top_k_accuracy(model: RandomForestClassifier, X_scaled: np.ndarray, y_true: pd.Series, k: int = DEFAULT_TOP_K) -> float:
        """Compute top-k accuracy from class probabilities."""
        proba = model.predict_proba(X_scaled)
        classes = model.classes_
        top_idx = np.argsort(proba, axis=1)[:, ::-1][:, :k]
        top_classes = classes[top_idx]
        hits = [label in preds for label, preds in zip(y_true, top_classes, strict=False)]
        return float(np.mean(hits)) if hits else 0.0

    @classmethod
    def evaluate_holdout(
        cls,
        meta_X: pd.DataFrame,
        meta_y: pd.Series,
        model: RandomForestClassifier | None = None,
    ) -> dict[str, object]:
        """Evaluate the meta-model on a held-out split."""
        meta_X, _ = cls._prepare_training_frame(meta_X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(meta_X.values)

        stratify = meta_y if meta_y.nunique() > 1 and meta_y.value_counts().min() >= 2 else None
        if len(meta_X) >= 5 and meta_y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                meta_y,
                test_size=0.25,
                random_state=RANDOM_STATE,
                stratify=stratify,
            )
        else:
            X_train, X_test, y_train, y_test = X_scaled, X_scaled, meta_y, meta_y

        if model is None:
            model = RandomForestClassifier(
                n_estimators=400,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                min_samples_leaf=1,
            )
            model.fit(X_train, y_train.values)

        y_pred = model.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "top_3_accuracy": float(cls.top_k_accuracy(model, X_test, y_test, k=DEFAULT_TOP_K)),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=model.classes_).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "labels": [str(label) for label in model.classes_],
        }

    @classmethod
    def train(cls, meta_X: pd.DataFrame, meta_y: pd.Series) -> "MetaModelPredictor":
        """Train the probabilistic meta-model and compute held-out metrics."""
        meta_X, fill_values = cls._prepare_training_frame(meta_X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(meta_X.values)

        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=1,
        )
        metrics = cls.evaluate_holdout(meta_X, meta_y, model=rf)
        rf.fit(X_scaled, meta_y.values)

        META_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_EVALUATION_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return cls(
            model=rf,
            scaler=scaler,
            feature_order=META_FEATURE_ORDER.copy(),
            fill_values=fill_values,
            metrics=metrics,
        )

    def save(self, path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> None:
        """Persist model and scaler artifacts to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_order": self.feature_order,
            "fill_values": self.fill_values,
            "metrics": self.metrics,
        }
        joblib.dump(payload, path)
        joblib.dump(self.scaler, scaler_path)
        logger.info("Saved meta-model to %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> "MetaModelPredictor":
        """Load persisted model and scaler artifacts."""
        payload = joblib.load(path)
        scaler = joblib.load(scaler_path)
        return cls(
            model=payload["model"],
            scaler=scaler,
            feature_order=payload["feature_order"],
            fill_values=payload.get("fill_values", {}),
            metrics=payload.get("metrics", {}),
        )

    def _prepare_vector(self, meta_features: dict[str, float]) -> np.ndarray:
        row = pd.DataFrame([meta_features])
        for col in self.feature_order:
            if col not in row.columns:
                row[col] = np.nan
        row = row[self.feature_order]
        row = row.fillna(self.fill_values).fillna(0.0)
        return self.scaler.transform(row.values)

    def predict_best_model(self, meta_features: dict[str, float]) -> str:
        """Predict the single best base learner label."""
        vec = self._prepare_vector(meta_features)
        return str(self.model.predict(vec)[0])

    def predict_top_k_models(self, meta_features: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
        """Predict ranked model recommendations with probabilities."""
        vec = self._prepare_vector(meta_features)
        proba = self.model.predict_proba(vec)[0]
        classes = self.model.classes_
        sorted_idx = np.argsort(proba)[::-1][:k]
        return [(str(classes[i]), float(proba[i])) for i in sorted_idx]
