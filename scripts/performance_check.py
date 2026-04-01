"""Performance checks for the trained meta-model."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split

from meta_recommender.config import META_DATASET_PATH, META_FEATURE_ORDER, RANDOM_STATE
from meta_recommender.predictor import MetaModelPredictor


def main() -> None:
    df = pd.read_csv(META_DATASET_PATH)
    X = df[[f"meta_{c}" for c in META_FEATURE_ORDER]].rename(columns={f"meta_{c}": c for c in META_FEATURE_ORDER})
    y = df["best_model"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    predictor, evaluation = MetaModelPredictor.train(X_train, y_train)

    proba = predictor.model.predict_proba(predictor.scaler.transform(X_test.values))
    y_pred = predictor.model.classes_[proba.argmax(axis=1)]

    top3_acc = top_k_accuracy_score(y_test, proba, k=min(3, proba.shape[1]), labels=predictor.model.classes_)
    acc = (y_pred == y_test.values).mean()

    print("Hold-out accuracy:", round(float(acc), 4))
    print("Top-3 accuracy:", round(float(top3_acc), 4))
    print("Train split summary:", evaluation["accuracy"])


if __name__ == "__main__":
    main()
