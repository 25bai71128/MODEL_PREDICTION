"""Fast Streamlit app for lightweight ML model recommendation."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:  # noqa: BLE001
    HAS_XGBOOST = False


st.set_page_config(page_title="ML Model Recommender", layout="wide")


def detect_problem_type(target: pd.Series) -> str:
    """Infer whether the uploaded task is classification or regression."""
    y = target.dropna()
    if y.empty:
        return "classification"
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    unique_count = y.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y), 1)
    if unique_count <= 20 and unique_ratio < 0.2:
        return "classification"
    return "regression"


def guess_target_column(df: pd.DataFrame) -> str:
    """Guess a likely target column using common naming heuristics."""
    priority_names = [
        "target",
        "label",
        "class",
        "y",
        "output",
        "prediction",
        "response",
        "outcome",
    ]
    lower_to_original = {column.lower(): column for column in df.columns}
    for name in priority_names:
        if name in lower_to_original:
            return lower_to_original[name]
    return df.columns[-1]


@st.cache_data(show_spinner=False)
def load_uploaded_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """Load the uploaded CSV into a DataFrame."""
    return pd.read_csv(BytesIO(file_bytes))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a quick preprocessing pipeline for mixed tabular data."""
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def _candidate_models(problem_type: str) -> list[tuple[str, Any]]:
    """Return a lightweight candidate pool."""
    if problem_type == "classification":
        models: list[tuple[str, Any]] = [
            ("Logistic Regression", LogisticRegression(max_iter=400)),
            ("Random Forest", RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=1)),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
            ("Decision Tree", DecisionTreeClassifier(max_depth=8, random_state=42)),
        ]
        if HAS_XGBOOST:
            models.append(
                (
                    "XGBoost",
                    XGBClassifier(
                        n_estimators=80,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        eval_metric="logloss",
                        n_jobs=1,
                    ),
                )
            )
        return models

    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=1)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
        ("Decision Tree", DecisionTreeRegressor(max_depth=8, random_state=42)),
    ]
    if HAS_XGBOOST:
        models.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=80,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    objective="reg:squarederror",
                    n_jobs=1,
                ),
            )
        )
    return models


def _build_pipeline(model: Any, X: pd.DataFrame) -> Pipeline:
    """Attach preprocessing to a model."""
    return Pipeline([("preprocessor", build_preprocessor(X)), ("model", clone(model))])


def _fit_and_predict_model(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, np.ndarray]:
    """Fit a candidate model and return predictions on the test split."""
    pipeline = _build_pipeline(model, X)

    if model_name == "XGBoost" and isinstance(model, XGBClassifier):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train.astype(str))
        pipeline.fit(X_train, y_train_encoded)
        y_pred_encoded = pipeline.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))
        return pipeline, y_pred

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return pipeline, y_pred


def _score_predictions(problem_type: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute task-appropriate metrics."""
    if problem_type == "classification":
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return {"primary_score": float(f1), "accuracy": float(accuracy), "f1_score": float(f1)}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    primary = 1.0 / (1.0 + rmse)
    return {"primary_score": primary, "rmse": rmse, "r2_score": r2}


@st.cache_data(show_spinner=False)
def benchmark_models(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Benchmark lightweight models on the uploaded dataset."""
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    working_df = df.copy()
    y = working_df.pop(target_column)
    problem_type = detect_problem_type(y)

    valid_index = y.notna()
    working_df = working_df.loc[valid_index].reset_index(drop=True)
    y = y.loc[valid_index].reset_index(drop=True)

    if working_df.empty or y.empty:
        raise ValueError("No valid rows remain after removing missing target values.")
    if working_df.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the target column.")
    if len(working_df) < 10:
        raise ValueError("Please upload a dataset with at least 10 rows for a stable recommendation.")
    if problem_type == "classification" and y.nunique(dropna=True) < 2:
        raise ValueError("Classification requires at least two target classes.")

    stratify = y if problem_type == "classification" and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        working_df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    leaderboard: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    failures: list[str] = []

    for model_name, model in _candidate_models(problem_type):
        try:
            pipeline, y_pred = _fit_and_predict_model(
                model_name=model_name,
                model=model,
                X=working_df,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
            )
            metrics = _score_predictions(problem_type, y_test, y_pred)
            leaderboard.append({"model": model_name, **metrics})
            fitted_models[model_name] = pipeline
            predictions[model_name] = y_pred
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{model_name}: {exc}")

    if not leaderboard:
        raise RuntimeError("All candidate models failed to train on this dataset.")

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("primary_score", ascending=False).reset_index(drop=True)
    leaderboard_df["confidence"] = leaderboard_df["primary_score"] / leaderboard_df["primary_score"].sum()
    leaderboard_df["confidence"] = leaderboard_df["confidence"].fillna(0.0).round(4)

    best_model_name = str(leaderboard_df.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]
    best_predictions = predictions[best_model_name]

    transformed_feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    best_estimator = best_pipeline.named_steps["model"]
    importance_df = pd.DataFrame(columns=["feature", "importance"])
    if hasattr(best_estimator, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "importance": best_estimator.feature_importances_,
            }
        ).sort_values("importance", ascending=False).head(15)
    elif hasattr(best_estimator, "coef_"):
        coefficients = np.ravel(np.abs(best_estimator.coef_))
        importance_df = pd.DataFrame(
            {
                "feature": transformed_feature_names[: len(coefficients)],
                "importance": coefficients,
            }
        ).sort_values("importance", ascending=False).head(15)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_column in numeric_df.columns:
        corr_df = numeric_df.corr(numeric_only=True).round(2)
    else:
        corr_df = numeric_df.corr(numeric_only=True).round(2)

    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "leaderboard": leaderboard_df,
        "best_model": best_model_name,
        "best_confidence": float(leaderboard_df.iloc[0]["confidence"]),
        "y_test": y_test.reset_index(drop=True),
        "best_predictions": pd.Series(best_predictions).reset_index(drop=True),
        "importance_df": importance_df.reset_index(drop=True),
        "correlation_df": corr_df,
        "failures": failures,
        "summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values": int(df.isna().sum().sum()),
        },
    }


def _heatmap_from_correlation(corr_df: pd.DataFrame) -> alt.Chart | None:
    """Build a compact correlation heatmap."""
    if corr_df.empty or corr_df.shape[0] < 2:
        return None
    trimmed = corr_df.iloc[:20, :20]
    melted = trimmed.reset_index(names="feature_x").melt("feature_x", var_name="feature_y", value_name="correlation")
    return (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X("feature_x:N", sort=None, title="Feature"),
            y=alt.Y("feature_y:N", sort=None, title="Feature"),
            color=alt.Color("correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["feature_x", "feature_y", "correlation"],
        )
        .properties(height=360)
    )


def _feature_importance_chart(importance_df: pd.DataFrame) -> alt.Chart | None:
    """Build a feature-importance bar chart."""
    if importance_df.empty:
        return None
    return (
        alt.Chart(importance_df)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", "importance"],
        )
        .properties(height=360)
    )


def _confusion_matrix_chart(y_true: pd.Series, y_pred: pd.Series) -> alt.Chart:
    """Build a confusion matrix heatmap."""
    labels = sorted(pd.unique(pd.concat([y_true, y_pred], ignore_index=True)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    melted = matrix_df.reset_index(names="actual").melt("actual", var_name="predicted", value_name="count")
    return (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", title="Predicted"),
            y=alt.Y("actual:N", title="Actual"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["actual", "predicted", "count"],
        )
        .properties(height=320)
    )


def _regression_plot(y_true: pd.Series, y_pred: pd.Series) -> alt.Chart:
    """Build an actual-vs-predicted scatter plot."""
    chart_df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    limits = [
        float(min(chart_df["actual"].min(), chart_df["predicted"].min())),
        float(max(chart_df["actual"].max(), chart_df["predicted"].max())),
    ]
    reference_df = pd.DataFrame({"x": limits, "y": limits})
    scatter = (
        alt.Chart(chart_df)
        .mark_circle(size=70, opacity=0.75)
        .encode(x=alt.X("actual:Q", title="Actual"), y=alt.Y("predicted:Q", title="Predicted"))
    )
    line = alt.Chart(reference_df).mark_line(color="red").encode(x="x:Q", y="y:Q")
    return (scatter + line).properties(height=320)


def render_app() -> None:
    """Render the Streamlit application."""
    st.title("ML Model Recommendation Studio")
    st.caption("Upload a CSV and get a fast leaderboard of lightweight models without blocking the app.")

    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to start.")
        return

    try:
        df = load_uploaded_dataframe(uploaded_file.getvalue())
        guessed_target = guess_target_column(df)
        target_index = df.columns.get_loc(guessed_target)
        target_column = st.selectbox("Detected target column", options=df.columns.tolist(), index=target_index)

        summary_columns = st.columns(4)
        summary_columns[0].metric("Rows", int(df.shape[0]))
        summary_columns[1].metric("Columns", int(df.shape[1]))
        summary_columns[2].metric("Missing Values", int(df.isna().sum().sum()))
        summary_columns[3].metric("Detected Target", target_column)

        preview_expander = st.expander("Dataset Preview", expanded=False)
        with preview_expander:
            st.dataframe(df.head(20), use_container_width=True)

        if st.button("Analyze Dataset", type="primary"):
            with st.spinner("Analyzing dataset and ranking candidate models..."):
                result = benchmark_models(df, target_column)

            st.subheader("Recommendation")
            recommendation_columns = st.columns(3)
            recommendation_columns[0].metric("Problem Type", result["problem_type"].title())
            recommendation_columns[1].metric("Best Model", result["best_model"])
            recommendation_columns[2].metric("Confidence", f"{result['best_confidence']:.1%}")

            st.subheader("Model Leaderboard")
            leaderboard = result["leaderboard"].copy()
            display_columns = ["model", "confidence", "accuracy", "f1_score"] if result["problem_type"] == "classification" else ["model", "confidence", "rmse", "r2_score"]
            leaderboard = leaderboard[[column for column in display_columns if column in leaderboard.columns]]
            st.dataframe(leaderboard, use_container_width=True)

            if result["failures"]:
                st.warning("Some models failed but results are still available.")
                st.write(result["failures"])

            viz_left, viz_right = st.columns(2)

            with viz_left:
                st.subheader("Feature Importance")
                importance_chart = _feature_importance_chart(result["importance_df"])
                if importance_chart is None:
                    st.info("Feature importance is not available for the selected best model.")
                else:
                    st.altair_chart(importance_chart, use_container_width=True)

            with viz_right:
                st.subheader("Dataset Correlation Heatmap")
                heatmap = _heatmap_from_correlation(result["correlation_df"])
                if heatmap is None:
                    st.info("Correlation heatmap needs at least two numeric columns.")
                else:
                    st.altair_chart(heatmap, use_container_width=True)

            st.subheader("Evaluation View")
            if result["problem_type"] == "classification":
                st.altair_chart(
                    _confusion_matrix_chart(result["y_test"], result["best_predictions"]),
                    use_container_width=True,
                )
            else:
                st.altair_chart(
                    _regression_plot(result["y_test"], result["best_predictions"]),
                    use_container_width=True,
                )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to analyze the uploaded dataset: {exc}")

if __name__ == "__main__":
    render_app()
