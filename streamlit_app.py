"""Streamlit app for model recommendation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from meta_recommender.features import detect_task_type, extract_meta_features
from meta_recommender.predictor import MetaModelPredictor

st.set_page_config(page_title="Meta Model Recommender", layout="wide")
st.title("Meta-Learning Model Recommender")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        with st.spinner("Loading dataset and preparing recommendations..."):
            df = pd.read_csv(uploaded)

            st.subheader("Dataset Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(df))
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing values", int(df.isna().sum().sum()))

            target_col = st.selectbox("Select target column (optional)", options=["<none>"] + list(df.columns))
            if target_col != "<none>":
                y = df[target_col]
                X = df.drop(columns=[target_col])
            else:
                y = pd.Series([0] * len(df), name="target")
                X = df

            task_type = detect_task_type(y)
            st.info(f"Detected task type: **{task_type}**")

            meta_features = extract_meta_features(X, y)

            if st.checkbox("Show meta-features (debug)"):
                st.json(meta_features)

            predictor = MetaModelPredictor.load()
            best = predictor.predict_best_model(meta_features)
            top3 = predictor.predict_top_k_models(meta_features, k=3)

        st.subheader("Recommendations")
        st.success(f"Best Model: **{best}**")

        chart_df = pd.DataFrame({"model": [m for m, _ in top3], "probability": [p for _, p in top3]}).set_index("model")
        st.bar_chart(chart_df)
        st.write("Top 3 models:")
        st.table(chart_df.reset_index())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to process file: {exc}")
