"""Streamlit UI for the meta-learning recommender."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from meta_recommender.logging_utils import setup_logging
from meta_recommender.pipeline import recommend_for_dataframe
from meta_recommender.predictor import MetaModelPredictor

setup_logging()

st.set_page_config(page_title="Meta Model Recommender", layout="wide")
st.title("Meta-Learning Model Recommender")
st.caption("Upload a tabular dataset and get the top model recommendations with confidence scores.")


@st.cache_resource
def load_predictor() -> MetaModelPredictor:
    """Load the trained meta-model once per Streamlit session."""
    return MetaModelPredictor.load()


uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Summary")
        summary_cols = st.columns(3)
        summary_cols[0].metric("Rows", int(df.shape[0]))
        summary_cols[1].metric("Columns", int(df.shape[1]))
        summary_cols[2].metric("Missing Values", int(df.isna().sum().sum()))

        target_column = st.selectbox("Select target column", options=df.columns.tolist())
        debug_mode = st.checkbox("Show meta-features debug section", value=False)

        if st.button("Recommend Models", type="primary"):
            with st.spinner("Analyzing dataset and ranking candidate models..."):
                predictor = load_predictor()
                result = recommend_for_dataframe(df, predictor, target_column=target_column)

            left, right = st.columns([1, 1])
            with left:
                st.subheader("Recommendation")
                st.success(f"Best model: {result['best_model']}")
                st.write(f"Detected problem type: `{result['problem_type']}`")
            with right:
                st.subheader("Top 3 Rankings")
                chart_df = pd.DataFrame(result["top_3"])
                chart_df = chart_df.rename(columns={"model": "Model", "probability": "Probability"})
                st.bar_chart(chart_df.set_index("Model"))
                st.dataframe(chart_df, use_container_width=True)

            if debug_mode:
                st.subheader("Meta-Features")
                st.json(result["meta_features"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to process the uploaded dataset: {exc}")
else:
    st.info("Upload a CSV file to start.")
