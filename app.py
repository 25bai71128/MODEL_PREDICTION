"""AutoML Assistant: Intelligent Model Recommendation and Analysis."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

from meta_recommender.features import detect_task_type
from meta_recommender.pipeline import recommend_for_dataframe
from meta_recommender.predictor import MetaModelPredictor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "AutoML Assistant"
APP_SUBTITLE = "Intelligent model recommendation and data analysis."
DEMO_DATASET_NAME = "iris_dataset.csv"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def _inject_styles() -> None:
    """Apply a custom visual language for the Streamlit UI."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe7;
            --bg-alt: #fbf8f2;
            --panel: rgba(255, 255, 255, 0.84);
            --panel-strong: #fffdf9;
            --ink: #101828;
            --muted: #5b6472;
            --accent: #0f766e;
            --gold: #b7791f;
            --line: rgba(15, 23, 42, 0.1);
            --card-shadow: 0 22px 50px rgba(15, 23, 42, 0.09);
            --sidebar-bg: #0d1320;
            --sidebar-panel: rgba(255, 255, 255, 0.06);
            --sidebar-line: rgba(148, 163, 184, 0.18);
            --sidebar-ink: #f8fafc;
            --sidebar-muted: #cbd5e1;
        }

        html, body, [class*="css"] {
            font-family: "Aptos", "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 34%),
                radial-gradient(circle at top right, rgba(183, 121, 31, 0.12), transparent 24%),
                linear-gradient(180deg, #f9f5ee 0%, #eef4f2 48%, #f9f6f0 100%);
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.22), transparent 28%),
                linear-gradient(180deg, #0d1320 0%, #111827 56%, #18212f 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        div[data-testid="stSidebar"] * {
            color: var(--sidebar-ink);
        }

        div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] span,
        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--sidebar-muted) !important;
        }

        div[data-testid="stSidebar"] h1,
        div[data-testid="stSidebar"] h2,
        div[data-testid="stSidebar"] h3 {
            color: var(--sidebar-ink) !important;
        }

        div[data-testid="stSidebar"] [data-baseweb="select"] > div,
        div[data-testid="stSidebar"] .stNumberInput input,
        div[data-testid="stSidebar"] .stTextInput input,
        div[data-testid="stSidebar"] .stTextArea textarea {
            background: var(--sidebar-panel) !important;
            color: var(--sidebar-ink) !important;
            border: 1px solid var(--sidebar-line) !important;
            border-radius: 16px !important;
        }

        div[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px dashed rgba(255, 255, 255, 0.18);
            border-radius: 18px;
        }

        div[data-testid="stSidebar"] button {
            border-radius: 16px !important;
            min-height: 48px;
            font-weight: 600;
        }

        div[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
        div[data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
            color: #f8fafc !important;
            border: none !important;
            box-shadow: 0 18px 34px rgba(20, 184, 166, 0.18);
        }

        div[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
        div[data-testid="stSidebar"] button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.03) !important;
            color: var(--sidebar-ink) !important;
            border: 1px solid var(--sidebar-line) !important;
        }

        .hero-shell {
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.14), transparent 28%),
                radial-gradient(circle at bottom right, rgba(15, 118, 110, 0.22), transparent 32%),
                linear-gradient(135deg, rgba(12, 17, 29, 0.98), rgba(14, 65, 70, 0.96), rgba(21, 128, 61, 0.92));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 30px;
            padding: 1.9rem 1.8rem;
            color: #f9fafb;
            box-shadow: 0 30px 70px rgba(12, 17, 29, 0.22);
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        }

        .hero-kicker {
            font-size: 0.8rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            opacity: 0.75;
            margin-bottom: 0.6rem;
            font-family: "Trebuchet MS", "Verdana", sans-serif;
        }

        .hero-title {
            font-size: clamp(2.1rem, 3.7vw, 3.25rem);
            line-height: 1.03;
            margin: 0 0 0.7rem 0;
            font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
            font-weight: 700;
        }

        .hero-copy {
            max-width: 58rem;
            color: rgba(249, 250, 251, 0.88);
            font-size: 1.02rem;
            line-height: 1.6;
            margin: 0;
            font-family: "Aptos", "Segoe UI", sans-serif;
        }

        .mini-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.95rem;
            margin: 1rem 0 1.3rem 0;
        }

        .mini-card {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(250, 248, 243, 0.9));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.05rem;
            box-shadow: var(--card-shadow);
        }

        .mini-card h4 {
            margin: 0 0 0.35rem 0;
            font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
            color: var(--ink);
        }

        .mini-card p,
        .section-copy,
        .callout span {
            margin: 0;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.55;
            font-family: "Aptos", "Segoe UI", sans-serif;
        }

        .callout {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(250, 248, 243, 0.92));
            border: 1px solid var(--line);
            border-left: 5px solid var(--accent);
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: var(--card-shadow);
        }

        .callout strong {
            display: block;
            margin-bottom: 0.35rem;
            color: var(--ink);
            font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1rem;
        }

        .pill {
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
            color: #f8fafc;
            font-size: 0.86rem;
            font-family: "Aptos", "Segoe UI", sans-serif;
            backdrop-filter: blur(8px);
        }

        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--ink) !important;
            font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
            letter-spacing: -0.02em;
        }

        p, li, label, span, .stMarkdown, .stText, .stCaption {
            color: var(--muted);
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(251, 248, 241, 0.96));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: var(--card-shadow);
            min-height: 118px;
        }

        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] label,
        [data-testid="stMetricLabel"] div {
            color: var(--muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem !important;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
            font-size: 2.15rem !important;
            line-height: 1.05;
        }

        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid var(--line);
            border-radius: 999px;
            display: inline-flex;
            gap: 0.4rem;
            padding: 0.35rem;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.05);
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        .stTabs [data-baseweb="tab"] {
            height: auto;
            background: transparent;
            border-radius: 999px;
            color: var(--muted);
            font-weight: 600;
            padding: 0.55rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0f172a, #0f766e) !important;
            color: #f8fafc !important;
            box-shadow: 0 12px 30px rgba(15, 118, 110, 0.22);
        }

        details[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.05);
            overflow: hidden;
        }

        [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid var(--line);
            border-radius: 20px;
            color: var(--ink);
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span,
        [data-testid="stAlert"] div {
            color: var(--ink) !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.2rem;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.05);
        }

        [data-baseweb="select"] > div,
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            background: rgba(255, 255, 255, 0.82) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
        }

        [data-baseweb="select"] * {
            color: var(--ink) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.7);
            border: 1px dashed rgba(15, 118, 110, 0.3);
            border-radius: 18px;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 16px;
            min-height: 48px;
            font-weight: 600;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.8);
            color: var(--ink);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
        }

        .stButton > button[data-testid="baseButton-primary"],
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f172a, #0f766e) !important;
            color: #f8fafc !important;
            border: none !important;
        }

        .stPlotlyChart,
        .stAltairChart {
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 0.55rem 0.7rem 0.35rem 0.7rem;
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.05);
        }

        @media (max-width: 960px) {
            .hero-shell {
                padding: 1.45rem 1.25rem;
                border-radius: 24px;
            }

            [data-testid="stMetric"] {
                min-height: 106px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    """Render the top hero band."""
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Hybrid ML Recommender</div>
            <div class="hero-title">{APP_TITLE}</div>
            <p class="hero-copy">{APP_SUBTITLE}</p>
            <div class="pill-row">
                <span class="pill">Live benchmark leaderboard</span>
                <span class="pill">Meta-model top 3 ranking</span>
                <span class="pill">Data quality diagnostics</span>
                <span class="pill">Export-ready report</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_altair_chart(chart: Any) -> Any:
    """Apply a shared premium theme to Altair charts."""
    return (
        chart.properties(padding={"left": 8, "right": 12, "top": 6, "bottom": 8})
        .configure(background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#475467",
            titleColor="#101828",
            domainColor="rgba(15, 23, 42, 0.16)",
            tickColor="rgba(15, 23, 42, 0.16)",
            gridColor="rgba(148, 163, 184, 0.22)",
            labelFont="Aptos",
            titleFont="Aptos",
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_legend(
            labelColor="#475467",
            titleColor="#101828",
            labelFont="Aptos",
            titleFont="Aptos",
        )
        .configure_header(
            labelColor="#101828",
            titleColor="#101828",
            labelFont="Aptos",
            titleFont="Aptos",
        )
        .configure_title(color="#101828", font="Aptos")
    )


def _show_altair_chart(chart: Any) -> None:
    """Render an Altair chart with the shared visual theme."""
    st.altair_chart(_style_altair_chart(chart), use_container_width=True, theme=None)


@st.cache_data(show_spinner=False)
def load_uploaded_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """Load an uploaded CSV into a DataFrame."""
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_demo_dataframe() -> pd.DataFrame:
    """Load the bundled demo dataset."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df


@st.cache_resource(show_spinner=False)
def load_meta_predictor() -> MetaModelPredictor | None:
    """Load the persisted meta-model artifacts if available."""
    try:
        return MetaModelPredictor.load()
    except Exception:  # noqa: BLE001
        return None


def run_eda(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Run comprehensive EDA and generate recommendations."""
    logger.info("Running EDA on dataset with %d rows, %d columns", len(df), len(df.columns))
    
    eda_results = {
        "missing_values": {},
        "correlations": {},
        "outliers": {},
        "distributions": {},
        "recommendations": {}
    }
    
    # Missing values analysis
    missing_pct = (df.isnull().sum() / len(df)) * 100
    eda_results["missing_values"] = missing_pct.to_dict()
    
    recommendations = []
    for col, pct in missing_pct.items():
        if pct > 40:
            recommendations.append(f"Drop column '{col}' ({pct:.1f}% missing)")
        elif pct > 0:
            recommendations.append(f"Impute '{col}' ({pct:.1f}% missing)")
    
    # Correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        eda_results["correlations"] = {"high_corr_pairs": high_corr}
        for col1, col2, corr in high_corr:
            recommendations.append(f"Consider dropping one of highly correlated features: '{col1}' and '{col2}' (corr={corr:.2f})")
    
    # Outlier detection using IQR
    outlier_info = {}
    for col in numeric_df.columns:
        if col != target_column:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_pct = (outliers / len(numeric_df)) * 100
            outlier_info[col] = {"count": outliers, "percentage": outlier_pct}
            if outlier_pct > 5:
                recommendations.append(f"Consider removing outliers in '{col}' ({outlier_pct:.1f}% detected)")
    
    eda_results["outliers"] = outlier_info
    
    # Feature importance (quick RF)
    if len(numeric_df.columns) > 1 and target_column in df.columns:
        from sklearn.ensemble import RandomForestClassifier
        X = numeric_df.drop(columns=[target_column], errors='ignore')
        y = df[target_column]
        if len(X.columns) > 0:
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            importance = dict(zip(X.columns, rf.feature_importances_))
            eda_results["feature_importance"] = importance
    
    eda_results["recommendations"] = recommendations
    return eda_results


def preprocess_data(df: pd.DataFrame, target_column: str, options: dict) -> tuple[pd.DataFrame, dict]:
    """Apply preprocessing based on options."""
    logger.info("Preprocessing data with options: %s", options)
    
    processed_df = df.copy()
    preprocessing_log = []
    
    # Handle missing values
    if options.get("impute_missing", True):
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            imputer_num = SimpleImputer(strategy='mean')
            processed_df[numeric_cols] = imputer_num.fit_transform(processed_df[numeric_cols])
            preprocessing_log.append("Imputed missing values in numeric columns with mean")
        
        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            processed_df[categorical_cols] = imputer_cat.fit_transform(processed_df[categorical_cols])
            preprocessing_log.append("Imputed missing values in categorical columns with mode")
    
    # Remove outliers
    if options.get("remove_outliers", False):
        numeric_df = processed_df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if col != target_column:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (numeric_df[col] >= (Q1 - 1.5 * IQR)) & (numeric_df[col] <= (Q3 + 1.5 * IQR))
                processed_df = processed_df[mask]
        preprocessing_log.append("Removed outliers using IQR method")
    
    # Scaling
    scaler_type = options.get("scaler", "none")
    if scaler_type != "none":
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if scaler_type == "standard":
            scaler = StandardScaler()
            preprocessing_log.append("Applied StandardScaler to numeric features")
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
            preprocessing_log.append("Applied MinMaxScaler to numeric features")
        
        if numeric_cols:
            processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
    
    # Encoding
    if options.get("encode_categorical", True):
        categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(sparse=False, drop='first')
            encoded = encoder.fit_transform(processed_df[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
            processed_df = processed_df.drop(columns=categorical_cols).reset_index(drop=True)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
            preprocessing_log.append("Applied one-hot encoding to categorical features")
    
    return processed_df, {"log": preprocessing_log}


def get_candidate_models(problem_type: str) -> dict[str, Any]:
    """Get expanded list of candidate models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42) if problem_type == "classification" else DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42) if problem_type == "classification" else RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42) if problem_type == "classification" else GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42) if problem_type == "classification" else SVR(),
        "KNN": KNeighborsClassifier() if problem_type == "classification" else KNeighborsRegressor(),
        "Naive Bayes": GaussianNB() if problem_type == "classification" else None,
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42) if problem_type == "classification" else XGBRegressor(n_estimators=100, random_state=42)
    
    return {k: v for k, v in models.items() if v is not None}


def train_models(df: pd.DataFrame, target_column: str, test_size: float, random_state: int, selected_models: list[str]) -> dict[str, Any]:
    """Train and evaluate selected models."""
    logger.info("Training models: %s", selected_models)
    
    if df.empty or target_column not in df.columns:
        raise ValueError("Invalid dataset or target column")
    
    working_df = df.copy()
    y = working_df.pop(target_column)
    problem_type = detect_task_type(y)
    
    valid_index = y.notna()
    working_df = working_df.loc[valid_index].reset_index(drop=True)
    y = y.loc[valid_index].reset_index(drop=True)
    
    if len(working_df) < 5:
        raise ValueError("Dataset must have at least 5 rows")
    
    X_train, X_test, y_train, y_test = train_test_split(working_df, y, test_size=test_size, random_state=random_state, stratify=y if problem_type == "classification" else None)
    
    results = {}
    for model_name in selected_models:
        try:
            model = get_candidate_models(problem_type)[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if problem_type == "classification":
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                results[model_name] = {"accuracy": acc, "f1_score": f1, "model": model}
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[model_name] = {"mse": mse, "r2_score": r2, "model": model}
        except Exception as e:
            logger.error("Error training %s: %s", model_name, str(e))
            results[model_name] = {"error": str(e)}
    
    return {"results": results, "problem_type": problem_type, "X_test": X_test, "y_test": y_test}


def generate_recommendations(results: dict, eda_results: dict) -> dict[str, Any]:
    """Generate model recommendations based on results."""
    if not results["results"]:
        return {"best_model": None, "reasoning": "No models trained successfully"}
    
    problem_type = results["problem_type"]
    metric = "f1_score" if problem_type == "classification" else "r2_score"
    
    valid_results = {k: v for k, v in results["results"].items() if "error" not in v}
    if not valid_results:
        return {"best_model": None, "reasoning": "All models failed to train"}
    
    best_model = max(valid_results, key=lambda x: valid_results[x][metric])
    score = valid_results[best_model][metric]
    
    reasoning = f"Best model is {best_model} with {metric} of {score:.3f}"
    if eda_results.get("correlations", {}).get("high_corr_pairs"):
        reasoning += ". Dataset has correlated features, tree-based models may perform well."
    if any(pct > 5 for pct in eda_results.get("outliers", {}).values() if isinstance(pct, dict) and "percentage" in pct):
        reasoning += ". Outliers detected, consider robust models."
    
    return {"best_model": best_model, "reasoning": reasoning, "score": score}


def export_results(results: dict, eda_results: dict, df: pd.DataFrame, best_model: str) -> None:
    """Provide download options for results."""
    st.subheader("Export Results")
    
    # EDA Report
    eda_text = f"EDA Report\n\nMissing Values:\n{json.dumps(eda_results['missing_values'], indent=2)}\n\nRecommendations:\n" + "\n".join(eda_results['recommendations'])
    st.download_button("Download EDA Report", eda_text, "eda_report.txt")
    
    # Model Results
    results_df = pd.DataFrame.from_dict({k: v for k, v in results["results"].items() if "error" not in v}, orient='index')
    csv = results_df.to_csv()
    st.download_button("Download Model Results", csv, "model_results.csv")
    
    # Best Model
    if best_model and "model" in results["results"][best_model]:
        import pickle
        model_bytes = pickle.dumps(results["results"][best_model]["model"])
        st.download_button("Download Best Model", model_bytes, "best_model.pkl")


# Removed old render_app
    pass
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Dataset Setup")
        source = st.radio("Dataset source", ["Upload CSV", "Demo dataset"])
        
        df = None
        dataset_name = ""
        if source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name
        else:
            df = load_demo_dataframe()
            dataset_name = DEMO_DATASET_NAME
        
        target_column = ""
        if df is not None and not df.empty:
            target_column = st.selectbox("Target column", df.columns.tolist(), index=df.columns.tolist().index(guess_target_column(df)) if guess_target_column(df) in df.columns else 0)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            test_size = st.slider("Test size", 0.1, 0.5, DEFAULT_TEST_SIZE)
            random_state = st.number_input("Random seed", value=DEFAULT_RANDOM_STATE)
            
            st.subheader("Preprocessing")
            impute_missing = st.checkbox("Impute missing values", True)
            remove_outliers = st.checkbox("Remove outliers", False)
            scaler = st.selectbox("Scaling", ["none", "standard", "minmax"])
            encode_categorical = st.checkbox("Encode categorical", True)
            
            st.subheader("Model Selection")
            problem_type = detect_task_type(df[target_column]) if df is not None and target_column else "classification"
            all_models = list(get_candidate_models(problem_type).keys())
            selected_models = st.multiselect("Models to train", all_models, default=all_models[:5])
        
        run_analysis = st.button("Run Analysis", type="primary")
    
    if df is not None and run_analysis:
        with st.spinner("Running analysis..."):
            try:
                # EDA
                eda_results = run_eda(df, target_column)
                
                # Preprocessing
                preprocess_options = {
                    "impute_missing": impute_missing,
                    "remove_outliers": remove_outliers,
                    "scaler": scaler,
                    "encode_categorical": encode_categorical
                }
                processed_df, preprocess_log = preprocess_data(df, target_column, preprocess_options)
                
                # Train models
                results = train_models(processed_df, target_column, test_size, random_state, selected_models)
                
                # Recommendations
                recommendations = generate_recommendations(results, eda_results)
                
                # Display results
                st.header("Dataset Insights")
                st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                st.write(f"Problem type: {results['problem_type']}")
                
                st.header("EDA Summary")
                st.subheader("Missing Values")
                st.json(eda_results["missing_values"])
                st.subheader("Recommendations")
                for rec in eda_results["recommendations"]:
                    st.write(f"- {rec}")
                
                st.header("Preprocessing Applied")
                for log in preprocess_log["log"]:
                    st.write(f"- {log}")
                
                st.header("Model Leaderboard")
                results_df = pd.DataFrame.from_dict({k: v for k, v in results["results"].items() if "error" not in v}, orient='index')
                st.dataframe(results_df)
                
                if recommendations["best_model"]:
                    st.header("Best Model")
                    st.write(f"**{recommendations['best_model']}**")
                    st.write(recommendations["reasoning"])
                
                # Meta-model insights
                meta_predictor = load_meta_predictor()
                if meta_predictor:
                    st.header("Meta-Model Insights")
                    try:
                        meta_result = recommend_for_dataframe(processed_df, meta_predictor, target_column=target_column)
                        st.write(f"Meta-model recommends: {meta_result.get('best_model', 'N/A')}")
                        if "meta_model_metrics" in meta_result:
                            st.json(meta_result["meta_model_metrics"])
                    except Exception as e:
                        st.warning(f"Meta-model failed: {str(e)}")
                else:
                    st.info("Meta-model artifacts not available. Train a meta-model first.")
                
                # Export
                export_results(results, eda_results, processed_df, recommendations.get("best_model"))
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.error("Analysis error", exc_info=True)


if __name__ == "__main__":
    render_app()


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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline for mixed tabular data."""
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
    """Attach preprocessing to a model clone."""
    return Pipeline([("preprocessor", build_preprocessor(X)), ("model", clone(model))])


def _align_label_types(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Make classification labels comparable for metrics and plots."""
    true_series = pd.Series(y_true).reset_index(drop=True)
    pred_series = pd.Series(y_pred).reset_index(drop=True)

    if getattr(true_series, "dtype", None) != getattr(pred_series, "dtype", None):
        try:
            pred_series = pred_series.astype(true_series.dtype)
        except (TypeError, ValueError):
            true_series = true_series.astype(str)
            pred_series = pred_series.astype(str)
    return true_series, pred_series


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
        y_train_encoded = label_encoder.fit_transform(y_train)
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
        y_true_aligned, y_pred_aligned = _align_label_types(y_true, y_pred)
        accuracy = accuracy_score(y_true_aligned, y_pred_aligned)
        f1 = f1_score(y_true_aligned, y_pred_aligned, average="weighted", zero_division=0)
        return {"primary_score": float(f1), "accuracy": float(accuracy), "f1_score": float(f1)}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    primary = 1.0 / (1.0 + rmse)
    return {"primary_score": primary, "rmse": rmse, "r2_score": r2}


def _split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset with a fallback when stratification is too tight."""
    stratify = None
    if problem_type == "classification" and y.nunique(dropna=True) > 1:
        class_counts = y.value_counts(dropna=True)
        if not class_counts.empty and class_counts.min() >= 2:
            stratify = y

    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


@st.cache_data(show_spinner=False)
def benchmark_models(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Benchmark lightweight models on the uploaded dataset."""
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    working_df = df.copy()
    y = working_df.pop(target_column)
    problem_type = detect_task_type(y)

    valid_index = y.notna()
    working_df = working_df.loc[valid_index].reset_index(drop=True)
    y = y.loc[valid_index].reset_index(drop=True)

    if working_df.empty or y.empty:
        raise ValueError("No valid rows remain after removing missing target values.")
    if working_df.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the target column.")
    if len(working_df) < 5:
        raise ValueError("Dataset must have at least 5 rows for stable recommendation.")
    if problem_type == "classification" and y.nunique(dropna=True) < 2:
        raise ValueError("Classification requires at least two target classes.")

    X_train, X_test, y_train, y_test = _split_dataset(
        working_df,
        y,
        problem_type=problem_type,
        test_size=test_size,
        random_state=random_state,
    )

    leaderboard: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, pd.Series] = {}
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
            if problem_type == "classification":
                _, aligned_pred = _align_label_types(y_test, y_pred)
                y_pred_series = aligned_pred
            else:
                y_pred_series = pd.Series(y_pred).reset_index(drop=True)

            metrics = _score_predictions(problem_type, y_test.reset_index(drop=True), y_pred_series.to_numpy())
            leaderboard.append({"model": model_name, **metrics})
            fitted_models[model_name] = pipeline
            predictions[model_name] = y_pred_series.reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{model_name}: {exc}")

    if not leaderboard:
        raise RuntimeError("All candidate models failed to train on this dataset.")

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("primary_score", ascending=False).reset_index(drop=True)
    confidence_denominator = leaderboard_df["primary_score"].clip(lower=0).sum()
    if confidence_denominator > 0:
        leaderboard_df["confidence"] = leaderboard_df["primary_score"].clip(lower=0) / confidence_denominator
    else:
        leaderboard_df["confidence"] = 0.0
    leaderboard_df["confidence"] = leaderboard_df["confidence"].fillna(0.0).round(4)

    best_model_name = str(leaderboard_df.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]
    best_predictions = predictions[best_model_name]
    best_estimator = best_pipeline.named_steps["model"]
    transformed_feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

    importance_df = pd.DataFrame(columns=["feature", "importance"])
    if hasattr(best_estimator, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "importance": best_estimator.feature_importances_,
            }
        ).sort_values("importance", ascending=False).head(15)
    elif hasattr(best_estimator, "coef_"):
        coefficients = np.abs(np.asarray(best_estimator.coef_))
        if coefficients.ndim > 1:
            coefficients = coefficients.mean(axis=0)
        coefficients = np.ravel(coefficients)
        importance_df = pd.DataFrame(
            {
                "feature": transformed_feature_names[: len(coefficients)],
                "importance": coefficients,
            }
        ).sort_values("importance", ascending=False).head(15)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr_df = numeric_df.corr(numeric_only=True).round(2) if not numeric_df.empty else pd.DataFrame()

    y_test_out = y_test.reset_index(drop=True)
    if problem_type == "classification":
        y_test_out, best_predictions = _align_label_types(y_test_out, best_predictions)

    prediction_sample = pd.DataFrame({"actual": y_test_out, "predicted": best_predictions}).head(50)
    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "leaderboard": leaderboard_df,
        "best_model": best_model_name,
        "best_confidence": float(leaderboard_df.iloc[0]["confidence"]),
        "best_metrics": leaderboard_df.iloc[0].to_dict(),
        "y_test": y_test_out,
        "best_predictions": best_predictions.reset_index(drop=True),
        "prediction_sample": prediction_sample.reset_index(drop=True),
        "importance_df": importance_df.reset_index(drop=True),
        "correlation_df": corr_df,
        "failures": failures,
        "summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values": int(df.isna().sum().sum()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
    }


def _summarize_dataset_health(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Build a user-facing health summary for the uploaded dataset."""
    rows, columns = df.shape
    missing_values = int(df.isna().sum().sum())
    total_cells = max(rows * max(columns, 1), 1)
    missing_ratio = float(missing_values / total_cells)
    duplicate_rows = int(df.duplicated().sum())
    numeric_columns = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    high_cardinality = [
        column
        for column in categorical_columns
        if df[column].nunique(dropna=True) > min(40, max(10, rows // 10))
    ]

    target_series = df[target_column] if target_column in df.columns else pd.Series(dtype="float64")
    target_missing = int(target_series.isna().sum()) if not target_series.empty else 0
    problem_type = detect_task_type(target_series) if not target_series.empty else "classification"
    target_unique = int(target_series.nunique(dropna=True)) if not target_series.empty else 0

    class_balance_ratio = None
    if problem_type == "classification" and not target_series.empty:
        counts = target_series.value_counts(dropna=True)
        if not counts.empty and counts.max() > 0:
            class_balance_ratio = float(counts.min() / counts.max())

    warnings: list[str] = []
    if rows < 50:
        warnings.append("Small datasets can make leaderboard scores noisy and unstable.")
    if missing_ratio >= 0.15:
        warnings.append("Missing values are high; models may be ranking the imputation strategy as much as the signal.")
    if duplicate_rows > 0:
        warnings.append("Duplicate rows were detected and may inflate evaluation quality.")
    if high_cardinality:
        warnings.append("High-cardinality categorical columns may create a wide encoded feature space.")
    if problem_type == "classification" and class_balance_ratio is not None and class_balance_ratio < 0.35:
        warnings.append("Class imbalance is significant, so accuracy alone may be misleading.")
    if target_missing > 0:
        warnings.append("Rows with missing target values will be dropped before benchmarking.")

    penalties = 0
    penalties += 20 if rows < 50 else 0
    penalties += 18 if missing_ratio >= 0.15 else 8 if missing_ratio >= 0.05 else 0
    penalties += 10 if duplicate_rows > 0 else 0
    penalties += 10 if high_cardinality else 0
    penalties += 12 if class_balance_ratio is not None and class_balance_ratio < 0.35 else 0
    penalties += 10 if target_missing > 0 else 0
    readiness_score = max(5, 100 - penalties)

    missing_df = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_ratio"})
    )
    missing_df = missing_df[missing_df["missing_ratio"] > 0].head(12)

    dtype_df = (
        df.dtypes.astype(str)
        .reset_index()
        .rename(columns={"index": "column", 0: "dtype"})
        .assign(non_null=lambda frame: [int(df[column].notna().sum()) for column in frame["column"]])
    )

    return {
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "missing_ratio": missing_ratio,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "memory_mb": memory_mb,
        "target_missing": target_missing,
        "target_unique": target_unique,
        "problem_type": problem_type,
        "class_balance_ratio": class_balance_ratio,
        "high_cardinality_columns": high_cardinality,
        "warnings": warnings,
        "readiness_score": readiness_score,
        "missing_df": missing_df,
        "dtype_df": dtype_df,
    }


def _target_distribution_chart(target: pd.Series, problem_type: str) -> alt.Chart | None:
    """Visualize the target distribution."""
    clean_target = target.dropna().reset_index(drop=True)
    if clean_target.empty:
        return None

    if problem_type == "classification":
        distribution = (
            clean_target.astype(str)
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count")
            .head(12)
        )
        return (
            alt.Chart(distribution)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("label:N", sort="-y", title="Class"),
                y=alt.Y("count:Q", title="Rows"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
                tooltip=["label", "count"],
            )
            .properties(height=300)
        )

    distribution = pd.DataFrame({"target": pd.to_numeric(clean_target, errors="coerce")}).dropna()
    if distribution.empty:
        return None
    return (
        alt.Chart(distribution)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("target:Q", bin=alt.Bin(maxbins=25), title="Target value"),
            y=alt.Y("count():Q", title="Rows"),
            color=alt.value("#0f766e"),
        )
        .properties(height=300)
    )


def _recommendation_chart(top_models: list[dict[str, Any]]) -> alt.Chart | None:
    """Visualize the meta-model recommendation ranking."""
    if not top_models:
        return None
    chart_df = pd.DataFrame(top_models)
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("model:N", sort="-y", title="Model"),
            y=alt.Y("probability:Q", title="Probability", axis=alt.Axis(format=".0%")),
            color=alt.Color("probability:Q", scale=alt.Scale(scheme="goldgreen"), legend=None),
            tooltip=["model", alt.Tooltip("probability:Q", format=".2%")],
        )
        .properties(height=300)
    )


def _leaderboard_chart(leaderboard_df: pd.DataFrame, problem_type: str) -> alt.Chart | None:
    """Visualize the live benchmark leaderboard."""
    if leaderboard_df.empty:
        return None
    value_field = "f1_score" if problem_type == "classification" else "r2_score"
    if value_field not in leaderboard_df.columns:
        value_field = "primary_score"
    return (
        alt.Chart(leaderboard_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("model:N", sort="-y", title="Model"),
            y=alt.Y(f"{value_field}:Q", title=value_field.replace("_", " ").title()),
            color=alt.Color(f"{value_field}:Q", scale=alt.Scale(scheme="teals"), legend=None),
            tooltip=["model", alt.Tooltip(f"{value_field}:Q", format=".4f")],
        )
        .properties(height=300)
    )


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
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color("importance:Q", scale=alt.Scale(scheme="greens"), legend=None),
            tooltip=["feature", "importance"],
        )
        .properties(height=360)
    )


def _missingness_chart(missing_df: pd.DataFrame) -> alt.Chart | None:
    """Visualize per-column missingness."""
    if missing_df.empty:
        return None
    return (
        alt.Chart(missing_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("column:N", sort="-y", title="Column"),
            y=alt.Y("missing_ratio:Q", axis=alt.Axis(format=".0%"), title="Missing ratio"),
            color=alt.Color("missing_ratio:Q", scale=alt.Scale(scheme="oranges"), legend=None),
            tooltip=["column", alt.Tooltip("missing_ratio:Q", format=".2%")],
        )
        .properties(height=300)
    )


def _confusion_matrix_chart(y_true: pd.Series, y_pred: pd.Series) -> alt.Chart:
    """Build a confusion matrix heatmap."""
    true_series, pred_series = _align_label_types(y_true, y_pred)
    display_true = true_series.astype(str)
    display_pred = pred_series.astype(str)
    labels = sorted(pd.unique(pd.concat([display_true, display_pred], ignore_index=True)))
    matrix = confusion_matrix(display_true, display_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    melted = matrix_df.reset_index(names="actual").melt("actual", var_name="predicted", value_name="count")
    return (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", title="Predicted"),
            y=alt.Y("actual:N", title="Actual"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="teals")),
            tooltip=["actual", "predicted", "count"],
        )
        .properties(height=340)
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
        .mark_circle(size=72, opacity=0.74, color="#0f766e")
        .encode(x=alt.X("actual:Q", title="Actual"), y=alt.Y("predicted:Q", title="Predicted"))
    )
    line = alt.Chart(reference_df).mark_line(color="#b45309", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
    return (scatter + line).properties(height=340)


def _meta_feature_frame(meta_features: dict[str, float]) -> pd.DataFrame:
    """Format meta-features for display."""
    feature_df = pd.DataFrame(
        [{"feature": feature.replace("_", " ").title(), "value": float(value)} for feature, value in meta_features.items()]
    )
    return feature_df.sort_values("value", ascending=False).reset_index(drop=True)


def _build_analysis_brief(
    dataset_health: dict[str, Any],
    benchmark_result: dict[str, Any] | None,
    meta_result: dict[str, Any] | None,
) -> list[str]:
    """Create short, high-signal observations for the command center."""
    points = [
        (
            f"Dataset readiness is {dataset_health['readiness_score']}/100 with "
            f"{dataset_health['missing_ratio']:.1%} missingness and {dataset_health['duplicate_rows']} duplicate rows."
        )
    ]
    if meta_result:
        top_entry = meta_result["top_3"][0]
        points.append(
            f"The meta-model favors {top_entry['model']} first with {top_entry['probability']:.1%} confidence."
        )
    if benchmark_result:
        best_model = benchmark_result["best_model"]
        score_label = "F1" if benchmark_result["problem_type"] == "classification" else "R2"
        metric_field = "f1_score" if benchmark_result["problem_type"] == "classification" else "r2_score"
        score_value = benchmark_result["best_metrics"].get(metric_field, benchmark_result["best_metrics"]["primary_score"])
        points.append(f"On the live holdout split, {best_model} leads with {score_label} {float(score_value):.3f}.")
    if meta_result and benchmark_result:
        meta_best = meta_result["best_model"]
        live_best = benchmark_result["best_model"]
        if meta_best == live_best:
            points.append("The trained meta-model and the live benchmark agree on the strongest candidate.")
        else:
            points.append(
                f"The meta-model suggests {meta_best}, while the holdout benchmark currently prefers {live_best}; "
                "that disagreement is useful signal when validating deployment choices."
            )
    return points


def _render_command_center_visuals(
    df: pd.DataFrame,
    target_column: str,
    dataset_health: dict[str, Any],
    benchmark_result: dict[str, Any] | None,
    meta_result: dict[str, Any] | None,
) -> None:
    """Render high-signal visuals on the first report tab."""
    st.markdown("### Visual Snapshot")
    st.markdown(
        '<p class="section-copy">A fast read on target behavior and model preference before you move into the detailed tabs.</p>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        if meta_result:
            st.subheader("Meta-model signal")
            recommendation_chart = _recommendation_chart(meta_result["top_3"])
            if recommendation_chart is not None:
                _show_altair_chart(recommendation_chart)
            else:
                st.info("The meta-model did not return a chartable ranking for this dataset.")
        elif benchmark_result:
            st.subheader("Benchmark signal")
            leaderboard_chart = _leaderboard_chart(benchmark_result["leaderboard"], benchmark_result["problem_type"])
            if leaderboard_chart is not None:
                _show_altair_chart(leaderboard_chart)
            else:
                st.info("The live benchmark did not return a chartable leaderboard.")
        else:
            st.info("Run either the meta-model or the live benchmark to populate this visual.")

    with right_col:
        st.subheader("Target profile")
        target_chart = _target_distribution_chart(df[target_column], dataset_health["problem_type"])
        if target_chart is not None:
            _show_altair_chart(target_chart)
        else:
            missing_chart = _missingness_chart(dataset_health["missing_df"])
            if missing_chart is not None:
                _show_altair_chart(missing_chart)
            else:
                st.info("A chart preview is not available for the current target column.")


def _build_export_payload(
    dataset_name: str,
    target_column: str,
    dataset_health: dict[str, Any],
    benchmark_result: dict[str, Any] | None,
    meta_result: dict[str, Any] | None,
    analysis_mode: str,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """Create a JSON-safe export structure for the current report."""
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_name": dataset_name,
        "target_column": target_column,
        "analysis_mode": analysis_mode,
        "test_size": test_size,
        "random_state": random_state,
        "dataset_health": {
            "rows": dataset_health["rows"],
            "columns": dataset_health["columns"],
            "missing_values": dataset_health["missing_values"],
            "missing_ratio": dataset_health["missing_ratio"],
            "duplicate_rows": dataset_health["duplicate_rows"],
            "numeric_columns": dataset_health["numeric_columns"],
            "categorical_columns": dataset_health["categorical_columns"],
            "memory_mb": dataset_health["memory_mb"],
            "target_missing": dataset_health["target_missing"],
            "target_unique": dataset_health["target_unique"],
            "problem_type": dataset_health["problem_type"],
            "readiness_score": dataset_health["readiness_score"],
            "warnings": dataset_health["warnings"],
            "high_cardinality_columns": dataset_health["high_cardinality_columns"],
            "class_balance_ratio": dataset_health["class_balance_ratio"],
        },
    }

    if meta_result:
        payload["meta_model"] = {
            "best_model": meta_result["best_model"],
            "top_3": meta_result["top_3"],
            "dataset_summary": meta_result["dataset_summary"],
            "problem_type": meta_result["problem_type"],
            "meta_features": meta_result["meta_features"],
        }

    if benchmark_result:
        payload["benchmark"] = {
            "best_model": benchmark_result["best_model"],
            "best_metrics": benchmark_result["best_metrics"],
            "problem_type": benchmark_result["problem_type"],
            "summary": benchmark_result["summary"],
            "leaderboard": benchmark_result["leaderboard"].to_dict(orient="records"),
            "failures": benchmark_result["failures"],
            "prediction_sample": benchmark_result["prediction_sample"].head(20).to_dict(orient="records"),
        }
    return payload


def _analysis_signature(
    dataset_name: str,
    df: pd.DataFrame,
    target_column: str,
    analysis_mode: str,
    test_size: float,
    random_state: int,
) -> str:
    """Generate a lightweight signature so stale results can be detected."""
    raw_signature = {
        "dataset_name": dataset_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_column,
        "analysis_mode": analysis_mode,
        "test_size": round(float(test_size), 3),
        "random_state": int(random_state),
        "missing_values": int(df.isna().sum().sum()),
    }
    return json.dumps(raw_signature, sort_keys=True)


def _load_current_dataset(source: str, uploaded_file: Any) -> tuple[pd.DataFrame | None, str]:
    """Resolve the active dataset source."""
    if source == "Demo dataset":
        return load_demo_dataframe(), DEMO_DATASET_NAME
    if uploaded_file is None:
        return None, ""
    return load_uploaded_dataframe(uploaded_file.getvalue()), uploaded_file.name


def render_app() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    _inject_styles()
    _render_hero()

    st.session_state.setdefault("analysis_payload", None)

    with st.sidebar:
        st.header("Analysis Setup")
        source = st.radio("Dataset source", options=["Upload CSV", "Demo dataset"])
        uploaded_file = None
        if source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
            st.caption("Streamlit uploads are usually capped at 200 MB by default unless app config changes it.")
        else:
            st.caption(f"Using bundled sample dataset: `{DEMO_DATASET_NAME}`")

        df, dataset_name = _load_current_dataset(source, uploaded_file)

        target_column = ""
        if df is not None:
            guessed_target = guess_target_column(df)
            target_index = df.columns.get_loc(guessed_target)
            target_column = st.selectbox("Target column", options=df.columns.tolist(), index=target_index)

        analysis_mode = st.radio(
            "Analysis mode",
            options=["Hybrid", "Benchmark only", "Meta only"],
            help="Hybrid runs both the live benchmark and the trained meta-model.",
        )
        test_size = st.slider("Holdout ratio", min_value=0.15, max_value=0.40, value=DEFAULT_TEST_SIZE, step=0.05)
        random_state = st.number_input("Random seed", min_value=1, max_value=9999, value=DEFAULT_RANDOM_STATE, step=1)
        show_meta_features = st.toggle("Show meta-feature panel", value=True)
        show_prediction_sample = st.toggle("Show prediction sample", value=True)

        run_analysis = st.button("Run analysis", type="primary", use_container_width=True, disabled=df is None)
        clear_report = st.button("Clear report", use_container_width=True)

        if clear_report:
            st.session_state["analysis_payload"] = None

    if df is None:
        st.markdown(
            """
            <div class="mini-card-grid">
                <div class="mini-card">
                    <h4>Start Fast</h4>
                    <p>Upload your own CSV or load the bundled iris demo to see the full workflow immediately.</p>
                </div>
                <div class="mini-card">
                    <h4>Compare Two Brains</h4>
                    <p>Use the trained meta-model for instant guidance, then validate it with a live benchmark leaderboard.</p>
                </div>
                <div class="mini-card">
                    <h4>Ship Better Decisions</h4>
                    <p>Review health checks, diagnostics, and export the final report instead of guessing from one score.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Choose a CSV in the sidebar or switch to the demo dataset to begin.")
        return

    dataset_health = _summarize_dataset_health(df, target_column)
    signature = _analysis_signature(
        dataset_name=dataset_name,
        df=df,
        target_column=target_column,
        analysis_mode=analysis_mode,
        test_size=test_size,
        random_state=int(random_state),
    )

    summary_cols = st.columns(5)
    summary_cols[0].metric("Rows", f"{dataset_health['rows']:,}")
    summary_cols[1].metric("Columns", dataset_health["columns"])
    summary_cols[2].metric("Problem Type", dataset_health["problem_type"].title())
    summary_cols[3].metric("Missing", f"{dataset_health['missing_ratio']:.1%}")
    summary_cols[4].metric("Readiness", f"{dataset_health['readiness_score']}/100")

    preview_col, status_col = st.columns([1.5, 1.0], gap="large")
    with preview_col:
        with st.expander("Dataset preview", expanded=False):
            st.dataframe(df.head(25), use_container_width=True)

    with status_col:
        warnings = dataset_health["warnings"]
        if warnings:
            st.warning("\n".join(f"- {warning}" for warning in warnings))
        else:
            st.success("No major health red flags were detected for the current target setup.")

    saved_payload = st.session_state.get("analysis_payload")
    if saved_payload and saved_payload.get("signature") != signature:
        st.info("Settings changed after the last run. Click `Run analysis` to refresh the report.")

    if run_analysis:
        meta_result: dict[str, Any] | None = None
        benchmark_result: dict[str, Any] | None = None
        meta_error = ""

        with st.spinner("Analyzing dataset, generating recommendations, and building diagnostics..."):
            if analysis_mode in {"Hybrid", "Meta only"}:
                predictor = load_meta_predictor()
                if predictor is None:
                    meta_error = "Meta-model artifacts are missing or could not be loaded."
                else:
                    meta_result = recommend_for_dataframe(df, predictor, target_column=target_column)
                    meta_result["model_metrics"] = meta_result.get("meta_model_metrics", predictor.metrics)

            if analysis_mode in {"Hybrid", "Benchmark only"}:
                benchmark_result = benchmark_models(
                    df,
                    target_column=target_column,
                    test_size=float(test_size),
                    random_state=int(random_state),
                )

        st.session_state["analysis_payload"] = {
            "signature": signature,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "analysis_mode": analysis_mode,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "dataset_health": dataset_health,
            "meta_result": meta_result,
            "benchmark_result": benchmark_result,
            "meta_error": meta_error,
            "show_meta_features": show_meta_features,
            "show_prediction_sample": show_prediction_sample,
        }

    payload = st.session_state.get("analysis_payload")
    if not payload or payload.get("signature") != signature:
        st.markdown(
            """
            <div class="callout">
                <strong>Ready when you are</strong>
                <span>
                    The dataset is loaded. Run the analysis to compare the trained recommender with a live holdout benchmark,
                    review health signals, and export a decision report.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    meta_result = payload.get("meta_result")
    benchmark_result = payload.get("benchmark_result")
    meta_error = payload.get("meta_error", "")
    analysis_brief = _build_analysis_brief(dataset_health, benchmark_result, meta_result)

    tabs = st.tabs(["Command Center", "Recommendations", "Data Quality", "Diagnostics", "Export"])

    with tabs[0]:
        st.markdown("### Operating Summary")
        st.markdown(
            '<p class="section-copy">A concise read on where the dataset stands and what the app currently prefers.</p>',
            unsafe_allow_html=True,
        )

        top_cols = st.columns(4)
        top_cols[0].metric("Target", payload["target_column"])
        top_cols[1].metric("Mode", payload["analysis_mode"])
        top_cols[2].metric("Duplicates", dataset_health["duplicate_rows"])
        top_cols[3].metric("Memory", f"{dataset_health['memory_mb']:.2f} MB")

        for point in analysis_brief:
            st.markdown(
                f"""
                <div class="callout" style="margin-top:0.7rem;">
                    <strong>Insight</strong>
                    <span>{point}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if meta_error:
            st.warning(meta_error)

        if meta_result and meta_result.get("model_metrics"):
            metric_cols = st.columns(4)
            metric_cols[0].metric("Meta-model Accuracy", f"{float(meta_result['model_metrics'].get('accuracy', 0.0)):.1%}")
            metric_cols[1].metric(
                "Meta-model Top-3 Accuracy",
                f"{float(meta_result['model_metrics'].get('top_3_accuracy', 0.0)):.1%}",
            )
            metric_cols[2].metric("Meta-model NDCG@3", f"{float(meta_result['model_metrics'].get('ndcg_at_3', 0.0)):.3f}")
            metric_cols[3].metric("Meta-model MAP@3", f"{float(meta_result['model_metrics'].get('map_at_3', 0.0)):.3f}")

        _render_command_center_visuals(df, target_column, dataset_health, benchmark_result, meta_result)

    with tabs[1]:
        st.markdown("### Recommendation Stack")
        st.markdown(
            '<p class="section-copy">Use the meta-model for fast prior guidance and the benchmark leaderboard for dataset-specific validation.</p>',
            unsafe_allow_html=True,
        )

        left_col, right_col = st.columns(2, gap="large")
        with left_col:
            st.subheader("Meta-model Ranking")
            if meta_result:
                recommendation_chart = _recommendation_chart(meta_result["top_3"])
                if recommendation_chart is not None:
                    _show_altair_chart(recommendation_chart)
                meta_table = pd.DataFrame(meta_result["top_3"])
                meta_table["probability"] = meta_table["probability"].map(lambda value: f"{value:.2%}")
                st.dataframe(meta_table, use_container_width=True, hide_index=True)
                if meta_result.get("recommendation_modes"):
                    with st.expander("Accurate vs fast recommendation modes", expanded=False):
                        mode_tables = st.columns(2)
                        accurate_table = pd.DataFrame(meta_result["recommendation_modes"].get("accurate", []))
                        fast_table = pd.DataFrame(meta_result["recommendation_modes"].get("fast", []))
                        if not accurate_table.empty and "probability" in accurate_table.columns:
                            accurate_table["probability"] = accurate_table["probability"].map(lambda value: f"{value:.2%}")
                        if not fast_table.empty and "probability" in fast_table.columns:
                            fast_table["probability"] = fast_table["probability"].map(lambda value: f"{value:.2%}")
                        mode_tables[0].caption("Accurate mode")
                        mode_tables[0].dataframe(accurate_table, use_container_width=True, hide_index=True)
                        mode_tables[1].caption("Fast mode")
                        mode_tables[1].dataframe(fast_table, use_container_width=True, hide_index=True)
                if meta_result.get("explanation"):
                    with st.expander("Why this model was recommended", expanded=False):
                        explanation = meta_result["explanation"]
                        top_features = pd.DataFrame(explanation.get("top_features", []))
                        if not top_features.empty:
                            st.dataframe(top_features, use_container_width=True, hide_index=True)
            else:
                st.info("Meta-model recommendations are not available for this run.")

        with right_col:
            st.subheader("Live Benchmark Leaderboard")
            if benchmark_result:
                leaderboard_chart = _leaderboard_chart(benchmark_result["leaderboard"], benchmark_result["problem_type"])
                if leaderboard_chart is not None:
                    _show_altair_chart(leaderboard_chart)
                display_columns = (
                    ["model", "confidence", "accuracy", "f1_score"]
                    if benchmark_result["problem_type"] == "classification"
                    else ["model", "confidence", "rmse", "r2_score"]
                )
                leaderboard = benchmark_result["leaderboard"][
                    [column for column in display_columns if column in benchmark_result["leaderboard"].columns]
                ].copy()
                if "confidence" in leaderboard.columns:
                    leaderboard["confidence"] = leaderboard["confidence"].map(lambda value: f"{value:.2%}")
                st.dataframe(leaderboard, use_container_width=True, hide_index=True)
            else:
                st.info("Live benchmarking was skipped for this run.")

        if meta_result and benchmark_result:
            if meta_result["best_model"] == benchmark_result["best_model"]:
                st.success(
                    f"Both engines agree on `{benchmark_result['best_model']}`, which is a strong sign for the current dataset."
                )
            else:
                st.info(
                    f"Meta-model choice: `{meta_result['best_model']}`. Live benchmark winner: `{benchmark_result['best_model']}`."
                )

        if benchmark_result and benchmark_result["failures"]:
            st.warning("A few benchmark models failed, but the report is still usable.")
            st.write(benchmark_result["failures"])

    with tabs[2]:
        st.markdown("### Data Quality")
        st.markdown(
            '<p class="section-copy">A quick audit of target stability, missingness, and schema shape before you trust the rankings.</p>',
            unsafe_allow_html=True,
        )

        health_cols = st.columns(4)
        health_cols[0].metric("Missing values", dataset_health["missing_values"])
        health_cols[1].metric("Target classes/levels", dataset_health["target_unique"])
        health_cols[2].metric("Numeric columns", dataset_health["numeric_columns"])
        health_cols[3].metric("Categorical columns", dataset_health["categorical_columns"])

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.subheader("Target distribution")
            target_chart = _target_distribution_chart(df[target_column], dataset_health["problem_type"])
            if target_chart is None:
                st.info("Target distribution could not be visualized.")
            else:
                _show_altair_chart(target_chart)

        with chart_right:
            st.subheader("Column missingness")
            missing_chart = _missingness_chart(dataset_health["missing_df"])
            if missing_chart is None:
                st.success("No missing values were detected by column.")
            else:
                _show_altair_chart(missing_chart)

        if dataset_health["high_cardinality_columns"]:
            st.info("High-cardinality columns: " + ", ".join(dataset_health["high_cardinality_columns"][:8]))

        with st.expander("Column types and coverage", expanded=False):
            st.dataframe(dataset_health["dtype_df"], use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("### Diagnostics")
        st.markdown(
            '<p class="section-copy">Detailed visuals that help explain why the recommendation looks the way it does.</p>',
            unsafe_allow_html=True,
        )

        diag_left, diag_right = st.columns(2, gap="large")
        with diag_left:
            st.subheader("Feature importance")
            if benchmark_result:
                importance_chart = _feature_importance_chart(benchmark_result["importance_df"])
                if importance_chart is None:
                    st.info("Feature importance is not available for the current best benchmark model.")
                else:
                    _show_altair_chart(importance_chart)
            else:
                st.info("Run the live benchmark to unlock feature importance diagnostics.")

        with diag_right:
            st.subheader("Correlation heatmap")
            if benchmark_result:
                heatmap = _heatmap_from_correlation(benchmark_result["correlation_df"])
                if heatmap is None:
                    st.info("A correlation heatmap needs at least two numeric columns.")
                else:
                    _show_altair_chart(heatmap)
            else:
                st.info("Run the live benchmark to unlock correlation diagnostics.")

        if benchmark_result:
            st.subheader("Evaluation view")
            if benchmark_result["problem_type"] == "classification":
                _show_altair_chart(_confusion_matrix_chart(benchmark_result["y_test"], benchmark_result["best_predictions"]))
            else:
                _show_altair_chart(_regression_plot(benchmark_result["y_test"], benchmark_result["best_predictions"]))

            if payload.get("show_prediction_sample", True):
                st.subheader("Prediction sample")
                st.dataframe(benchmark_result["prediction_sample"], use_container_width=True, hide_index=True)

        if payload.get("show_meta_features", True) and meta_result:
            st.subheader("Meta-feature profile")
            st.dataframe(_meta_feature_frame(meta_result["meta_features"]), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.markdown("### Export")
        st.markdown(
            '<p class="section-copy">Carry the recommendation forward as a JSON report or a benchmark CSV snapshot.</p>',
            unsafe_allow_html=True,
        )

        export_payload = _build_export_payload(
            dataset_name=payload["dataset_name"],
            target_column=payload["target_column"],
            dataset_health=dataset_health,
            benchmark_result=benchmark_result,
            meta_result=meta_result,
            analysis_mode=payload["analysis_mode"],
            test_size=payload["test_size"],
            random_state=payload["random_state"],
        )

        export_cols = st.columns(2)
        export_cols[0].download_button(
            "Download JSON report",
            data=json.dumps(export_payload, indent=2),
            file_name=f"{Path(payload['dataset_name']).stem}_analysis_report.json",
            mime="application/json",
            use_container_width=True,
        )

        leaderboard_csv = benchmark_result["leaderboard"].to_csv(index=False) if benchmark_result else "model\n"
        export_cols[1].download_button(
            "Download leaderboard CSV",
            data=leaderboard_csv,
            file_name=f"{Path(payload['dataset_name']).stem}_leaderboard.csv",
            mime="text/csv",
            disabled=benchmark_result is None,
            use_container_width=True,
        )

        st.code(
            f"python main.py --file {payload['dataset_name']} --target {payload['target_column']}",
            language="bash",
        )
        st.caption("The CLI command above mirrors the target selection for batch runs outside the UI.")


if __name__ == "__main__":
    render_app()
