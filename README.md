# Meta-Learning Model Recommender

Production-grade, scalable meta-learning system that learns from many tabular datasets and recommends the best ML algorithm.

## Architecture

```text
OpenML Datasets
   │
   ├── data_loader.py (validation, cleaning, sampling)
   │
   ├── features.py (meta-feature extraction)
   │
   ├── evaluator.py (base-model CV scoring + timing + cache)
   │
   ├── pipeline.py (parallel dataset processing + tqdm + meta-dataset CSV)
   │
   └── predictor.py (RandomForest meta-model + scaler + top-k inference)
                     │
                     ├── main.py (CLI)
                     └── streamlit_app.py (UI)
```

## Key Capabilities

- Robust OpenML ingestion with safe skipping of invalid datasets.
- Advanced meta-features (kurtosis, entropy approximation, PCA top-2 variance, sparsity, outlier ratio).
- Probabilistic top-3 model ranking via `predict_proba()`.
- Class-imbalance-aware meta-model (`class_weight="balanced"`).
- Meta-model evaluation output: accuracy, confusion matrix, classification report.
- Per-model training and inference timing + timeout guardrails.
- Parallel dataset processing and progress bar support.
- Streamlit UI for CSV upload, target selection, summary, and recommendation charts.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Train Meta-Model

```bash
meta-recommender --train --openml-limit 30
```

Outputs:
- `models/meta_model.joblib`
- `models/meta_scaler.joblib`
- `models/meta_dataset.csv`
- `logs.txt`

## CLI Inference

```bash
python main.py --file data.csv --target target_column
```

## Streamlit App

```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment

1. Push repository to GitHub.
2. In Streamlit Cloud, create a new app and set:
   - Main file: `streamlit_app.py`
   - Python dependencies: `requirements.txt`
3. Deploy and share link.

Deployment URL placeholder: `https://<your-streamlit-app>.streamlit.app`

## Performance Check

```bash
python scripts/performance_check.py
```

Prints:
- hold-out meta-model accuracy
- top-3 accuracy

## Screenshots

Add screenshots after deployment:
- `docs/images/upload_and_summary.png`
- `docs/images/top3_recommendations.png`
