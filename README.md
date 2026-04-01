# Meta-Learning Model Recommender

Production-grade Python project that learns from multiple OpenML datasets and recommends the most suitable machine learning model for new tabular datasets.

## Features

- Downloads and validates 20-50+ datasets from OpenML.
- Handles invalid datasets safely (missing target, empty data, duplicates, malformed records).
- Performs robust preprocessing for mixed data types.
- Extracts fixed-order meta-features from datasets.
- Evaluates candidate base learners with cross-validation.
- Creates a meta-dataset (`meta_X`, `meta_y`) and trains a `RandomForestClassifier` meta-model.
- Supports top-1 and top-k recommendations.
- Includes CLI for training and CSV-based prediction.

## Project Structure

```text
src/meta_recommender/
  data_loader.py      # OpenML ingestion + validation
  features.py         # cleaning, task detection, meta-feature extraction
  evaluator.py        # candidate model evaluation
  predictor.py        # meta-model training + inference + persistence
  pipeline.py         # orchestration and CLI logic
  cli.py              # console entrypoint
```

## Installation

```bash
pip install -e .
```

## Usage

### Train meta-model

```bash
meta-recommender --train --openml-limit 30
```

### Recommend model for a new CSV

```bash
meta-recommender --predict-csv /path/to/new_dataset.csv
```

Output includes:
- best model
- top 3 models with probabilities
- extracted meta-features

## Notes

- For prediction from CSV without explicit target column, the pipeline extracts structural meta-features and class-imbalance defaults to 1.0.
- Regression uses negative RMSE scorer convention from scikit-learn (`higher is better` when less negative).
- Model evaluation is protected with timeout and exception handling.
