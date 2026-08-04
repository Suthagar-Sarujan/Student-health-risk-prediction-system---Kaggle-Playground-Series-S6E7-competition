"""
train_utils.py

Shared helpers used by every train_*.py script: loading data, doing
the stratified 80/20 split, and saving a trained model + its metrics
in a consistent format so evaluate_models.py can compare all six
models on equal footing.
"""

import json
import os
import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import (
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_CSV,
)
from preprocessing import fit_and_save_preprocessor, load_preprocessor, transform


def load_and_split():
    """Load train.csv and produce a stratified 80/20 split on health_condition."""
    df = pd.read_csv(TRAIN_CSV)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


def get_or_fit_preprocessor(X_train: pd.DataFrame):
    """Reuse the saved preprocessor if it already exists (so all six
    training scripts share the exact same fitted transformer), otherwise
    fit it once on this training split and save it."""
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    if os.path.exists(preprocessor_path):
        return load_preprocessor()
    return fit_and_save_preprocessor(X_train)


def evaluate_predictions(y_true, y_pred, y_proba, classes) -> dict:
    """Compute the macro-averaged metric set used throughout Section E."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "roc_auc_macro_ovr": round(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes),
            4,
        ),
    }


def save_model_and_metrics(model, model_name: str, model_path: str, metrics: dict,
                            training_time_seconds: float) -> None:
    """Persist the trained model and append its metrics to the shared
    model_metrics.json file used to build the Section E comparison table."""
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)

    all_metrics[model_name] = {
        **metrics,
        "training_time_seconds": round(training_time_seconds, 2),
    }

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Updated {metrics_path} with '{model_name}' results")


class Timer:
    """Small context manager to time model.fit() calls consistently."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
