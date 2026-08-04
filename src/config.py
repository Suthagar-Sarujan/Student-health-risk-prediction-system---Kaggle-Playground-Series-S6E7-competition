"""
config.py
Shared constants used by every script in this project so that the
feature list, target name, and file paths stay consistent between
training, evaluation, submission generation, and the Streamlit app.
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
FEATURE_ORDER_PATH = os.path.join(MODELS_DIR, "feature_order.json")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

MODEL_PATHS = {
    "logistic_regression": os.path.join(MODELS_DIR, "logistic_regression_model.pkl"),
    "random_forest": os.path.join(MODELS_DIR, "random_forest_model.pkl"),
    "xgboost": os.path.join(MODELS_DIR, "xgboost_model.pkl"),
    "lightgbm": os.path.join(MODELS_DIR, "lightgbm_model.pkl"),
    "catboost": os.path.join(MODELS_DIR, "catboost_model.pkl"),
    "neural_network": os.path.join(MODELS_DIR, "neural_network_model.pkl"),
}

DEPLOYED_MODEL_NAME = "lightgbm"
DEPLOYED_MODEL_PATH = MODEL_PATHS[DEPLOYED_MODEL_NAME]

# ---------------------------------------------------------------------
# Schema (Student Health Risk — Kaggle Playground Series S6E7)
# ---------------------------------------------------------------------
ID_COL = "id"
TARGET_COL = "health_condition"
TARGET_CLASSES = ["at-risk", "fit", "unhealthy"]

NUMERIC_FEATURES = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

CATEGORICAL_FEATURES = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20
