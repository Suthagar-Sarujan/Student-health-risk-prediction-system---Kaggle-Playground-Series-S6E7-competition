import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    FEATURE_ORDER_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
)


def build_preprocessor() -> ColumnTransformer:
    """Construct (but do not fit) the shared ColumnTransformer."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def fit_and_save_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on the training split only, then persist it."""
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train[FEATURE_ORDER])

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(FEATURE_ORDER, f, indent=2)

    print(f"Saved fitted preprocessor to {PREPROCESSOR_PATH}")
    print(f"Saved feature order to {FEATURE_ORDER_PATH}")
    return preprocessor


def load_preprocessor() -> ColumnTransformer:
    """Load the already-fitted preprocessor (used by eval/submission/app)."""
    return joblib.load(PREPROCESSOR_PATH)


def load_feature_order() -> list:
    with open(FEATURE_ORDER_PATH, "r") as f:
        return json.load(f)


def transform(df: pd.DataFrame, preprocessor: ColumnTransformer):
    """Apply the fitted preprocessor to any new dataframe with the
    same raw columns (validation split, Kaggle test.csv, or a single
    row submitted through the Streamlit form)."""
    return preprocessor.transform(df[FEATURE_ORDER])
