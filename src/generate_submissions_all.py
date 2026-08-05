

import os

import joblib
import pandas as pd

from config import (
    ID_COL,
    MODEL_PATHS,
    MODELS_DIR,
    PROJECT_ROOT,
    TARGET_CLASSES,
    TARGET_COL,
    TEST_CSV,
)
from preprocessing import load_preprocessor, transform

# xgboost and neural_network were trained on integer-encoded labels
# (see train_xgboost.py / train_neural_network.py) - everything else
# (logistic_regression, random_forest, lightgbm, catboost) was trained
# directly on the string labels and needs no mapping.
INTEGER_LABEL_MODELS = {"xgboost"}

TUNED_MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_tuned_model.pkl")


def predict_for_model(model, model_name: str, X_enc):
    """Handle the string-vs-integer label split across the six models,
    plus CatBoost's quirk of returning a 2D (n, 1) array from .predict()
    instead of a flat 1D array (same fix already applied in
    train_catboost.py's local evaluation step)."""
    raw_pred = model.predict(X_enc)
    if hasattr(raw_pred, "ravel"):
        raw_pred = raw_pred.ravel()
    if model_name in INTEGER_LABEL_MODELS:
        idx_to_class = {i: c for i, c in enumerate(TARGET_CLASSES)}
        return [idx_to_class[i] for i in raw_pred]
    return raw_pred


def write_submission(predicted_classes, ids, model_label: str) -> str:
    submission = pd.DataFrame({ID_COL: ids, TARGET_COL: predicted_classes})
    output_path = os.path.join(PROJECT_ROOT, f"submission_{model_label}.csv")
    submission.to_csv(output_path, index=False)

    print(f"\n=== {model_label} ===")
    print(f"Saved to {output_path}")
    print(submission.head(3).to_string(index=False))
    print("Predicted class distribution:")
    print((submission[TARGET_COL].value_counts(normalize=True) * 100).round(1))
    return output_path


def main() -> None:
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_df):,} rows from {TEST_CSV}")

    preprocessor = load_preprocessor()
    X_test_enc = transform(test_df, preprocessor)

    # Models to submit - all six base models plus the tuned LightGBM
    # (handled separately below). Logistic Regression and the Neural
    # Network are included here for completeness (see Section E.3 of
    # the report for the original reasoning on why they were initially
    # left out of the Kaggle-submitted set: Logistic Regression is the
    # weakest performer on every metric, and the Neural Network's
    # precision-favouring behaviour differs from the recall-favouring
    # tree ensembles this project otherwise prioritises).
    models_to_submit = [
        "lightgbm",
        "random_forest",
        "xgboost",
        "catboost",
        "logistic_regression",
        "neural_network",
    ]

    generated_paths = []
    for model_name in models_to_submit:
        model_path = MODEL_PATHS[model_name]
        if not os.path.exists(model_path):
            print(f"Skipping '{model_name}' - model file not found at {model_path}. "
                  f"Train it first with train_{model_name}.py")
            continue

        model = joblib.load(model_path)
        predicted_classes = predict_for_model(model, model_name, X_test_enc)
        path = write_submission(predicted_classes, test_df[ID_COL], model_name)
        generated_paths.append(path)

    # Tuned LightGBM, if it has been trained via tune_lightgbm.py
    if os.path.exists(TUNED_MODEL_PATH):
        tuned_model = joblib.load(TUNED_MODEL_PATH)
        predicted_classes = predict_for_model(tuned_model, "lightgbm_tuned", X_test_enc)
        path = write_submission(predicted_classes, test_df[ID_COL], "lightgbm_tuned")
        generated_paths.append(path)
    else:
        print(f"\nNote: {TUNED_MODEL_PATH} not found - run tune_lightgbm.py first "
              f"if you want a submission_lightgbm_tuned.csv too.")

    print(f"\nGenerated {len(generated_paths)} submission file(s):")
    for p in generated_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
