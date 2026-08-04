"""
generate_submission_tuned.py

Same as generate_submission.py but scores the competition test.csv with
the tuned LightGBM model produced by tune_lightgbm.py
(lightgbm_tuned_model.pkl) instead of the deployed baseline. Lets the
tuned model's Kaggle score be compared against the baseline submission
before deciding whether to promote it to DEPLOYED_MODEL_PATH.

Run AFTER tune_lightgbm.py, from the project root:
    python src/generate_submission_tuned.py
"""

import os

import joblib
import pandas as pd

from config import (
    ID_COL,
    MODELS_DIR,
    PROJECT_ROOT,
    TARGET_COL,
    TEST_CSV,
)
from preprocessing import load_preprocessor, transform

TUNED_MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_tuned_model.pkl")


def main() -> None:
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_df):,} rows from {TEST_CSV}")

    preprocessor = load_preprocessor()
    model = joblib.load(TUNED_MODEL_PATH)

    X_test_enc = transform(test_df, preprocessor)
    predicted_classes = model.predict(X_test_enc)

    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET_COL: predicted_classes,
    })

    output_path = f"{PROJECT_ROOT}/submission_lightgbm_tuned.csv"
    submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(submission.head())
    print(f"\nPredicted class distribution:")
    print(submission[TARGET_COL].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()
