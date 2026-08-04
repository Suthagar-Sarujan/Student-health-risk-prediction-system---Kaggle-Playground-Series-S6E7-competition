"""
generate_submission.py

Loads the fitted preprocessing pipeline and the deployed LightGBM model,
applies the SAME train-derived imputation to the competition test.csv
(never recomputed from the test set, to avoid leakage), and writes the
Kaggle submission file.

Run from the project root:
    python src/generate_submission.py
"""

import joblib
import pandas as pd

from config import (
    DEPLOYED_MODEL_PATH,
    ID_COL,
    PROJECT_ROOT,
    TARGET_COL,
    TEST_CSV,
)
from preprocessing import load_preprocessor, transform


def main() -> None:
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_df):,} rows from {TEST_CSV}")

    preprocessor = load_preprocessor()
    model = joblib.load(DEPLOYED_MODEL_PATH)

    X_test_enc = transform(test_df, preprocessor)
    predicted_classes = model.predict(X_test_enc)

    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET_COL: predicted_classes,
    })

    output_path = f"{PROJECT_ROOT}/submission_lightgbm.csv"
    submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(submission.head())
    print(f"\nPredicted class distribution:")
    print(submission[TARGET_COL].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()
