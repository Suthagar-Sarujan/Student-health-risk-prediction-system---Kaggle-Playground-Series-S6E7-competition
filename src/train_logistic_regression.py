"""
train_logistic_regression.py

Baseline model. Trained with class_weight='balanced' to account for
the severe class imbalance identified in the EDA (Section C).

Run from the project root:
    python src/train_logistic_regression.py
"""

from sklearn.linear_model import LogisticRegression

from config import MODEL_PATHS, TARGET_CLASSES
from train_utils import (
    Timer,
    evaluate_predictions,
    get_or_fit_preprocessor,
    load_and_split,
    save_model_and_metrics,
)
from preprocessing import transform


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = get_or_fit_preprocessor(X_train)

    X_train_enc = transform(X_train, preprocessor)
    X_test_enc = transform(X_test, preprocessor)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    with Timer() as t:
        model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_test_enc)
    y_proba = model.predict_proba(X_test_enc)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, model.classes_)
    print("\n=== Logistic Regression — Validation Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_model_and_metrics(
        model,
        model_name="logistic_regression",
        model_path=MODEL_PATHS["logistic_regression"],
        metrics=metrics,
        training_time_seconds=t.elapsed,
    )


if __name__ == "__main__":
    main()
