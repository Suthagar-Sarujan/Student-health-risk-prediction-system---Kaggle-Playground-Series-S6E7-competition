"""
train_lightgbm.py

LightGBM classifier — this is the model deployed in the Streamlit app
(see Section D for the justification: best macro recall/ROC-AUC
trade-off relative to training cost among the six candidates).

Run from the project root:
    python src/train_lightgbm.py
"""

from lightgbm import LGBMClassifier

from config import MODEL_PATHS
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

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        class_weight="balanced",
        objective="multiclass",
        random_state=42,
        n_jobs=-1,
    )

    with Timer() as t:
        model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_test_enc)
    y_proba = model.predict_proba(X_test_enc)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, model.classes_)
    print("\n=== LightGBM — Validation Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_model_and_metrics(
        model,
        model_name="lightgbm",
        model_path=MODEL_PATHS["lightgbm"],
        metrics=metrics,
        training_time_seconds=t.elapsed,
    )

    print("\nThis is the DEPLOYED model used by app/app.py.")


if __name__ == "__main__":
    main()
