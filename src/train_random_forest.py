

from sklearn.ensemble import RandomForestClassifier

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

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    with Timer() as t:
        model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_test_enc)
    y_proba = model.predict_proba(X_test_enc)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, model.classes_)
    print("\n=== Random Forest — Validation Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_model_and_metrics(
        model,
        model_name="random_forest",
        model_path=MODEL_PATHS["random_forest"],
        metrics=metrics,
        training_time_seconds=t.elapsed,
    )


if __name__ == "__main__":
    main()
