
from catboost import CatBoostClassifier

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

    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.1,
        auto_class_weights="Balanced",
        random_state=42,
        verbose=False,
    )

    with Timer() as t:
        model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_test_enc)
    if hasattr(y_pred, "ravel"):
        y_pred = y_pred.ravel()
    y_proba = model.predict_proba(X_test_enc)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, model.classes_)
    print("\n=== CatBoost — Validation Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_model_and_metrics(
        model,
        model_name="catboost",
        model_path=MODEL_PATHS["catboost"],
        metrics=metrics,
        training_time_seconds=t.elapsed,
    )


if __name__ == "__main__":
    main()
