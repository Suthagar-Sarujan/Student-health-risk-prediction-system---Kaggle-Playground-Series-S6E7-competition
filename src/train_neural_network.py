
from sklearn.neural_network import MLPClassifier

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

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        max_iter=200,
    )

    with Timer() as t:
        model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_test_enc)
    y_proba = model.predict_proba(X_test_enc)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, model.classes_)
    print("\n=== Neural Network (MLP) — Validation Performance ===")
    print(f"Stopped at iteration: {model.n_iter_}")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_model_and_metrics(
        model,
        model_name="neural_network",
        model_path=MODEL_PATHS["neural_network"],
        metrics=metrics,
        training_time_seconds=t.elapsed,
    )


if __name__ == "__main__":
    main()
