"""
train_xgboost.py

XGBoost multiclass classifier. XGBoost has no built-in
class_weight='balanced' option, so per-sample weights are computed
via sklearn's compute_sample_weight and passed explicitly to fit().

Run from the project root:
    python src/train_xgboost.py
"""

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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

    # XGBoost requires integer-encoded labels
    class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    y_train_idx = y_train.map(class_to_idx)
    y_test_idx = y_test.map(class_to_idx)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_idx)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(TARGET_CLASSES),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    with Timer() as t:
        model.fit(X_train_enc, y_train_idx, sample_weight=sample_weights)

    y_pred_idx = model.predict(X_test_enc)
    y_proba = model.predict_proba(X_test_enc)

    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y_pred = [idx_to_class[i] for i in y_pred_idx]

    metrics = evaluate_predictions(y_test, y_pred, y_proba, TARGET_CLASSES)
    print("\n=== XGBoost — Validation Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_model_and_metrics(
        model,
        model_name="xgboost",
        model_path=MODEL_PATHS["xgboost"],
        metrics=metrics,
        training_time_seconds=t.elapsed,
    )


if __name__ == "__main__":
    main()
