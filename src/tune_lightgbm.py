

import json
import os

from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from config import MODELS_DIR, RANDOM_STATE
from train_utils import (
    Timer,
    evaluate_predictions,
    get_or_fit_preprocessor,
    load_and_split,
    save_model_and_metrics,
)
from preprocessing import transform

TUNED_MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_tuned_model.pkl")
TUNED_PARAMS_PATH = os.path.join(MODELS_DIR, "lightgbm_tuned_params.json")

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 10),
    "num_leaves": randint(15, 100),
    "learning_rate": uniform(0.01, 0.29),
    "min_child_samples": randint(5, 100),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_alpha": uniform(0.0, 1.0),
    "reg_lambda": uniform(0.0, 1.0),
}

N_ITER = 25
CV_FOLDS = 3
# Parallelize across search candidates (outer), not within each LightGBM
# fit (inner) — nesting n_jobs=-1 on both levels oversubscribes the CPU
# and blew up memory usage on a 552k-row training set.
SEARCH_N_JOBS = 4


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = get_or_fit_preprocessor(X_train)

    X_train_enc = transform(X_train, preprocessor)
    X_test_enc = transform(X_test, preprocessor)

    base_model = LGBMClassifier(
        class_weight="balanced",
        objective="multiclass",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=SEARCH_N_JOBS,
        verbose=1,
        refit=True,
    )

    print(f"Running RandomizedSearchCV: {N_ITER} candidates x {CV_FOLDS} folds "
          f"= {N_ITER * CV_FOLDS} fits...")

    with Timer() as t:
        search.fit(X_train_enc, y_train)

    model = search.best_estimator_

    print(f"\nBest CV macro-F1: {search.best_score_:.4f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    y_pred = model.predict(X_test_enc)
    y_proba = model.predict_proba(X_test_enc)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, model.classes_)
    print("\n=== LightGBM (tuned) — Validation Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_model_and_metrics(
        model,
        model_name="lightgbm_tuned",
        model_path=TUNED_MODEL_PATH,
        metrics=metrics,
        training_time_seconds=t.elapsed,
    )

    best_params_record = {
        "best_params": {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()
        },
        "best_cv_score_f1_macro": round(search.best_score_, 4),
        "cv_folds": CV_FOLDS,
        "n_iter": N_ITER,
        "scoring": "f1_macro",
    }
    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump(best_params_record, f, indent=2)
    print(f"Saved best hyperparameters to {TUNED_PARAMS_PATH}")


if __name__ == "__main__":
    main()
