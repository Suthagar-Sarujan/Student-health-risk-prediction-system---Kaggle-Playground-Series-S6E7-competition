

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import (
    CATEGORICAL_FEATURES,
    DEPLOYED_MODEL_PATH,
    FIGURES_DIR,
    NUMERIC_FEATURES,
)
from train_utils import load_and_split
from preprocessing import load_preprocessor, transform


def get_encoded_feature_names(preprocessor) -> list:
    """Reconstruct human-readable feature names after the
    ColumnTransformer's StandardScaler + OneHotEncoder have expanded
    the categorical columns into many binary columns."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    return list(NUMERIC_FEATURES) + list(cat_names)


def plot_split_importance(model, feature_names: list) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:15]  # top 15

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#27ae60",
    )
    plt.title("LightGBM — Default (Split-Count) Feature Importance")
    plt.xlabel("Importance (number of splits)")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_importance_split.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/feature_importance_split.png")


def plot_shap_summary(model, X_sample_enc, feature_names: list) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_enc)

    # Multiclass LightGBM: shap>=0.45 returns a single (n_samples, n_features,
    # n_classes) ndarray; older versions return a list of per-class arrays.
    # Either way, average |SHAP| across classes and samples to get one
    # importance ranking per feature.
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    order = np.argsort(mean_abs_shap)[::-1][:15]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in order][::-1],
        [mean_abs_shap[i] for i in order][::-1],
        color="#2980b9",
    )
    plt.title("LightGBM — Mean |SHAP value| (averaged across all 3 classes)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/shap_summary.png")


def main(sample_size: int = 3000) -> None:
    _, X_test, _, _ = load_and_split()
    preprocessor = load_preprocessor()
    model = joblib.load(DEPLOYED_MODEL_PATH)

    feature_names = get_encoded_feature_names(preprocessor)

    plot_split_importance(model, feature_names)

    # SHAP on a 3,000-row sample for computational efficiency, as noted
    # in Section E ("averaged across all three classes over a 3,000-row sample")
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    X_sample_enc = transform(X_sample, preprocessor)
    if hasattr(X_sample_enc, "toarray"):
        X_sample_enc = X_sample_enc.toarray()

    plot_shap_summary(model, X_sample_enc, feature_names)


if __name__ == "__main__":
    main()
