

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import (
    FIGURES_DIR,
    MODEL_PATHS,
    METRICS_PATH,
    TARGET_CLASSES,
)
from train_utils import load_and_split
from preprocessing import load_preprocessor, transform

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "neural_network": "Neural Network (MLP)",
}


def get_predictions(model, X_enc, model_name):
    """Handle the fact that XGBoost was trained on integer-encoded labels
    while the other five models were trained directly on string labels."""
    y_proba = model.predict_proba(X_enc)

    if model_name == "xgboost":
        idx_to_class = {i: c for i, c in enumerate(TARGET_CLASSES)}
        y_pred_idx = model.predict(X_enc)
        y_pred = np.array([idx_to_class[i] for i in y_pred_idx])
        classes = TARGET_CLASSES
    else:
        y_pred = model.predict(X_enc)
        classes = model.classes_

    return y_pred, y_proba, classes


def plot_confusion_matrices(results: dict) -> None:
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, (model_name, r) in enumerate(results.items()):
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=TARGET_CLASSES, normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_CLASSES)
        disp.plot(ax=axes[i], cmap="Greens", colorbar=False, values_format=".2f")
        axes[i].set_title(MODEL_DISPLAY_NAMES[model_name])

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrices.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/confusion_matrices.png")


def plot_roc_curves(results: dict) -> None:
    y_true_bin_cache = {}
    plt.figure(figsize=(8, 6))

    for model_name, r in results.items():
        y_true_bin = label_binarize(r["y_true"], classes=TARGET_CLASSES)
        y_true_bin_cache[model_name] = y_true_bin

        # macro-average ROC curve across the 3 classes for this model
        fpr_grid = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(fpr_grid)

        for c_idx in range(len(TARGET_CLASSES)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, c_idx], r["y_proba"][:, c_idx])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)

        mean_tpr /= len(TARGET_CLASSES)
        macro_auc = auc(fpr_grid, mean_tpr)

        plt.plot(fpr_grid, mean_tpr, label=f"{MODEL_DISPLAY_NAMES[model_name]} (AUC={macro_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves (macro-averaged across 3 classes)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/roc_curves.png")


def print_comparison_table() -> None:
    with open(METRICS_PATH, "r") as f:
        all_metrics = json.load(f)

    print("\n=== Model Comparison (macro-averaged, held-out test split) ===")
    header = f"{'Model':<22}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<9}{'ROC-AUC':<9}{'Train(s)':<9}"
    print(header)
    print("-" * len(header))
    for model_name, m in all_metrics.items():
        print(
            f"{MODEL_DISPLAY_NAMES.get(model_name, model_name):<22}"
            f"{m['accuracy']:<10}{m['precision_macro']:<11}"
            f"{m['recall_macro']:<9}{m['f1_macro']:<9}"
            f"{m['roc_auc_macro_ovr']:<9}{m['training_time_seconds']:<9}"
        )


def main() -> None:
    _, X_test, _, y_test = load_and_split()
    preprocessor = load_preprocessor()
    X_test_enc = transform(X_test, preprocessor)

    results = {}
    for model_name, path in MODEL_PATHS.items():
        try:
            model = joblib.load(path)
        except FileNotFoundError:
            print(f"Skipping '{model_name}' — model file not found at {path}. "
                  f"Train it first with train_{model_name}.py")
            continue

        y_pred, y_proba, _ = get_predictions(model, X_test_enc, model_name)
        results[model_name] = {"y_true": y_test.values, "y_pred": y_pred, "y_proba": y_proba}

    if not results:
        print("No trained models found. Run the train_*.py scripts first.")
        return

    plot_confusion_matrices(results)
    plot_roc_curves(results)
    print_comparison_table()


if __name__ == "__main__":
    main()
