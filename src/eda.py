"""
eda.py

Loads train.csv, checks missingness and correlation structure, and
generates the EDA figures referenced in Section C of the report:

  Figure 1 - class_balance.png                (target distribution)
  Figure 2 - missingness.png                   (missing values per column)
  Figure 3 - correlation_heatmap.png           (numeric feature correlation)
  Figure 4 - stress_level_vs_target.png        (non-monotonic stress effect)
  Figure 5 - feature_distributions_by_class.png (numeric distributions by class)

Run from the project root:
    python src/eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TRAIN_CSV,
    FIGURES_DIR,
    TARGET_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

sns.set_style("whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def report_missingness(df: pd.DataFrame) -> pd.Series:
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print("\n=== Missing values (%) ===")
    print(missing_pct[missing_pct > 0])
    return missing_pct


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df[TARGET_COL].value_counts(normalize=True) * 100
    print("\n=== Target class balance (%) ===")
    print(counts)

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#e74c3c", "#3498db", "#f39c12"])
    plt.title("Target Class Balance (health_condition)")
    plt.ylabel("Percentage of records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/class_balance.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/class_balance.png")


def plot_missingness(missing_pct: pd.Series) -> None:
    missing_pct = missing_pct[missing_pct > 0].sort_values()
    plt.figure(figsize=(7, 5))
    missing_pct.plot(kind="barh", color="#8e44ad")
    plt.title("Missingness by Feature (%)")
    plt.xlabel("% missing")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/missingness.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/missingness.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[NUMERIC_FEATURES].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/correlation_heatmap.png")


def plot_stress_vs_target(df: pd.DataFrame) -> None:
    if "stress_level" not in df.columns:
        print("stress_level column not found, skipping Figure 4")
        return

    ct = pd.crosstab(df["stress_level"], df[TARGET_COL], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, figsize=(7, 5),
            color=["#e74c3c", "#3498db", "#f39c12"])
    plt.title("stress_level vs health_condition (row %)")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="health_condition")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/stress_level_vs_target.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/stress_level_vs_target.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    n = len(NUMERIC_FEATURES)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, feat in enumerate(NUMERIC_FEATURES):
        for cls in df[TARGET_COL].dropna().unique():
            subset = df.loc[df[TARGET_COL] == cls, feat].dropna()
            sns.kdeplot(subset, ax=axes[i], label=str(cls), fill=True, alpha=0.3)
        axes[i].set_title(feat)
        axes[i].legend(fontsize=7)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_distributions_by_class.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/feature_distributions_by_class.png")


def main() -> None:
    df = load_data()
    missing_pct = report_missingness(df)

    plot_class_balance(df)
    plot_missingness(missing_pct)
    plot_correlation_heatmap(df)
    plot_stress_vs_target(df)
    plot_feature_distributions(df)

    print("\nEDA complete. All figures saved to the figures/ directory.")


if __name__ == "__main__":
    main()
