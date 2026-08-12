"""Phase 1: exploratory data analysis.

Reusable EDA functions called from notebooks/01_eda.ipynb. Kept as plain functions
(rather than only notebook cells) so they are unit-testable and so Phase 2
preprocessing can reuse load_data() without duplicating logic.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic schema/quality check before anything downstream touches the data.

    Catches the kind of problem that would otherwise surface as a confusing
    error three phases later - a truncated download, a schema change, a
    corrupted CSV - right at the point of loading instead.
    """
    required = config.EXPECTED_FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    if df.empty:
        raise ValueError("Dataset has no rows.")
    if df[required].isnull().any().any():
        raise ValueError("Dataset contains missing values in a required column.")
    if not np.isfinite(df[config.EXPECTED_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite feature values (NaN/inf).")

    classes = set(df[config.TARGET_COLUMN].unique())
    if not classes.issubset({0, 1}):
        raise ValueError(f"Class column must be binary (0/1); found {classes}.")

    return df


def load_data(path=None) -> pd.DataFrame:
    """Load and validate the raw dataset from disk."""
    path = path or config.RAW_DATA_FILE
    df = pd.read_csv(path)
    return validate_dataset(df)


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Return the core EDA facts used to open Chapter 4.1 (Dataset Description)."""
    class_counts = df[config.TARGET_COLUMN].value_counts()
    class_proportions = df[config.TARGET_COLUMN].value_counts(normalize=True)
    return {
        "shape": df.shape,
        "missing_values_total": int(df.isnull().sum().sum()),
        "class_counts": class_counts.to_dict(),
        "fraud_rate_pct": round(class_proportions.get(1, 0.0) * 100, 3),
        "amount_describe": df["Amount"].describe().to_dict(),
        "time_describe": df["Time"].describe().to_dict(),
    }


def plot_class_distribution(df: pd.DataFrame, save_path=None):
    """Bar chart of genuine vs fraudulent counts, log-scale y-axis.

    Log scale is necessary because 492 vs 284,315 renders the fraud bar as
    invisible on a linear axis - the imbalance itself is the point of the plot.
    """
    save_path = save_path or config.RESULTS_FIGURES_DIR / "class_distribution.png"
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Genuine (0)", "Fraud (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("Number of transactions (log scale)")
    ax.set_title("Class Distribution: Genuine vs Fraudulent Transactions")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_amount_time_by_class(df: pd.DataFrame, save_dir=None):
    """Histograms of Amount and Time split by class.

    Amount and Time are the only two human-interpretable features (V1-V28 are
    anonymised PCA components), so they are the only ones worth visualising
    individually before modelling.
    """
    save_dir = save_dir or config.RESULTS_FIGURES_DIR
    paths = []

    for feature, xlabel in [("Amount", "Transaction Amount"), ("Time", "Time (seconds since first transaction)")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for ax, cls, label, color in zip(
            axes, [0, 1], ["Genuine", "Fraud"], ["#4C72B0", "#C44E52"]
        ):
            subset = df.loc[df[config.TARGET_COLUMN] == cls, feature]
            ax.hist(subset, bins=50, color=color, alpha=0.8)
            ax.set_title(f"{label} transactions")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
        fig.suptitle(f"{feature} Distribution by Class")
        fig.tight_layout()
        path = save_dir / f"{feature.lower()}_distribution_by_class.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_correlation_heatmap(df: pd.DataFrame, save_path=None):
    """Correlation heatmap across V1-V28 + Amount + Time.

    PCA components are constructed to be mutually near-uncorrelated, so this
    plot doubles as a sanity check on the provided data, not just an EDA figure.
    """
    save_path = save_path or config.RESULTS_FIGURES_DIR / "correlation_heatmap.png"
    feature_cols = [c for c in df.columns if c != config.TARGET_COLUMN]
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation Heatmap: V1-V28, Amount, Time")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def run_full_eda(path=None) -> dict:
    """Convenience entry point: load data, produce every figure, return summary stats."""
    df = load_data(path)
    config.ensure_directories()

    summary = summarize_dataset(df)
    plot_class_distribution(df)
    plot_amount_time_by_class(df)
    plot_correlation_heatmap(df)

    return summary


if __name__ == "__main__":
    result = run_full_eda()
    print(result)
