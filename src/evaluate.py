"""Shared evaluation pipeline used by Phase 2 (imbalance comparison), Phase 3
(baselines), Phase 4 (RXT), and Phase 5 (comparative analysis).

Kept as one module, deliberately, so every model - Logistic Regression through
to the RXT deep learning model - is scored with the exact same code. Evaluating
models with subtly different metric implementations is a common source of
invalid comparisons; this file exists to rule that out entirely.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

MASTER_METRICS_CSV = config.RESULTS_METRICS_DIR / "model_comparison.csv"
PREDICTIONS_DIR = config.RESULTS_METRICS_DIR / "predictions"
PREDICTIONS_MANIFEST = PREDICTIONS_DIR / "manifest.csv"


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Accuracy, precision, recall, F1, AUC-ROC, MCC for one model's predictions.

    Accuracy is included but is not the metric to optimise for: with a 0.172%
    fraud rate, a model that always predicts "genuine" scores ~99.8% accuracy
    while catching zero fraud. F1 and MCC are the metrics that matter here,
    per the ToR's evaluation plan.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    metrics["auc_roc"] = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir=None):
    """Confusion matrix heatmap. False negatives (missed fraud) are the cell
    that matters most in this problem, so labels are explicit about direction.
    """
    save_dir = save_dir or config.RESULTS_FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Genuine", "Predicted Fraud"],
        yticklabels=["Actual Genuine", "Actual Fraud"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    path = save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def append_metrics_to_master(metrics: dict, model_name: str, csv_path=None) -> pd.DataFrame:
    """Upsert one model's metrics row into the master comparison CSV.

    This CSV is the single running table referenced throughout Chapter 6 -
    every phase that trains a model adds/updates its row here rather than
    keeping separate, hard-to-compare result files per model.
    """
    csv_path = csv_path or MASTER_METRICS_CSV
    config.ensure_directories()

    row = {"model": model_name, **metrics}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["model"] != model_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
    return df


def save_predictions(model_name: str, y_true, y_proba, save_dir=None) -> None:
    """Persist one model's test-set predictions to disk, keyed by model name.

    Exists so that Phase 3 (baselines) and Phase 4 (RXT) can each be run as
    independent scripts - per the README's "run each script separately" - while
    Phase 5 can still overlay every model's ROC/PR curve by loading these back,
    without needing one monolithic script that trains everything in memory.
    """
    save_dir = save_dir or PREDICTIONS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    key = model_name.lower().replace(" ", "_")

    np.save(save_dir / f"{key}_y_true.npy", np.asarray(y_true))
    np.save(save_dir / f"{key}_y_proba.npy", np.asarray(y_proba))

    manifest_path = save_dir / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["key"] != key]
    else:
        manifest = pd.DataFrame(columns=["key", "model"])
    manifest = pd.concat([manifest, pd.DataFrame([{"key": key, "model": model_name}])], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)


def load_all_predictions(save_dir=None) -> dict:
    """Load every saved model's (y_true, y_proba) back, keyed by original model name."""
    save_dir = save_dir or PREDICTIONS_DIR
    manifest_path = save_dir / "manifest.csv"

    if not manifest_path.exists():
        return {}

    manifest = pd.read_csv(manifest_path)
    predictions = {}
    for _, row in manifest.iterrows():
        y_true = np.load(save_dir / f"{row['key']}_y_true.npy")
        y_proba = np.load(save_dir / f"{row['key']}_y_proba.npy")
        predictions[row["model"]] = (y_true, y_proba)

    return predictions


def plot_roc_curves(predictions: dict, save_path=None):
    """Overlaid ROC curves for every model. predictions = {model_name: (y_true, y_proba)}."""
    save_path = save_path or config.RESULTS_FIGURES_DIR / "roc_curves_comparison.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, (y_true, y_proba) in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_pr_curves(predictions: dict, save_path=None):
    """Overlaid precision-recall curves - typically more informative than ROC
    under heavy class imbalance, since PR curves are insensitive to the large
    true-negative count that dominates ROC's false-positive-rate axis.
    """
    save_path = save_path or config.RESULTS_FIGURES_DIR / "pr_curves_comparison.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, (y_true, y_proba) in predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(recall, precision, label=model_name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_grouped_bar_metrics(metrics_df: pd.DataFrame, save_path=None):
    """Grouped bar chart: each metric, side by side, across every model."""
    save_path = save_path or config.RESULTS_FIGURES_DIR / "grouped_metrics_comparison.png"

    plot_metrics = ["precision", "recall", "f1", "auc_roc", "mcc"]
    models = metrics_df["model"].tolist()
    n_models = len(models)
    n_metrics = len(plot_metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(models):
        values = metrics_df.loc[metrics_df["model"] == model_name, plot_metrics].values.flatten()
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(plot_metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison Across Metrics")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def evaluate_model(model_name, y_true, y_pred, y_proba=None) -> dict:
    """Convenience wrapper: compute metrics, save confusion matrix, update master CSV."""
    metrics = compute_metrics(y_true, y_pred, y_proba)
    plot_confusion_matrix(y_true, y_pred, model_name)
    append_metrics_to_master(metrics, model_name)
    return metrics
