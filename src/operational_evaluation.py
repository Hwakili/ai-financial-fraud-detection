"""Phase 3c: operational relevance layer.

Added following further investigation, which noted that F1/MCC/AUC are the
right metrics for comparing models on paper, but don't answer the question a
fraud operations team actually has: "if we can only manually review N alerts
a day, how many of those are genuine fraud, and how much fraud do we still
miss?" This module adds three things the standard classification metrics
don't cover:

1. Precision/recall at a FIXED ALERT CAPACITY (top-k), rather than at a
   probability threshold - the operationally realistic constraint for a
   team with finite investigator hours.
2. A cost-sensitivity analysis with EXPLICITLY STATED assumed costs - framed
   as illustrative, not as real-world financial figures this project can
   defend, since this dataset carries no investigation-cost or recovery-cost
   data (a limitation to state plainly in the dissertation, not paper over).
3. A time-aware (chronological) split, exploiting the one genuinely temporal
   feature this dataset provides (Time = seconds since the first transaction
   in the two-day collection window) - training on the earlier portion and
   testing on the later portion, closer to how a deployed model would
   actually be evaluated (predicting the future from the past) than a random
   stratified split, which lets the model implicitly "see" the full time
   range during training.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.baseline_models import train_logistic_regression, train_random_forest, train_xgboost
from src.evaluate import compute_metrics, select_threshold
from src.preprocessing import scale_features


# ---------------------------------------------------------------------------
# 1. Precision / recall at a fixed alert capacity (top-k)
# ---------------------------------------------------------------------------

def precision_recall_at_k(y_true, y_proba, k_values=(50, 100, 500)) -> pd.DataFrame:
    """For each k in k_values: take the k transactions with the highest
    predicted fraud probability (regardless of the tuned classification
    threshold), and report how many are genuine fraud.

    This answers a different, more operational question than the
    threshold-based metrics elsewhere in this project: "given capacity to
    investigate exactly k alerts, what would we get?" rather than "given this
    probability cutoff, what precision/recall results?" The two views can
    disagree - a model might have poor precision at its F1-optimal threshold
    but still put nearly all its true fraud cases in the very top of its
    ranking, which is exactly the pattern a strong AUC/PR-AUC with weak F1 at
    threshold implies (see this project's RXT discussion).

    k values are capped at len(y_true) to avoid an out-of-range request
    silently returning the whole test set.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_total_fraud = int(y_true.sum())

    order = np.argsort(-y_proba)  # descending by predicted probability

    rows = []
    for k in k_values:
        k_capped = min(k, len(y_true))
        top_k_idx = order[:k_capped]
        true_positives_in_k = int(y_true[top_k_idx].sum())

        rows.append({
            "k": k,
            "k_used": k_capped,
            "true_positives_in_top_k": true_positives_in_k,
            "precision_at_k": true_positives_in_k / k_capped if k_capped > 0 else np.nan,
            "recall_at_k": true_positives_in_k / n_total_fraud if n_total_fraud > 0 else np.nan,
            "total_fraud_in_test_set": n_total_fraud,
        })

    return pd.DataFrame(rows)


def precision_recall_at_k_all_models(predictions: dict, k_values=(50, 100, 500)) -> pd.DataFrame:
    """precision_recall_at_k for every saved model; saves a combined CSV.

    predictions: {model_name: (y_true, y_proba)}, typically from
    evaluate.load_all_predictions().
    """
    frames = []
    for model_name, (y_true, y_proba) in predictions.items():
        df = precision_recall_at_k(y_true, y_proba, k_values)
        df.insert(0, "model", model_name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    config.ensure_directories()
    combined.to_csv(config.RESULTS_METRICS_DIR / "precision_at_k.csv", index=False)
    return combined


def plot_precision_at_k(pak_df: pd.DataFrame, save_path=None):
    """Line chart: precision_at_k across k values, one line per model."""
    save_path = save_path or config.RESULTS_FIGURES_DIR / "precision_at_k.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, group in pak_df.groupby("model"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["precision_at_k"], marker="o", label=model_name)

    ax.set_xlabel("Alert capacity (top k transactions by predicted probability)")
    ax.set_ylabel("Precision within top k")
    ax.set_title("Precision at Fixed Alert Capacity")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 2. Cost-sensitivity analysis (assumed costs, explicitly labelled as such)
# ---------------------------------------------------------------------------

def expected_cost(y_true, y_pred, cost_fp: float, cost_fn: float) -> dict:
    """Total and mean assumed cost for one model's predictions at one threshold.

    cost_fp: assumed cost of a false positive (investigator time reviewing a
    genuine transaction wrongly flagged; customer friction from a declined/
    delayed legitimate payment).
    cost_fn: assumed cost of a false negative (the fraud loss that goes
    undetected).

    These are ILLUSTRATIVE ASSUMPTIONS, not figures derived from this
    dataset or any real institution's data - the Kaggle/ULB dataset carries
    no investigation-cost or recovery-cost information. State the assumed
    values explicitly wherever this function's output is reported (the
    dissertation's limitations section should say so directly, not just here
    in a code comment).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total_cost = fp * cost_fp + fn * cost_fn
    return {
        "cost_fp_assumed": cost_fp,
        "cost_fn_assumed": cost_fn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "total_assumed_cost": total_cost,
        "mean_assumed_cost_per_transaction": total_cost / len(y_true),
    }


def cost_sensitivity_sweep(
    y_true,
    y_proba,
    cost_fn_values=(50, 100, 200, 500),
    cost_fp: float = 5.0,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """For each assumed false-negative cost, find the threshold that MINIMISES
    total assumed cost (not the F1-maximising threshold used elsewhere), and
    report how that cost-optimal threshold and its resulting precision/recall
    shift as the false-negative cost assumption changes.

    cost_fp is held fixed (default: an illustrative low cost representing
    investigator review time for one flagged transaction) while cost_fn is
    swept across a range representing "a missed fraud is worth roughly
    10x-100x an investigation" - a deliberately wide range, since this
    project has no real basis for pinning down the true ratio and the point
    of a sensitivity analysis is to show how much the "right" threshold
    depends on that assumption, not to claim a single correct value.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    candidate_thresholds = np.quantile(y_proba, np.linspace(0.5, 0.9999, n_thresholds))

    rows = []
    for cost_fn in cost_fn_values:
        best_threshold = None
        best_cost = np.inf
        for threshold in candidate_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            result = expected_cost(y_true, y_pred, cost_fp, cost_fn)
            if result["total_assumed_cost"] < best_cost:
                best_cost = result["total_assumed_cost"]
                best_threshold = threshold
                best_result = result

        best_result["cost_optimal_threshold"] = best_threshold
        best_result["precision"] = (
            best_result["true_positives"] / (best_result["true_positives"] + best_result["false_positives"])
            if (best_result["true_positives"] + best_result["false_positives"]) > 0 else np.nan
        )
        best_result["recall"] = (
            best_result["true_positives"] / (best_result["true_positives"] + best_result["false_negatives"])
            if (best_result["true_positives"] + best_result["false_negatives"]) > 0 else np.nan
        )
        rows.append(best_result)

    return pd.DataFrame(rows)


def cost_sensitivity_all_models(predictions: dict, cost_fn_values=(50, 100, 200, 500), cost_fp: float = 5.0) -> pd.DataFrame:
    """cost_sensitivity_sweep for every saved model; saves a combined CSV."""
    frames = []
    for model_name, (y_true, y_proba) in predictions.items():
        df = cost_sensitivity_sweep(y_true, y_proba, cost_fn_values, cost_fp)
        df.insert(0, "model", model_name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    config.ensure_directories()
    combined.to_csv(config.RESULTS_METRICS_DIR / "cost_sensitivity_analysis.csv", index=False)
    return combined


# ---------------------------------------------------------------------------
# 3. Time-aware (chronological) split robustness test
# ---------------------------------------------------------------------------

CHRONOLOGICAL_TRAINERS = {
    "Logistic Regression": train_logistic_regression,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
}


def chronological_split(df: pd.DataFrame, train_frac: float = 0.64, val_frac: float = 0.16):
    """Split by Time (ascending) rather than randomly, matching the 64/16/20
    proportions used elsewhere so results are comparable in scale.

    This is deliberately NOT stratified (an ordered split can't be, by
    definition - you take whichever transactions happen to fall in the
    earlier or later time window). Report the resulting fraud counts per
    split alongside any metric from this function, since an uneven fraud
    distribution across the time window is itself a finding worth discussing,
    not just a nuisance to note in passing.
    """
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    return train_df, val_df, test_df


def run_chronological_robustness_test(df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate every baseline under a chronological split, using
    the identical modelling/thresholding protocol used for the headline
    random-split result, so the two are comparable as a robustness check -
    NOT as a replacement for the headline result, which remains the random
    stratified split reported in Chapter 5.

    A material gap between the random-split and chronological-split numbers
    for the same model is itself a finding: it would indicate the fraud
    patterns in this dataset are not stationary across the two-day collection
    window, which has direct implications for how confidently the headline
    result generalises to deployment (where a model only ever sees the
    future, never a random sample of it).
    """
    train_df, val_df, test_df = chronological_split(df)

    fraud_counts = {
        "train_fraud": int(train_df[config.TARGET_COLUMN].sum()),
        "train_n": len(train_df),
        "val_fraud": int(val_df[config.TARGET_COLUMN].sum()),
        "val_n": len(val_df),
        "test_fraud": int(test_df[config.TARGET_COLUMN].sum()),
        "test_n": len(test_df),
    }

    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_val = val_df.drop(columns=[config.TARGET_COLUMN])
    y_val = val_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]

    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

    rows = []
    for model_name, trainer in CHRONOLOGICAL_TRAINERS.items():
        model = trainer(X_train_s, y_train)

        y_proba_val = model.predict_proba(X_val_s)[:, 1]
        threshold = select_threshold(y_val, y_proba_val)

        y_proba_test = model.predict_proba(X_test_s)[:, 1]
        y_pred_test = (y_proba_test >= threshold).astype(int)

        metrics = compute_metrics(y_test, y_pred_test, y_proba_test)
        metrics["model"] = model_name
        metrics["threshold"] = threshold
        metrics["split_type"] = "chronological"
        metrics.update(fraud_counts)
        rows.append(metrics)

    result = pd.DataFrame(rows)
    config.ensure_directories()
    result.to_csv(config.RESULTS_METRICS_DIR / "chronological_split_robustness.csv", index=False)
    return result


if __name__ == "__main__":
    from src.evaluate import load_all_predictions

    predictions = load_all_predictions()
    if not predictions:
        print("No saved predictions found - run Phase 3 first.")
    else:
        print("=== Precision at fixed alert capacity ===")
        pak_df = precision_recall_at_k_all_models(predictions, k_values=(50, 100, 500))
        print(pak_df)
        plot_precision_at_k(pak_df)

        print("\n=== Cost sensitivity analysis (illustrative assumed costs) ===")
        cost_df = cost_sensitivity_all_models(predictions, cost_fn_values=(50, 100, 200, 500), cost_fp=5.0)
        print(cost_df[["model", "cost_fn_assumed", "cost_optimal_threshold", "precision", "recall"]])

    print("\n=== Chronological split robustness test ===")
    raw_path = config.RAW_DATA_FILE
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        chrono_result = run_chronological_robustness_test(df)
        print(chrono_result[["model", "f1", "mcc", "auc_roc", "threshold", "test_fraud", "test_n"]])
    else:
        print(f"Raw dataset not found at {raw_path} - skipping chronological split test.")
