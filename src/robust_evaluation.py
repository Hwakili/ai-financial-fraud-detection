"""Phase 3b: statistical robustness layer for the baseline models.

Added following further investigation and assessment of the dissertation
draft, which identified that a single train/validation/test split per model -
while methodologically sound and leakage-free - doesn't on its own tell you
how much the reported F1/PR-AUC would vary on a different random split.
RXT already had 5-fold CV (see rxt_model.py); this module brings the same kind
of variability evidence to the baselines, plus two things neither the baselines
nor RXT had: bootstrapped confidence intervals on the actual held-out test
predictions, and a calibration assessment motivated directly by this project's
own threshold-tuning finding (see evaluate.select_threshold and CHANGELOG.md) -
if class weighting distorts predicted probabilities enough to make a 0.5
threshold meaningless, that same distortion should show up as poor calibration,
and it's worth checking that connection explicitly rather than asserting it.

Deliberately NOT extended to RXT: repeating RXT's ~21-minute CPU training time
across multiple repeats of a 5-fold CV would take hours, and RXT's own 5-fold
CV (already reported in Chapter 5/results/metrics/rxt_kfold_summary.csv) already
serves the "how variable is this model's performance" purpose that repeated CV
serves for the fast baselines here.
"""

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src import config
from src.baseline_models import train_logistic_regression, train_random_forest, train_xgboost
from src.evaluate import compute_metrics, select_threshold


# ---------------------------------------------------------------------------
# 1. Repeated stratified cross-validation for the baseline models
# ---------------------------------------------------------------------------

BASELINE_TRAINERS = {
    "Logistic Regression": train_logistic_regression,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
}


def repeated_cv_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Repeated stratified k-fold CV for every baseline model, reporting mean
    and standard deviation per metric - the same kind of variability evidence
    RXT's 5-fold CV already provides, extended to the baselines.

    Threshold protocol inside each fold: each fold is itself split further
    into an inner-train/inner-validation partition (80/20 of that fold's
    training portion), so the threshold is still tuned on data the model
    never trained on before being applied to that fold's held-out portion -
    the same validation-only rule used everywhere else in this project,
    applied consistently even inside cross-validation rather than relaxed
    "just for CV". This is what makes it a fair comparison to the headline
    Phase 3 result, not a looser, more favourable protocol.

    X, y should be the FULL pre-split feature/target arrays for whichever
    population you want variability evidence over - typically the combined
    train+validation portion (never including the final untouched test set,
    which must stay reserved for the one-shot Phase 3/5 headline comparison).
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)

    rows = []
    for model_name, trainer in BASELINE_TRAINERS.items():
        for fold_idx, (train_idx, holdout_idx) in enumerate(rskf.split(X_arr, y_arr)):
            X_fold_train_full = X_arr.iloc[train_idx]
            y_fold_train_full = y_arr.iloc[train_idx]
            X_fold_holdout = X_arr.iloc[holdout_idx]
            y_fold_holdout = y_arr.iloc[holdout_idx]

            # Inner split for threshold tuning, kept separate from this
            # fold's own held-out portion.
            n_inner_val = max(int(0.2 * len(X_fold_train_full)), 50)
            rng = np.random.RandomState(random_state + fold_idx)
            shuffled = rng.permutation(len(X_fold_train_full))
            inner_val_idx = shuffled[:n_inner_val]
            inner_train_idx = shuffled[n_inner_val:]

            X_inner_train = X_fold_train_full.iloc[inner_train_idx]
            y_inner_train = y_fold_train_full.iloc[inner_train_idx]
            X_inner_val = X_fold_train_full.iloc[inner_val_idx]
            y_inner_val = y_fold_train_full.iloc[inner_val_idx]

            model = trainer(X_inner_train, y_inner_train)

            y_proba_inner_val = model.predict_proba(X_inner_val)[:, 1]
            threshold = select_threshold(y_inner_val, y_proba_inner_val)

            y_proba_holdout = model.predict_proba(X_fold_holdout)[:, 1]
            y_pred_holdout = (y_proba_holdout >= threshold).astype(int)

            metrics = compute_metrics(y_fold_holdout, y_pred_holdout, y_proba_holdout)
            metrics["model"] = model_name
            metrics["fold"] = fold_idx
            metrics["threshold"] = threshold
            rows.append(metrics)

    per_fold = pd.DataFrame(rows)

    summary = (
        per_fold.groupby("model")[["accuracy", "precision", "recall", "f1", "mcc", "auc_roc", "pr_auc", "threshold"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    config.ensure_directories()
    per_fold.to_csv(config.RESULTS_METRICS_DIR / "baseline_repeated_cv_per_fold.csv", index=False)
    summary.to_csv(config.RESULTS_METRICS_DIR / "baseline_repeated_cv_summary.csv", index=False)

    return summary


# ---------------------------------------------------------------------------
# 2. Bootstrapped confidence intervals on the held-out test predictions
# ---------------------------------------------------------------------------

def bootstrap_metric_ci(
    y_true,
    y_proba,
    threshold: float,
    metric: str = "f1",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = config.RANDOM_SEED,
) -> dict:
    """Bootstrap confidence interval for one metric on one model's ALREADY-SAVED
    test predictions - no retraining required, since this resamples the
    (y_true, y_proba) pairs themselves rather than the raw data.

    metric: "f1" or "pr_auc". Both are the metrics this project's own
    threshold-tuning finding (see CHANGELOG.md) flagged as most sensitive to
    exactly the kind of measurement fragility a single point estimate can't
    reveal - a confidence interval says whether "Random Forest beats RXT by
    0.37 F1" is a robust conclusion or one that a different bootstrap sample
    could meaningfully narrow or widen.

    Percentile method: sort the n_bootstrap resampled metric values and take
    the (1-ci)/2 and 1-(1-ci)/2 quantiles. Simpler and more transparent than
    BCa correction for a dissertation-level analysis, and adequate here since
    the metric distributions involved aren't heavily skewed at this sample size.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rng = np.random.RandomState(random_state)

    def score(y_t, y_p):
        if metric == "pr_auc":
            return average_precision_score(y_t, y_p)
        y_pred = (y_p >= threshold).astype(int)
        return f1_score(y_t, y_pred, zero_division=0)

    point_estimate = score(y_true, y_proba)

    boot_scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        y_t_boot = y_true[idx]
        y_p_boot = y_proba[idx]
        # A bootstrap resample can occasionally contain zero fraud cases at
        # this prevalence (~0.17%); such resamples are skipped and redrawn
        # rather than left to produce an undefined metric.
        while y_t_boot.sum() == 0:
            idx = rng.randint(0, n, n)
            y_t_boot = y_true[idx]
            y_p_boot = y_proba[idx]
        boot_scores[i] = score(y_t_boot, y_p_boot)

    alpha = 1 - ci
    lower = float(np.quantile(boot_scores, alpha / 2))
    upper = float(np.quantile(boot_scores, 1 - alpha / 2))

    return {
        "metric": metric,
        "point_estimate": float(point_estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_width": upper - lower,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_all_models(predictions: dict, thresholds: dict, metrics=("f1", "pr_auc")) -> pd.DataFrame:
    """Run bootstrap_metric_ci for every saved model and every requested metric.

    predictions: {model_name: (y_true, y_proba)} - typically from
    evaluate.load_all_predictions().
    thresholds: {model_name: threshold} - the validation-tuned threshold each
    model was actually evaluated at (needed for F1; ignored for PR-AUC, which
    is threshold-independent).
    """
    rows = []
    for model_name, (y_true, y_proba) in predictions.items():
        threshold = thresholds.get(model_name, 0.5)
        for metric in metrics:
            result = bootstrap_metric_ci(y_true, y_proba, threshold, metric=metric)
            result["model"] = model_name
            rows.append(result)

    df = pd.DataFrame(rows)
    config.ensure_directories()
    df.to_csv(config.RESULTS_METRICS_DIR / "bootstrap_confidence_intervals.csv", index=False)
    return df


def plot_bootstrap_ci(bootstrap_df: pd.DataFrame, metric: str = "f1", save_path=None):
    """Error-bar chart: point estimate +/- 95% CI, one bar per model, for one metric."""
    save_path = save_path or config.RESULTS_FIGURES_DIR / f"bootstrap_ci_{metric}.png"
    subset = bootstrap_df[bootstrap_df["metric"] == metric].sort_values("point_estimate", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(subset))
    errors = [
        subset["point_estimate"] - subset["ci_lower"],
        subset["ci_upper"] - subset["point_estimate"],
    ]
    ax.barh(y_pos, subset["point_estimate"], xerr=errors, capsize=4, color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset["model"])
    ax.set_xlabel(metric.upper())
    ax.set_title(f"{metric.upper()} with 95% Bootstrap Confidence Interval")
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 3. Calibration assessment
# ---------------------------------------------------------------------------

def calibration_report(y_true, y_proba, model_name: str, n_bins: int = 10, save_dir=None) -> dict:
    """Reliability diagram + Brier score for one model's held-out predictions.

    Directly motivated by this project's threshold-tuning finding: class
    weighting (class_weight='balanced' / scale_pos_weight) was shown to push
    a fixed 0.5 threshold into an unusable region for Logistic Regression and
    RXT (see evaluate.select_threshold's docstring and CHANGELOG.md). If that
    diagnosis is right, the same class weighting should show up here as poor
    calibration - predicted probabilities that don't match observed fraud
    frequency - rather than being asserted only through the threshold finding
    on its own. A well-calibrated model's reliability curve sits on the
    diagonal; systematic deviation above or below it is what the threshold
    fix was implicitly correcting for.

    Brier score (mean squared error between predicted probability and actual
    outcome, 0 = perfect, 1 = worst) is reported as a single-number summary
    alongside the plot, since a plot alone isn't citable as a table value.
    """
    save_dir = save_dir or config.RESULTS_FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed fraud frequency (per bin)")
    ax.set_title(f"Calibration Curve — {model_name}\n(Brier score = {brier:.4f})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = save_dir / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {"model": model_name, "brier_score": float(brier), "n_bins": n_bins, "figure_path": str(path)}


def calibration_report_all(predictions: dict) -> pd.DataFrame:
    """calibration_report for every saved model; saves a combined CSV."""
    rows = [calibration_report(y_true, y_proba, name) for name, (y_true, y_proba) in predictions.items()]
    df = pd.DataFrame(rows)
    config.ensure_directories()
    df.to_csv(config.RESULTS_METRICS_DIR / "calibration_summary.csv", index=False)
    return df


if __name__ == "__main__":
    import pandas as pd

    from src.evaluate import load_all_predictions

    train_df = pd.read_csv(config.DATA_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(config.DATA_PROCESSED_DIR / "val.csv")

    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_val = val_df.drop(columns=[config.TARGET_COLUMN])
    y_val = val_df[config.TARGET_COLUMN]

    # Repeated CV runs over train+validation combined - test stays untouched.
    X_cv = pd.concat([X_train, X_val], ignore_index=True)
    y_cv = pd.concat([y_train, y_val], ignore_index=True)

    print("Running repeated stratified CV for baseline models (this takes a few minutes)...")
    start = time.time()
    cv_summary = repeated_cv_baselines(X_cv, y_cv, n_splits=5, n_repeats=3)
    print(f"Done in {time.time() - start:.1f}s")
    print(cv_summary)

    predictions = load_all_predictions()
    if predictions:
        # Threshold per model must match what was actually used for the
        # headline test-set result - read it back from model_comparison.csv
        # rather than re-deriving it, so the CI is for the exact same
        # threshold the dissertation reports.
        comparison = pd.read_csv(config.RESULTS_METRICS_DIR / "model_comparison.csv")
        thresholds = dict(zip(comparison["model"], comparison["threshold"]))

        print("\nBootstrapping confidence intervals on held-out test predictions...")
        bootstrap_df = bootstrap_all_models(predictions, thresholds)
        print(bootstrap_df)
        plot_bootstrap_ci(bootstrap_df, metric="f1")
        plot_bootstrap_ci(bootstrap_df, metric="pr_auc")

        print("\nCalibration assessment...")
        calibration_df = calibration_report_all(predictions)
        print(calibration_df)
    else:
        print("\nNo saved predictions found - run Phase 3/4 first to generate "
              "results/metrics/predictions/ before bootstrap CIs or calibration can run.")
