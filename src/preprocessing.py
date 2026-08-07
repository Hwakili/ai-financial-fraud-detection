"""Phase 2: preprocessing and class imbalance handling.

Design decisions (documented here because they belong directly in Chapter 4.2):

- Split BEFORE any resampling. Resampling the whole dataset first and splitting
  afterward would leak synthetic/duplicated minority-class information between
  splits, inflating held-out performance artificially.
- stratify=y on every split. Without it, a 20% test split of a 0.172%-fraud
  dataset risks a split with only a handful of fraud cases, making metrics
  unstable.
- Three-way train/validation/test split (64/16/20), not just train/test. The
  validation split exists so classification-threshold tuning (see
  evaluate.select_threshold) and RXT's early stopping have real held-out data
  to work with, without ever touching the test set. Tuning a threshold on the
  same data used for final reporting is a soft form of test-set leakage - the
  three-way split rules that out structurally rather than relying on
  discipline alone.
- The validation and test sets are NEVER resampled. Only the training set is
  rebalanced; val/test must reflect real-world imbalance so that
  precision/recall/F1 on them are honest estimates of real deployment
  performance.
- StandardScaler is fit on Amount and Time only, and fit on the training set
  only (then applied to validation and test) - V1-V28 are already PCA-scaled
  upstream.
"""

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluate import compute_metrics

# SMOTE/undersampling ratios, as a design decision worth stating explicitly:
#
# An earlier version of this pipeline used the imblearn defaults (a full 1:1
# rebalance - sampling_strategy=1.0 for both SMOTE and undersampling). Run for
# real against the full dataset, that full rebalance produced a severe
# precision collapse: SMOTE/undersampling/class-weighting all pushed recall to
# ~92% but cut precision to 4-6% (F1 0.07-0.11), while an *unweighted* model
# with no resampling at all reached F1=0.72. Forcing the minority class up to
# full parity with the majority overcorrects - the model ends up trained on a
# class balance nothing like what it will see at inference time, and firing
# on far too many genuine transactions as a result.
#
# SMOTE at sampling_strategy=0.20 (raise fraud to ~20% of the majority count,
# not 100%) and undersampling at sampling_strategy=0.10 (raise fraud to ~10%
# of the reduced majority count) are deliberately less aggressive: enough to
# meaningfully help the minority class without moving the training
# distribution as far from reality. This must still be validated empirically
# per run (see results/metrics/phase2_imbalance_comparison.csv) rather than
# assumed - the point of Phase 2 is exactly to check whether this compromise
# actually beats both the full-rebalance and no-resampling extremes on F1/MCC,
# not to declare it correct by construction.
SMOTE_SAMPLING_STRATEGY = 0.20
UNDERSAMPLE_SAMPLING_STRATEGY = 0.10


def load_and_split(df: pd.DataFrame):
    """Stratified 64/16/20 train/validation/test split, performed before any
    scaling or resampling.

    Implemented as two successive two-way splits (carve out test first, then
    split the remainder into train/validation) rather than a single three-way
    call, since scikit-learn's train_test_split only supports two-way splits.
    """
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )

    val_share_of_remainder = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_share_of_remainder,
        stratify=y_train_val,
        random_state=config.RANDOM_SEED,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """Fit StandardScaler on Time/Amount in the training set only, apply to all three splits."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[config.SCALED_COLUMNS] = scaler.fit_transform(X_train[config.SCALED_COLUMNS])
    X_val_scaled[config.SCALED_COLUMNS] = scaler.transform(X_val[config.SCALED_COLUMNS])
    X_test_scaled[config.SCALED_COLUMNS] = scaler.transform(X_test[config.SCALED_COLUMNS])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, sampling_strategy=SMOTE_SAMPLING_STRATEGY):
    """Generate synthetic minority-class samples via interpolation between neighbours.

    sampling_strategy=0.20 by default (not 1.0) - see the module-level comment
    above for why a partial rather than full rebalance was chosen.
    """
    smote = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def apply_undersampling(X_train: pd.DataFrame, y_train: pd.Series, sampling_strategy=UNDERSAMPLE_SAMPLING_STRATEGY):
    """Reduce the majority class instead of growing the minority class.

    sampling_strategy=0.10 by default (not 1.0) - see the module-level comment
    above for why a partial rather than full rebalance was chosen.
    """
    rus = RandomUnderSampler(random_state=config.RANDOM_SEED, sampling_strategy=sampling_strategy)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, y_res


def compute_class_weights(y_train: pd.Series) -> dict:
    """Balanced class weights - no synthetic data, just a heavier loss penalty on fraud."""
    counts = y_train.value_counts()
    n = len(y_train)
    return {cls: n / (len(counts) * count) for cls, count in counts.items()}


def compare_imbalance_strategies(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """Train a quick Logistic Regression under each imbalance strategy and compare
    on the VALIDATION set - never the test set, which stays untouched until
    Phase 5's final reporting.

    This is a fast diagnostic (Phase 2), not the final baseline suite (Phase 3) -
    its only purpose is to decide which imbalance strategy carries forward.
    """
    results = []

    strategies = {
        "smote": apply_smote(X_train, y_train),
        "undersampling": apply_undersampling(X_train, y_train),
        "class_weighting": (X_train, y_train),
        "none_baseline": (X_train, y_train),
    }

    for name, (X_res, y_res) in strategies.items():
        start = time.time()
        if name == "class_weighting":
            model = LogisticRegression(
                class_weight=compute_class_weights(y_train), max_iter=1000
            )
        else:
            model = LogisticRegression(max_iter=1000)
        model.fit(X_res, y_res)
        train_time = time.time() - start

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics["strategy"] = name
        metrics["train_time_sec"] = round(train_time, 4)
        results.append(metrics)

    df_results = pd.DataFrame(results).set_index("strategy")
    column_order = [
        "accuracy", "precision", "recall", "f1", "auc_roc", "pr_auc", "mcc", "train_time_sec"
    ]
    return df_results[column_order]


def run_preprocessing_pipeline(df: pd.DataFrame) -> dict:
    """Full Phase 2 pipeline: split, scale, compare imbalance strategies, persist outputs."""
    config.ensure_directories()

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(df)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    comparison = compare_imbalance_strategies(X_train_scaled, y_train, X_val_scaled, y_val)
    comparison.to_csv(config.RESULTS_METRICS_DIR / "phase2_imbalance_comparison.csv")

    X_train_scaled.assign(**{config.TARGET_COLUMN: y_train.values}).to_csv(
        config.DATA_PROCESSED_DIR / "train.csv", index=False
    )
    X_val_scaled.assign(**{config.TARGET_COLUMN: y_val.values}).to_csv(
        config.DATA_PROCESSED_DIR / "val.csv", index=False
    )
    X_test_scaled.assign(**{config.TARGET_COLUMN: y_test.values}).to_csv(
        config.DATA_PROCESSED_DIR / "test.csv", index=False
    )

    return {
        "train_shape": X_train_scaled.shape,
        "val_shape": X_val_scaled.shape,
        "test_shape": X_test_scaled.shape,
        "comparison": comparison,
    }


if __name__ == "__main__":
    from src.eda import load_data

    dataset = load_data()
    result = run_preprocessing_pipeline(dataset)
    print(result["comparison"])
