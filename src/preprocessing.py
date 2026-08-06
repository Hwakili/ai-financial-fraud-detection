"""Phase 2: preprocessing and class imbalance handling.

Design decisions (documented here because they belong directly in Chapter 4.2):

- Split BEFORE any resampling. Resampling the whole dataset first and splitting
  afterward would leak synthetic/duplicated minority-class information between
  train and test, inflating test performance artificially.
- stratify=y on the split. Without it, a 20% test split of a 0.172%-fraud dataset
  risks a test set with only a handful of fraud cases, making metrics unstable.
- The test set is NEVER resampled. It must reflect real-world imbalance so that
  precision/recall/F1 on it are honest estimates of real deployment performance.
- StandardScaler is fit on Amount and Time only, and fit on the training set only
  (then applied to both train and test) - V1-V28 are already PCA-scaled upstream.
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


def load_and_split(df: pd.DataFrame):
    """Stratified train/test split, performed before any scaling or resampling."""
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit StandardScaler on Time/Amount in the training set only, apply to both."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[config.SCALED_COLUMNS] = scaler.fit_transform(X_train[config.SCALED_COLUMNS])
    X_test_scaled[config.SCALED_COLUMNS] = scaler.transform(X_test[config.SCALED_COLUMNS])

    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """Generate synthetic minority-class samples via interpolation between neighbours."""
    smote = SMOTE(random_state=config.RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def apply_undersampling(X_train: pd.DataFrame, y_train: pd.Series):
    """Reduce the majority class instead of growing the minority class."""
    rus = RandomUnderSampler(random_state=config.RANDOM_SEED)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, y_res


def compute_class_weights(y_train: pd.Series) -> dict:
    """Balanced class weights - no synthetic data, just a heavier loss penalty on fraud."""
    counts = y_train.value_counts()
    n = len(y_train)
    return {cls: n / (len(counts) * count) for cls, count in counts.items()}


def compare_imbalance_strategies(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Train a quick Logistic Regression under each imbalance strategy and compare.

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

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["strategy"] = name
        metrics["train_time_sec"] = round(train_time, 4)
        results.append(metrics)

    df_results = pd.DataFrame(results).set_index("strategy")
    column_order = [
        "accuracy", "precision", "recall", "f1", "auc_roc", "mcc", "train_time_sec"
    ]
    return df_results[column_order]


def run_preprocessing_pipeline(df: pd.DataFrame) -> dict:
    """Full Phase 2 pipeline: split, scale, compare imbalance strategies, persist outputs."""
    config.ensure_directories()

    X_train, X_test, y_train, y_test = load_and_split(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    comparison = compare_imbalance_strategies(X_train_scaled, y_train, X_test_scaled, y_test)
    comparison.to_csv(config.RESULTS_METRICS_DIR / "phase2_imbalance_comparison.csv")

    X_train_scaled.assign(**{config.TARGET_COLUMN: y_train.values}).to_csv(
        config.DATA_PROCESSED_DIR / "train.csv", index=False
    )
    X_test_scaled.assign(**{config.TARGET_COLUMN: y_test.values}).to_csv(
        config.DATA_PROCESSED_DIR / "test.csv", index=False
    )

    return {
        "train_shape": X_train_scaled.shape,
        "test_shape": X_test_scaled.shape,
        "comparison": comparison,
    }


if __name__ == "__main__":
    from src.eda import load_data

    dataset = load_data()
    result = run_preprocessing_pipeline(dataset)
    print(result["comparison"])
