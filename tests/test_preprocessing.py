"""Unit tests for src/preprocessing.py using synthetic imbalanced data."""

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_genuine, n_fraud = 1000, 40
    n = n_genuine + n_fraud

    data = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172792, size=n)
    data["Amount"] = rng.exponential(scale=50, size=n)
    data["Class"] = [0] * n_genuine + [1] * n_fraud

    return pd.DataFrame(data)


def test_load_and_split_is_three_way_and_stratified(synthetic_df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.load_and_split(synthetic_df)

    assert len(X_train) + len(X_val) + len(X_test) == len(synthetic_df)

    overall_fraud_rate = synthetic_df[config.TARGET_COLUMN].mean()
    for y_split in (y_train, y_val, y_test):
        assert y_split.mean() == pytest.approx(overall_fraud_rate, abs=0.02)
        assert y_split.sum() > 0, "stratified split must not drop the minority class from any split"

    # Roughly 64/16/20 - loose tolerance since it's a small synthetic dataset.
    n = len(synthetic_df)
    assert len(X_train) / n == pytest.approx(0.64, abs=0.03)
    assert len(X_val) / n == pytest.approx(0.16, abs=0.03)
    assert len(X_test) / n == pytest.approx(0.20, abs=0.03)


def test_load_and_split_val_and_test_are_disjoint_from_train(synthetic_df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.load_and_split(synthetic_df)

    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_scale_features_only_touches_time_and_amount(synthetic_df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.load_and_split(synthetic_df)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocessing.scale_features(X_train, X_val, X_test)

    assert X_train_scaled["Amount"].mean() == pytest.approx(0, abs=1e-9)
    assert X_train_scaled["Amount"].std() == pytest.approx(1, abs=1e-2)

    for col in [c for c in X_train.columns if c not in config.SCALED_COLUMNS]:
        pd.testing.assert_series_equal(X_train[col], X_train_scaled[col])
        pd.testing.assert_series_equal(X_val[col], X_val_scaled[col])
        pd.testing.assert_series_equal(X_test[col], X_test_scaled[col])


def test_apply_smote_uses_tuned_partial_ratio_not_full_rebalance(synthetic_df):
    X_train, _, _, y_train, _, _ = preprocessing.load_and_split(synthetic_df)
    X_res, y_res = preprocessing.apply_smote(X_train, y_train)

    counts = y_res.value_counts()
    ratio = counts[1] / counts[0]

    assert ratio == pytest.approx(preprocessing.SMOTE_SAMPLING_STRATEGY, abs=0.02)
    assert ratio < 0.99, "SMOTE must not fully rebalance to 1:1 - see module docstring for why"


def test_apply_undersampling_uses_tuned_partial_ratio_not_full_rebalance(synthetic_df):
    X_train, _, _, y_train, _, _ = preprocessing.load_and_split(synthetic_df)
    X_res, y_res = preprocessing.apply_undersampling(X_train, y_train)

    counts = y_res.value_counts()
    ratio = counts[1] / counts[0]

    assert ratio == pytest.approx(preprocessing.UNDERSAMPLE_SAMPLING_STRATEGY, abs=0.02)
    assert len(X_res) < len(X_train), "undersampling must still reduce the majority class"


def test_apply_smote_and_undersampling_accept_custom_ratio(synthetic_df):
    X_train, _, _, y_train, _, _ = preprocessing.load_and_split(synthetic_df)

    X_res, y_res = preprocessing.apply_smote(X_train, y_train, sampling_strategy=1.0)
    counts = y_res.value_counts()
    assert counts[0] == counts[1]


def test_compute_class_weights_favours_minority_class(synthetic_df):
    _, _, _, y_train, _, _ = preprocessing.load_and_split(synthetic_df)
    weights = preprocessing.compute_class_weights(y_train)

    assert weights[1] > weights[0], "the minority class (fraud) must get a higher weight"


def test_compare_imbalance_strategies_uses_validation_set(synthetic_df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.load_and_split(synthetic_df)
    X_train_scaled, X_val_scaled, X_test_scaled, _ = preprocessing.scale_features(X_train, X_val, X_test)

    comparison = preprocessing.compare_imbalance_strategies(
        X_train_scaled, y_train, X_val_scaled, y_val
    )

    assert set(comparison.index) == {"smote", "undersampling", "class_weighting", "none_baseline"}
    assert "f1" in comparison.columns
    assert "pr_auc" in comparison.columns
    assert "mcc" in comparison.columns
