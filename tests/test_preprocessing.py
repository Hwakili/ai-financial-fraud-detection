"""Unit tests for src/preprocessing.py using synthetic imbalanced data."""

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_genuine, n_fraud = 500, 20
    n = n_genuine + n_fraud

    data = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172792, size=n)
    data["Amount"] = rng.exponential(scale=50, size=n)
    data["Class"] = [0] * n_genuine + [1] * n_fraud

    return pd.DataFrame(data)


def test_load_and_split_is_stratified(synthetic_df):
    X_train, X_test, y_train, y_test = preprocessing.load_and_split(synthetic_df)

    assert len(X_train) + len(X_test) == len(synthetic_df)

    train_fraud_rate = y_train.mean()
    test_fraud_rate = y_test.mean()
    overall_fraud_rate = synthetic_df[config.TARGET_COLUMN].mean()

    assert train_fraud_rate == pytest.approx(overall_fraud_rate, abs=0.02)
    assert test_fraud_rate == pytest.approx(overall_fraud_rate, abs=0.02)
    assert y_test.sum() > 0, "stratified split must not drop the minority class from the test set"


def test_scale_features_only_touches_time_and_amount(synthetic_df):
    X_train, X_test, y_train, y_test = preprocessing.load_and_split(synthetic_df)
    X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(X_train, X_test)

    assert X_train_scaled["Amount"].mean() == pytest.approx(0, abs=1e-9)
    assert X_train_scaled["Amount"].std() == pytest.approx(1, abs=1e-2)

    for col in [c for c in X_train.columns if c not in config.SCALED_COLUMNS]:
        pd.testing.assert_series_equal(X_train[col], X_train_scaled[col])


def test_apply_smote_balances_classes(synthetic_df):
    X_train, _, y_train, _ = preprocessing.load_and_split(synthetic_df)
    X_res, y_res = preprocessing.apply_smote(X_train, y_train)

    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) == len(y_res)


def test_apply_undersampling_balances_classes(synthetic_df):
    X_train, _, y_train, _ = preprocessing.load_and_split(synthetic_df)
    X_res, y_res = preprocessing.apply_undersampling(X_train, y_train)

    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) < len(X_train), "undersampling must reduce the majority class"


def test_compute_class_weights_favours_minority_class(synthetic_df):
    _, _, y_train, _ = preprocessing.load_and_split(synthetic_df)
    weights = preprocessing.compute_class_weights(y_train)

    assert weights[1] > weights[0], "the minority class (fraud) must get a higher weight"


def test_compare_imbalance_strategies_returns_all_strategies(synthetic_df):
    X_train, X_test, y_train, y_test = preprocessing.load_and_split(synthetic_df)
    X_train_scaled, X_test_scaled, _ = preprocessing.scale_features(X_train, X_test)

    comparison = preprocessing.compare_imbalance_strategies(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    assert set(comparison.index) == {"smote", "undersampling", "class_weighting", "none_baseline"}
    assert "f1" in comparison.columns
    assert "mcc" in comparison.columns
