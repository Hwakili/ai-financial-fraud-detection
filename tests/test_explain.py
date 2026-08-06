"""Unit tests for src/explain.py using small synthetic data and a tiny tree model
(fast, deterministic - real SHAP/LIME depth is assessed in the dissertation
notebooks against the real dataset and trained models)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import explain


@pytest.fixture
def synthetic_setup():
    rng = np.random.default_rng(0)
    n_samples, n_features = 80, 6
    feature_names = [f"V{i}" for i in range(n_features)]

    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + rng.normal(scale=0.1, size=n_samples) > 0).astype(int)

    X_df = pd.DataFrame(X, columns=feature_names)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_df, y)

    return model, X_df, y, feature_names


def test_shap_tree_explain_saves_plot_and_returns_values(tmp_path, synthetic_setup):
    model, X_df, y, feature_names = synthetic_setup
    values, path = explain.shap_tree_explain(
        model, X_df.iloc[:20], feature_names, "TestRF", save_dir=tmp_path
    )
    assert path.exists()
    assert values.shape[0] == 20


def test_select_instructive_cases_finds_each_category():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 1, 0, 0]
    cases = explain.select_instructive_cases(y_true, y_pred, None)

    assert cases["true_positive_fraud_caught"] == 0
    assert cases["false_negative_fraud_missed"] == 1
    assert cases["false_positive_genuine_flagged"] == 2


def test_select_instructive_cases_handles_missing_category():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    cases = explain.select_instructive_cases(y_true, y_pred, None)
    assert cases == {}


def test_lime_explain_instance_saves_plot(tmp_path, synthetic_setup):
    model, X_df, y, feature_names = synthetic_setup
    instance = X_df.iloc[0].values
    save_path = tmp_path / "lime_test.png"

    explanation = explain.lime_explain_instance(
        X_df.values, feature_names, model.predict_proba, instance, save_path
    )

    assert save_path.exists()
    assert explanation is not None


def test_run_lime_on_instructive_cases_saves_expected_files(tmp_path, synthetic_setup):
    model, X_df, y, feature_names = synthetic_setup
    y_pred = model.predict(X_df)

    explanations = explain.run_lime_on_instructive_cases(
        X_df.values, X_df, y, y_pred, model.predict_proba, feature_names, "TestRF", save_dir=tmp_path
    )

    assert len(explanations) > 0
    saved_files = list(tmp_path.glob("lime_testrf_*.png"))
    assert len(saved_files) == len(explanations)
