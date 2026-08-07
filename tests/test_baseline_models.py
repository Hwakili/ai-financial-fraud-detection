"""Unit tests for src/baseline_models.py using synthetic imbalanced data."""

import numpy as np
import pandas as pd
import pytest

from src import baseline_models, config, preprocessing


@pytest.fixture
def synthetic_split():
    rng = np.random.default_rng(42)
    n_genuine, n_fraud = 600, 30
    n = n_genuine + n_fraud

    data = {f"V{i}": rng.normal(size=n) for i in range(1, 6)}
    data["Time"] = rng.uniform(0, 172792, size=n)
    data["Amount"] = rng.exponential(scale=50, size=n)
    data["Class"] = [0] * n_genuine + [1] * n_fraud
    df = pd.DataFrame(data)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.load_and_split(df)
    X_train_scaled, X_val_scaled, X_test_scaled, _ = preprocessing.scale_features(X_train, X_val, X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


def test_train_logistic_regression_fits_and_predicts(synthetic_split):
    X_train, X_val, X_test, y_train, y_val, y_test = synthetic_split
    model = baseline_models.train_logistic_regression(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_random_forest_fits_and_predicts(synthetic_split):
    X_train, X_val, X_test, y_train, y_val, y_test = synthetic_split
    model = baseline_models.train_random_forest(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_xgboost_fits_and_predicts(synthetic_split):
    X_train, X_val, X_test, y_train, y_val, y_test = synthetic_split
    model = baseline_models.train_xgboost(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_all_baselines_returns_three_models(synthetic_split):
    X_train, X_val, X_test, y_train, y_val, y_test = synthetic_split
    models = baseline_models.train_all_baselines(X_train, y_train)
    assert set(models.keys()) == {"Logistic Regression", "Random Forest", "XGBoost"}


def test_run_baseline_pipeline_returns_models_and_metrics(tmp_path, monkeypatch, synthetic_split):
    monkeypatch.setattr(config, "RESULTS_FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "RESULTS_MODELS_DIR", tmp_path / "models")

    from src import evaluate
    monkeypatch.setattr(evaluate, "MASTER_METRICS_CSV", tmp_path / "metrics" / "model_comparison.csv")
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "metrics" / "predictions")

    X_train, X_val, X_test, y_train, y_val, y_test = synthetic_split
    models, results = baseline_models.run_baseline_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)

    assert set(models.keys()) == set(results.keys())
    for name, metrics in results.items():
        assert "f1" in metrics
        assert "mcc" in metrics
        assert "pr_auc" in metrics
        assert "threshold" in metrics


def test_run_baseline_pipeline_threshold_is_tuned_on_validation_not_test(tmp_path, monkeypatch, synthetic_split):
    """The threshold each model ends up using must come from select_threshold()
    applied to validation predictions - not a hardcoded 0.5 - and must be the
    same value regardless of what the test set looks like."""
    monkeypatch.setattr(config, "RESULTS_FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "RESULTS_MODELS_DIR", tmp_path / "models")

    from src import evaluate
    monkeypatch.setattr(evaluate, "MASTER_METRICS_CSV", tmp_path / "metrics" / "model_comparison.csv")
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "metrics" / "predictions")

    X_train, X_val, X_test, y_train, y_val, y_test = synthetic_split
    models, results = baseline_models.run_baseline_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)

    for name, model in models.items():
        y_proba_val = model.predict_proba(X_val)[:, 1]
        expected_threshold = evaluate.select_threshold(y_val, y_proba_val)
        assert results[name]["threshold"] == pytest.approx(expected_threshold)
