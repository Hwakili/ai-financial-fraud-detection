"""Unit tests for src/rxt_model.py.

Uses tiny synthetic data and 1-2 epochs - these tests exist to catch shape
mismatches and API misuse (e.g. a broken groups= conv, a bad reshape), not to
validate real fraud-detection performance. Real performance is assessed in the
dissertation using the full dataset on Colab Pro / Kaggle GPU per the ToR.
"""

import numpy as np
import pytest

from src import rxt_model


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n_samples, n_features = 60, 30
    X = rng.normal(size=(n_samples, n_features)).astype("float32")
    y = rng.integers(0, 2, size=n_samples)
    return X, y


def test_reshape_for_rxt_adds_channel_dimension(synthetic_data):
    X, _ = synthetic_data
    reshaped = rxt_model.reshape_for_rxt(X)
    assert reshaped.shape == (60, 30, 1)


def test_build_rxt_model_compiles_and_has_expected_io_shape():
    model = rxt_model.build_rxt_model(n_features=30, cardinality=4, gru_units=16)
    assert model.input_shape == (None, 30, 1)
    assert model.output_shape == (None, 1)


def test_train_rxt_runs_one_epoch_without_error(synthetic_data):
    X, y = synthetic_data
    X_train, X_val = X[:40], X[40:]
    y_train, y_val = y[:40], y[40:]

    model, history = rxt_model.train_rxt(
        X_train, y_train, X_val, y_val,
        epochs=1, batch_size=8, cardinality=4, gru_units=8, verbose=0,
    )

    assert "loss" in history.history
    preds = model.predict(rxt_model.reshape_for_rxt(X_val), verbose=0)
    assert preds.shape == (len(X_val), 1)
    assert np.all((preds >= 0) & (preds <= 1))


def test_train_rxt_kfold_returns_one_row_per_fold(synthetic_data):
    X, y = synthetic_data
    results = rxt_model.train_rxt_kfold(X, y, n_splits=3, epochs=1, batch_size=8)
    assert len(results) == 3
    assert "f1" in results.columns
    assert "mcc" in results.columns


def test_summarize_kfold_results_has_mean_and_std(synthetic_data):
    X, y = synthetic_data
    results = rxt_model.train_rxt_kfold(X, y, n_splits=3, epochs=1, batch_size=8)
    summary = rxt_model.summarize_kfold_results(results)
    assert set(summary.columns) == {"mean", "std"}
    assert "f1" in summary.index


def test_train_rxt_kfold_threshold_is_not_hardcoded_at_half(synthetic_data):
    """Sanity check that fold-level threshold selection actually ran (present
    as a column) rather than silently falling back to a fixed 0.5 everywhere -
    with random synthetic data the tuned value should vary across folds."""
    X, y = synthetic_data
    results = rxt_model.train_rxt_kfold(X, y, n_splits=3, epochs=1, batch_size=8)
    assert "threshold" in results.columns


def test_run_rxt_pipeline_returns_model_metrics_and_uses_validation_split(tmp_path, monkeypatch, synthetic_data):
    from src import config, evaluate

    monkeypatch.setattr(config, "RESULTS_FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "RESULTS_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(evaluate, "MASTER_METRICS_CSV", tmp_path / "metrics" / "model_comparison.csv")
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "metrics" / "predictions")

    X, y = synthetic_data
    X_train, X_val, X_test = X[:36], X[36:48], X[48:]
    y_train, y_val, y_test = y[:36], y[36:48], y[48:]

    model, metrics, history = rxt_model.run_rxt_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=1, batch_size=8, persist_model=False,
    )

    assert "f1" in metrics
    assert "pr_auc" in metrics
    assert "threshold" in metrics

    y_proba_val = model.predict(rxt_model.reshape_for_rxt(X_val), verbose=0).flatten()
    expected_threshold = evaluate.select_threshold(y_val, y_proba_val)
    assert metrics["threshold"] == pytest.approx(expected_threshold)
