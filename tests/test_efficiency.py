"""Unit tests for src/efficiency.py using a trivial sklearn model and synthetic data."""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import efficiency


@pytest.fixture
def trained_model_and_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"V{i}" for i in range(5)])
    y = (X["V0"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return model, X, y


def test_measure_training_time_returns_result_and_positive_elapsed():
    def slow_train():
        time.sleep(0.01)
        return "trained"

    result, elapsed = efficiency.measure_training_time(slow_train)
    assert result == "trained"
    assert elapsed > 0


def test_measure_inference_time_single_is_positive(trained_model_and_data):
    model, X, _ = trained_model_and_data
    single_row = X.iloc[[0]]
    avg_time = efficiency.measure_inference_time_single(model.predict, single_row, n_repeats=5)
    assert avg_time > 0


def test_measure_inference_time_batch_is_positive(trained_model_and_data):
    model, X, _ = trained_model_and_data
    elapsed = efficiency.measure_inference_time_batch(model.predict, X)
    assert elapsed > 0


def test_get_sklearn_model_disk_size_is_positive(tmp_path, trained_model_and_data):
    model, _, _ = trained_model_and_data
    size = efficiency.get_sklearn_model_disk_size(model, tmp_path / "model.joblib")
    assert size > 0


def test_benchmark_model_returns_expected_keys(trained_model_and_data):
    model, X, _ = trained_model_and_data
    record = efficiency.benchmark_model(
        "LogReg",
        model.predict,
        X.iloc[[0]],
        X,
        train_time_sec=0.5,
        model_size_bytes=2048,
        param_count=6,
        n_repeats=5,
    )
    assert record["model"] == "LogReg"
    assert record["batch_size"] == len(X)
    assert record["model_size_kb"] == 2.0
    assert record["param_count"] == 6


def test_build_efficiency_table_saves_csv(tmp_path):
    records = [
        {"model": "A", "train_time_sec": 1.0},
        {"model": "B", "train_time_sec": 2.0},
    ]
    save_path = tmp_path / "efficiency.csv"
    df = efficiency.build_efficiency_table(records, save_path=save_path)
    assert save_path.exists()
    assert len(df) == 2
