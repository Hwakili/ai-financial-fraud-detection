"""Unit tests for src/comparative_evaluation.py."""

import numpy as np
import pandas as pd
import pytest

from src import comparative_evaluation, evaluate


@pytest.fixture
def populated_results(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "MASTER_METRICS_CSV", tmp_path / "model_comparison.csv")
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(comparative_evaluation, "MASTER_METRICS_CSV", tmp_path / "model_comparison.csv")

    from src import config
    monkeypatch.setattr(config, "RESULTS_FIGURES_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)

    for name in ["Logistic Regression", "Random Forest", "RXT (ResNeXt-GRU)"]:
        y_proba = rng.random(size=100)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = evaluate.compute_metrics(y_true, y_pred, y_proba)
        evaluate.append_metrics_to_master(metrics, name)
        evaluate.save_predictions(name, y_true, y_proba)

    return tmp_path


def test_run_comparative_evaluation_raises_without_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "MASTER_METRICS_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "missing_predictions")
    monkeypatch.setattr(comparative_evaluation, "MASTER_METRICS_CSV", tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError):
        comparative_evaluation.run_comparative_evaluation()


def test_run_comparative_evaluation_produces_expected_outputs(populated_results):
    result = comparative_evaluation.run_comparative_evaluation()

    assert len(result["metrics_table"]) == 3
    assert result["roc_curve_path"].exists()
    assert result["pr_curve_path"].exists()
    assert result["grouped_bar_path"].exists()
    assert list(result["ranked_by_f1"]["f1"]) == sorted(result["ranked_by_f1"]["f1"], reverse=True)
