"""Unit tests for src/robust_evaluation.py using synthetic imbalanced data.

n_splits/n_repeats/n_bootstrap are kept small here purely for test speed - the
real run (src/robust_evaluation.py's __main__ block) uses larger values.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, robust_evaluation


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_genuine, n_fraud = 400, 40
    n = n_genuine + n_fraud

    X = pd.DataFrame({f"V{i}": rng.normal(size=n) for i in range(1, 6)})
    X["Time"] = rng.uniform(0, 172792, size=n)
    X["Amount"] = rng.exponential(scale=50, size=n)
    y = pd.Series([0] * n_genuine + [1] * n_fraud, name="Class")

    return X, y


def test_repeated_cv_baselines_returns_one_row_per_model(tmp_path, monkeypatch, synthetic_data):
    # repeated_cv_baselines() always writes its CSVs via config.RESULTS_METRICS_DIR
    # (there's no way to opt out) - every test that calls it MUST monkeypatch this,
    # or it silently overwrites the real project's results/metrics/ output with
    # whatever tiny synthetic data the test happens to use. (This bit a real run:
    # see CHANGELOG.md for how a version of this test without the monkeypatch
    # clobbered the genuine repeated-CV results.)
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    X, y = synthetic_data
    summary = robust_evaluation.repeated_cv_baselines(X, y, n_splits=2, n_repeats=1)

    assert set(summary["model"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert "f1_mean" in summary.columns
    assert "f1_std" in summary.columns


def test_repeated_cv_baselines_saves_per_fold_and_summary_csvs(tmp_path, monkeypatch, synthetic_data):
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    X, y = synthetic_data

    robust_evaluation.repeated_cv_baselines(X, y, n_splits=2, n_repeats=1)

    assert (tmp_path / "baseline_repeated_cv_per_fold.csv").exists()
    assert (tmp_path / "baseline_repeated_cv_summary.csv").exists()


def test_bootstrap_metric_ci_contains_point_estimate_within_interval():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = np.clip(y_true + rng.normal(scale=0.3, size=200), 0, 1)

    result = robust_evaluation.bootstrap_metric_ci(y_true, y_proba, threshold=0.5, metric="f1", n_bootstrap=100)

    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
    assert result["ci_width"] >= 0


def test_bootstrap_metric_ci_handles_rare_positive_class_without_error():
    """At ~0.17% prevalence a bootstrap resample can draw zero fraud cases -
    must redraw rather than crash or return an undefined metric."""
    rng = np.random.default_rng(0)
    n = 500
    y_true = np.zeros(n, dtype=int)
    y_true[:2] = 1  # only 2 positives, matching real-world-scale rarity
    y_proba = rng.random(size=n)

    result = robust_evaluation.bootstrap_metric_ci(y_true, y_proba, threshold=0.5, metric="pr_auc", n_bootstrap=50)

    assert not np.isnan(result["point_estimate"])
    assert not np.isnan(result["ci_lower"])


def test_bootstrap_all_models_saves_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    predictions = {
        "ModelA": (y_true, rng.random(size=100)),
        "ModelB": (y_true, rng.random(size=100)),
    }
    thresholds = {"ModelA": 0.5, "ModelB": 0.6}

    df = robust_evaluation.bootstrap_all_models(predictions, thresholds, metrics=("f1",))

    assert (tmp_path / "bootstrap_confidence_intervals.csv").exists()
    assert len(df) == 2


def test_plot_bootstrap_ci_saves_file(tmp_path):
    df = pd.DataFrame([
        {"metric": "f1", "model": "ModelA", "point_estimate": 0.7, "ci_lower": 0.6, "ci_upper": 0.8},
        {"metric": "f1", "model": "ModelB", "point_estimate": 0.5, "ci_lower": 0.4, "ci_upper": 0.6},
    ])
    path = robust_evaluation.plot_bootstrap_ci(df, metric="f1", save_path=tmp_path / "ci.png")
    assert path.exists()


def test_calibration_report_returns_brier_score_and_saves_figure(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = np.clip(y_true * 0.7 + rng.random(size=200) * 0.3, 0, 1)

    result = robust_evaluation.calibration_report(y_true, y_proba, "TestModel", save_dir=tmp_path)

    assert 0.0 <= result["brier_score"] <= 1.0
    assert result["model"] == "TestModel"
    from pathlib import Path
    assert Path(result["figure_path"]).exists()


def test_calibration_report_all_saves_summary_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_FIGURES_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    predictions = {"ModelA": (y_true, rng.random(size=100))}

    df = robust_evaluation.calibration_report_all(predictions)

    assert (tmp_path / "calibration_summary.csv").exists()
    assert len(df) == 1
