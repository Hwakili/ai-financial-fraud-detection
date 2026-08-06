"""Unit tests for src/eda.py using a small synthetic dataset shaped like creditcard.csv.

Uses synthetic data (not the real download) so these tests run without needing
the Kaggle dataset present - fast, deterministic, and CI-friendly.
"""

import numpy as np
import pandas as pd
import pytest

from src import eda


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(42)
    n_genuine, n_fraud = 200, 5
    n = n_genuine + n_fraud

    data = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172792, size=n)
    data["Amount"] = rng.exponential(scale=50, size=n)
    data["Class"] = [0] * n_genuine + [1] * n_fraud

    return pd.DataFrame(data)


def test_summarize_dataset_reports_correct_shape_and_fraud_rate(synthetic_df):
    summary = eda.summarize_dataset(synthetic_df)
    assert summary["shape"] == (205, 31)
    assert summary["missing_values_total"] == 0
    assert summary["class_counts"][0] == 200
    assert summary["class_counts"][1] == 5
    assert summary["fraud_rate_pct"] == pytest.approx(5 / 205 * 100, abs=0.01)


def test_plot_class_distribution_saves_file(tmp_path, synthetic_df):
    out_path = tmp_path / "class_dist.png"
    result_path = eda.plot_class_distribution(synthetic_df, save_path=out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_amount_time_by_class_saves_both_files(tmp_path, synthetic_df):
    paths = eda.plot_amount_time_by_class(synthetic_df, save_dir=tmp_path)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()


def test_plot_correlation_heatmap_saves_file(tmp_path, synthetic_df):
    out_path = tmp_path / "corr.png"
    result_path = eda.plot_correlation_heatmap(synthetic_df, save_path=out_path)
    assert result_path == out_path
    assert out_path.exists()
