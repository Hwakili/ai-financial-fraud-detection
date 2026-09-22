"""Unit tests for src/operational_evaluation.py using synthetic imbalanced data."""

import numpy as np
import pandas as pd
import pytest

from src import config, operational_evaluation


def test_precision_recall_at_k_counts_true_positives_correctly():
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    # Highest-probability transactions are indices 4, 2, 7 (all fraud) then genuine ones.
    y_proba = np.array([0.1, 0.2, 0.8, 0.05, 0.95, 0.3, 0.15, 0.7, 0.4, 0.25])

    result = operational_evaluation.precision_recall_at_k(y_true, y_proba, k_values=(3,))
    row = result.iloc[0]

    assert row["k_used"] == 3
    assert row["true_positives_in_top_k"] == 3  # all three fraud cases are the top-3 scored
    assert row["precision_at_k"] == pytest.approx(1.0)
    assert row["recall_at_k"] == pytest.approx(1.0)


def test_precision_recall_at_k_caps_k_to_dataset_size():
    y_true = np.array([0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2])

    result = operational_evaluation.precision_recall_at_k(y_true, y_proba, k_values=(50,))

    assert result.iloc[0]["k_used"] == 3
    assert result.iloc[0]["k"] == 50


def test_precision_recall_at_k_all_models_saves_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    predictions = {"ModelA": (y_true, rng.random(size=50))}

    df = operational_evaluation.precision_recall_at_k_all_models(predictions, k_values=(10,))

    assert (tmp_path / "precision_at_k.csv").exists()
    assert len(df) == 1


def test_plot_precision_at_k_saves_file(tmp_path):
    df = pd.DataFrame([
        {"model": "ModelA", "k": 50, "precision_at_k": 0.8},
        {"model": "ModelA", "k": 100, "precision_at_k": 0.6},
    ])
    path = operational_evaluation.plot_precision_at_k(df, save_path=tmp_path / "pak.png")
    assert path.exists()


def test_expected_cost_basic_correctness():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]  # one FP (index 1), one FN (index 2)

    result = operational_evaluation.expected_cost(y_true, y_pred, cost_fp=5.0, cost_fn=100.0)

    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["total_assumed_cost"] == pytest.approx(105.0)
    assert result["mean_assumed_cost_per_transaction"] == pytest.approx(105.0 / 4)


def test_cost_sensitivity_sweep_returns_one_row_per_cost_value():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = np.clip(y_true * 0.6 + rng.random(size=200) * 0.4, 0, 1)

    result = operational_evaluation.cost_sensitivity_sweep(
        y_true, y_proba, cost_fn_values=(50, 100), cost_fp=5.0, n_thresholds=20
    )

    assert len(result) == 2
    assert set(result["cost_fn_assumed"]) == {50, 100}
    assert "cost_optimal_threshold" in result.columns


def test_higher_assumed_fn_cost_favours_lower_threshold_on_average():
    """As missing fraud is assumed more costly, the cost-minimising threshold
    should trend lower (favouring recall) - not an iron law for every dataset,
    but true for well-separated synthetic data, and a reasonable sanity check
    on the sweep's core logic."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=300)
    y_proba = np.clip(y_true * 0.6 + rng.normal(scale=0.15, size=300), 0, 1)

    result = operational_evaluation.cost_sensitivity_sweep(
        y_true, y_proba, cost_fn_values=(20, 1000), cost_fp=5.0, n_thresholds=100
    )
    low_cost_row = result[result["cost_fn_assumed"] == 20].iloc[0]
    high_cost_row = result[result["cost_fn_assumed"] == 1000].iloc[0]

    assert high_cost_row["cost_optimal_threshold"] <= low_cost_row["cost_optimal_threshold"]


def test_cost_sensitivity_all_models_saves_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    predictions = {"ModelA": (y_true, rng.random(size=100))}

    df = operational_evaluation.cost_sensitivity_all_models(predictions, cost_fn_values=(50,), cost_fp=5.0)

    assert (tmp_path / "cost_sensitivity_analysis.csv").exists()
    assert len(df) == 1


@pytest.fixture
def synthetic_raw_df():
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_genuine, n_fraud = 400, 40
    n = n_genuine + n_fraud

    data = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172792, size=n)
    data["Amount"] = rng.exponential(scale=50, size=n)
    data["Class"] = [0] * n_genuine + [1] * n_fraud
    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)  # shuffle so it isn't pre-sorted


def test_chronological_split_is_ordered_by_time_and_sums_to_full_dataset(synthetic_raw_df):
    train_df, val_df, test_df = operational_evaluation.chronological_split(synthetic_raw_df)

    assert len(train_df) + len(val_df) + len(test_df) == len(synthetic_raw_df)
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()


def test_run_chronological_robustness_test_returns_all_baselines(tmp_path, monkeypatch, synthetic_raw_df):
    # run_chronological_robustness_test() always writes to config.RESULTS_METRICS_DIR -
    # must monkeypatch or this silently overwrites the real project's
    # chronological_split_robustness.csv with synthetic-data results.
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    result = operational_evaluation.run_chronological_robustness_test(synthetic_raw_df)

    assert set(result["model"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert (result["split_type"] == "chronological").all()
    assert "test_fraud" in result.columns
    assert "threshold" in result.columns


def test_run_chronological_robustness_test_saves_csv(tmp_path, monkeypatch, synthetic_raw_df):
    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    operational_evaluation.run_chronological_robustness_test(synthetic_raw_df)
    assert (tmp_path / "chronological_split_robustness.csv").exists()
