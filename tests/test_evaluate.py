"""Unit tests for src/evaluate.py — the shared metrics/plotting pipeline used
by every model phase, so correctness here matters more than almost anywhere
else in the codebase."""

import numpy as np
import pandas as pd

from src import evaluate


def test_compute_metrics_perfect_predictions():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.1, 0.9, 0.8]

    metrics = evaluate.compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["auc_roc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_compute_metrics_all_wrong_on_minority_class():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 0, 0]

    metrics = evaluate.compute_metrics(y_true, y_pred)

    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0
    assert np.isnan(metrics["auc_roc"])
    assert np.isnan(metrics["pr_auc"])


def test_select_threshold_maximises_f1_on_given_data():
    # Constructed so that thresholding at 0.6 gives perfect F1 (catches both
    # positives, no false positives), while the default 0.5 would also work
    # here but a naive low threshold (e.g. 0.05) would pull in false positives.
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.3, 0.55, 0.7, 0.9])

    threshold = evaluate.select_threshold(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    from sklearn.metrics import f1_score
    assert f1_score(y_true, y_pred) == 1.0


def test_select_threshold_is_deterministic_and_bounded():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_proba = rng.random(size=50)

    threshold_a = evaluate.select_threshold(y_true, y_proba)
    threshold_b = evaluate.select_threshold(y_true, y_proba)

    assert threshold_a == threshold_b
    assert 0.0 <= threshold_a <= 1.0


def test_evaluate_model_records_threshold(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "RESULTS_FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(evaluate, "MASTER_METRICS_CSV", tmp_path / "master.csv")

    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    metrics_default = evaluate.evaluate_model("ModelA", y_true, y_pred, y_proba)
    assert metrics_default["threshold"] == 0.5

    metrics_tuned = evaluate.evaluate_model("ModelB", y_true, y_pred, y_proba, threshold=0.63)
    assert metrics_tuned["threshold"] == 0.63


def test_plot_confusion_matrix_saves_file(tmp_path):
    y_true = [0, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 0]
    path = evaluate.plot_confusion_matrix(y_true, y_pred, "TestModel", save_dir=tmp_path)
    assert path.exists()


def test_append_metrics_to_master_upserts_by_model_name(tmp_path):
    csv_path = tmp_path / "master.csv"

    evaluate.append_metrics_to_master({"accuracy": 0.9, "f1": 0.5}, "ModelA", csv_path=csv_path)
    df1 = evaluate.append_metrics_to_master(
        {"accuracy": 0.95, "f1": 0.6}, "ModelB", csv_path=csv_path
    )
    assert len(df1) == 2

    df2 = evaluate.append_metrics_to_master(
        {"accuracy": 0.99, "f1": 0.7}, "ModelA", csv_path=csv_path
    )
    assert len(df2) == 2
    updated_row = df2.loc[df2["model"] == "ModelA"].iloc[0]
    assert updated_row["accuracy"] == 0.99


def test_plot_roc_curves_saves_file(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    predictions = {
        "ModelA": (y_true, rng.random(size=100)),
        "ModelB": (y_true, rng.random(size=100)),
    }
    path = evaluate.plot_roc_curves(predictions, save_path=tmp_path / "roc.png")
    assert path.exists()


def test_plot_pr_curves_saves_file(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    predictions = {
        "ModelA": (y_true, rng.random(size=100)),
    }
    path = evaluate.plot_pr_curves(predictions, save_path=tmp_path / "pr.png")
    assert path.exists()


def test_plot_grouped_bar_metrics_saves_file(tmp_path):
    df = pd.DataFrame(
        [
            {"model": "ModelA", "precision": 0.8, "recall": 0.7, "f1": 0.75, "auc_roc": 0.9, "pr_auc": 0.7, "mcc": 0.6},
            {"model": "ModelB", "precision": 0.85, "recall": 0.75, "f1": 0.8, "auc_roc": 0.92, "pr_auc": 0.75, "mcc": 0.65},
        ]
    )
    path = evaluate.plot_grouped_bar_metrics(df, save_path=tmp_path / "grouped.png")
    assert path.exists()
