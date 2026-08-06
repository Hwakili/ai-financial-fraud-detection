"""Phase 7: computational efficiency benchmarking.

Measures training time, inference time (single transaction and batch), and
model size/parameter count for every model, so that Chapter 6.5 can discuss
the accuracy-vs-speed trade-off explicitly - a fast, cheap baseline might lose
on F1 but still be the right choice if RXT's accuracy gain comes at a cost
that rules out real-time deployment.

`predict_fn` is deliberately generic (any callable X -> predictions) so this
module works identically for sklearn/XGBoost models and for the RXT Keras
model - the caller is responsible for any input reshaping the model needs.
"""

import time
from pathlib import Path

import joblib
import pandas as pd

from src import config


def measure_training_time(train_fn, *args, **kwargs):
    """Wall-clock time for a full training run. Returns (trained_object, seconds)."""
    start = time.perf_counter()
    result = train_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def measure_inference_time_single(predict_fn, X_single_row, n_repeats: int = 100) -> float:
    """Average seconds to predict one transaction, over n_repeats calls to
    smooth out noise from a single timing measurement."""
    start = time.perf_counter()
    for _ in range(n_repeats):
        predict_fn(X_single_row)
    elapsed = time.perf_counter() - start
    return elapsed / n_repeats


def measure_inference_time_batch(predict_fn, X_batch) -> float:
    """Seconds to predict an entire batch in one call - relevant to throughput
    claims, as opposed to single-transaction latency."""
    start = time.perf_counter()
    predict_fn(X_batch)
    return time.perf_counter() - start


def get_sklearn_model_disk_size(model, tmp_path: Path) -> int:
    """Bytes on disk for a joblib-serialised sklearn/XGBoost model."""
    joblib.dump(model, tmp_path)
    size = tmp_path.stat().st_size
    return size


def get_keras_model_disk_size(model, tmp_path: Path) -> int:
    """Bytes on disk for a saved Keras model (.keras format)."""
    model.save(tmp_path)
    if tmp_path.is_dir():
        return sum(f.stat().st_size for f in tmp_path.rglob("*") if f.is_file())
    return tmp_path.stat().st_size


def benchmark_model(
    model_name: str,
    predict_fn,
    X_test_single_row,
    X_test_batch,
    train_time_sec: float,
    model_size_bytes: int,
    param_count: int | None = None,
    n_repeats: int = 100,
) -> dict:
    """Assemble one model's full efficiency record."""
    inference_single_sec = measure_inference_time_single(predict_fn, X_test_single_row, n_repeats)
    inference_batch_sec = measure_inference_time_batch(predict_fn, X_test_batch)

    return {
        "model": model_name,
        "train_time_sec": round(train_time_sec, 4),
        "inference_time_single_ms": round(inference_single_sec * 1000, 4),
        "inference_time_batch_sec": round(inference_batch_sec, 4),
        "batch_size": len(X_test_batch),
        "model_size_kb": round(model_size_bytes / 1024, 2),
        "param_count": param_count,
    }


def build_efficiency_table(records: list, save_path=None) -> pd.DataFrame:
    """Combine every model's efficiency record into one comparison table."""
    save_path = save_path or config.RESULTS_METRICS_DIR / "efficiency_comparison.csv"
    config.ensure_directories()

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    return df
