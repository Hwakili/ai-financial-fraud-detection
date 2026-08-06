"""End-to-end pipeline runner: Phases 1-7 in sequence.

Each phase also has its own `python -m src.<module>` entry point (see README)
so any single phase can be re-run in isolation - this script exists for a full
reproducible run from a clean checkout, e.g. before submission.

Usage:
    python run_pipeline.py
"""

import numpy as np
import pandas as pd

from src import (
    baseline_models,
    comparative_evaluation,
    config,
    data_acquisition,
    eda,
    efficiency,
    explain,
    preprocessing,
    rxt_model,
)


def main():
    print("=" * 60)
    print("PHASE 1 — Data acquisition & EDA")
    print("=" * 60)
    data_path = data_acquisition.download_dataset()
    df = eda.load_data(data_path)
    summary = eda.run_full_eda(data_path)
    print(f"Dataset shape: {summary['shape']}, fraud rate: {summary['fraud_rate_pct']}%")

    print("\n" + "=" * 60)
    print("PHASE 2 — Preprocessing & imbalance handling")
    print("=" * 60)
    prep_result = preprocessing.run_preprocessing_pipeline(df)
    print(prep_result["comparison"])

    train_df = pd.read_csv(config.DATA_PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(config.DATA_PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]

    class_weights = preprocessing.compute_class_weights(y_train)

    print("\n" + "=" * 60)
    print("PHASE 3 — Baseline models")
    print("=" * 60)
    models, baseline_metrics = baseline_models.run_baseline_pipeline(X_train, y_train, X_test, y_test)
    for name, metrics in baseline_metrics.items():
        print(name, metrics)

    print("\n" + "=" * 60)
    print("PHASE 4 — RXT (ResNeXt-embedded GRU)")
    print("=" * 60)
    rxt_trained, rxt_metrics, _ = rxt_model.run_rxt_pipeline(
        X_train, y_train, X_test, y_test, class_weight=class_weights, epochs=50
    )
    print("RXT held-out test metrics:", rxt_metrics)

    kfold_results = rxt_model.train_rxt_kfold(
        X_train, y_train, n_splits=5, epochs=30, class_weight=class_weights
    )
    kfold_summary = rxt_model.summarize_kfold_results(kfold_results)
    kfold_summary.to_csv(config.RESULTS_METRICS_DIR / "rxt_kfold_summary.csv")
    print("RXT 5-fold CV summary (mean +/- std):")
    print(kfold_summary)

    models["RXT (ResNeXt-GRU)"] = rxt_trained

    print("\n" + "=" * 60)
    print("PHASE 5 — Comparative evaluation")
    print("=" * 60)
    comparative_result = comparative_evaluation.run_comparative_evaluation()
    print(comparative_result["ranked_by_f1"][["model", "f1", "auc_roc", "mcc"]])

    print("\n" + "=" * 60)
    print("PHASE 6 — Explainable AI (SHAP / LIME)")
    print("=" * 60)
    feature_names = X_test.columns.tolist()

    for name in ["Random Forest", "XGBoost"]:
        sample = X_test.sample(min(200, len(X_test)), random_state=config.RANDOM_SEED)
        explain.shap_tree_explain(models[name], sample, feature_names, name)

    def rxt_predict_fn(X):
        return rxt_trained.predict(rxt_model.reshape_for_rxt(X), verbose=0).flatten()

    def rxt_predict_proba(X):
        p1 = rxt_predict_fn(X)
        return np.column_stack([1 - p1, p1])

    bg_sample = X_train.sample(min(50, len(X_train)), random_state=config.RANDOM_SEED).values
    explain_sample = X_test.sample(min(20, len(X_test)), random_state=config.RANDOM_SEED).values
    explain.shap_kernel_explain_rxt(rxt_predict_fn, bg_sample, explain_sample, feature_names)

    best_model_name = comparative_result["ranked_by_f1"].iloc[0]["model"]
    best_model = models[best_model_name]
    y_pred_best = (
        best_model.predict(X_test) if best_model_name != "RXT (ResNeXt-GRU)" else (rxt_predict_fn(X_test.values) >= 0.5).astype(int)
    )
    predict_proba_fn = (
        best_model.predict_proba if best_model_name != "RXT (ResNeXt-GRU)" else rxt_predict_proba
    )
    explain.run_lime_on_instructive_cases(
        X_train.values, X_test, y_test, y_pred_best, predict_proba_fn, feature_names, best_model_name
    )

    print("\n" + "=" * 60)
    print("PHASE 7 — Computational efficiency benchmarking")
    print("=" * 60)
    single_row = X_test.iloc[[0]]
    batch = X_test.iloc[: min(1000, len(X_test))]
    records = []

    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        model = models[name]
        key = name.lower().replace(" ", "_")
        size_bytes = efficiency.get_sklearn_model_disk_size(model, config.RESULTS_MODELS_DIR / f"{key}.joblib")
        record = efficiency.benchmark_model(
            name, model.predict, single_row, batch,
            train_time_sec=float("nan"), model_size_bytes=size_bytes,
        )
        records.append(record)

    rxt_single = single_row.values.reshape(1, -1)
    rxt_batch = batch.values
    # RXT was already persisted by run_rxt_pipeline() above - measure that file
    # directly rather than saving a second copy just for its size.
    size_bytes = (config.RESULTS_MODELS_DIR / "rxt_resnext_gru.keras").stat().st_size
    rxt_record = efficiency.benchmark_model(
        "RXT (ResNeXt-GRU)", rxt_predict_fn, rxt_single, rxt_batch,
        train_time_sec=float("nan"), model_size_bytes=size_bytes,
        param_count=rxt_trained.count_params(),
    )
    records.append(rxt_record)

    efficiency_table = efficiency.build_efficiency_table(records)
    print(efficiency_table)

    print("\nAll phases complete. See results/figures, results/metrics, results/models.")


if __name__ == "__main__":
    main()
