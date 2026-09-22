"""Phase 6b: explanation stability testing.

Added following further investigation, which correctly noted that Phase 6's
SHAP/LIME explanations, as implemented, are a single snapshot - they don't
say whether the SAME explanation would come out again on a rerun. Both
methods used here have a real source of randomness that can affect this:

- LIME (LimeTabularExplainer.explain_instance) perturbs the instance being
  explained by sampling from a local neighbourhood, then fits a local
  surrogate model to those samples. The explainer's internal random state
  advances with every call, so calling explain_instance on the SAME instance
  twice - even with the same explainer object - produces two different sets
  of perturbed samples, and therefore potentially different feature
  attributions and even a different ranked feature list.
- SHAP's KernelExplainer (used here for RXT - see explain.py's docstring for
  why TreeExplainer isn't used for RXT) approximates Shapley values via
  sampled feature coalitions (nsamples=100 in this project's explain.py) -
  another source of run-to-run variation for the same instance.
- SHAP's TreeExplainer (used for Random Forest / XGBoost) computes EXACT
  Shapley values with no sampling - deterministic by construction, so it is
  correctly excluded from stability testing below. Testing it would just
  confirm variance of exactly zero, which is true by design and not a
  finding worth a table row.

This matters directly for this dissertation's argument: if LIME/SHAP
attributions for the same transaction change materially between reruns, that
limits how much weight a single explanation should be given as evidence of
"what the model is really doing" - a finding that belongs in the
explainability discussion alongside the existing PCA-anonymisation
limitation, not just asserted.
"""

import numpy as np
import pandas as pd

from src import config


def _top_k_features(feature_names, importances, k: int) -> set:
    """Names of the k features with the largest |importance|."""
    order = np.argsort(-np.abs(importances))[:k]
    return set(np.asarray(feature_names)[order])


def lime_stability_test(
    explainer_builder,
    predict_proba_fn,
    instance,
    feature_names,
    n_reruns: int = 20,
    top_k: int = 5,
    num_features: int = 10,
) -> dict:
    """Re-run LIME's explain_instance n_reruns times on the SAME instance,
    using a FRESH explainer each time (built by explainer_builder, a
    zero-argument callable returning a new LimeTabularExplainer) so each
    rerun's randomness is independent rather than carried over from a mutating
    shared explainer object - the more conservative test of the two, since
    reusing one explainer's advancing random state could understate variation
    a real "explain this transaction again" workflow would actually see.

    Reports two things:
    1. Jaccard stability of the top-k feature SET across reruns (how often the
       same features appear in the top-k at all, regardless of their exact
       ranking or attributed weight).
    2. Per-feature coefficient variability (mean and std of each feature's
       attributed weight across reruns, for whichever features appeared in
       the top-k on at least one rerun) - the set-level Jaccard score alone
       can look stable while individual weights still swing considerably.
    """
    all_top_k_sets = []
    per_feature_weights = {}

    for _ in range(n_reruns):
        explainer = explainer_builder()
        explanation = explainer.explain_instance(instance, predict_proba_fn, num_features=num_features)
        weights = dict(explanation.as_list())

        # LIME's as_list() keys are human-readable condition strings - either
        # "V14 <= -2.31" (feature name first) or a range like
        # "-0.73 < V2 <= -0.04" (feature name in the middle). Match back to
        # the underlying feature name by finding which known feature name
        # appears as a whole token in the condition string, rather than
        # assuming it's always at the start - a plain startswith() check
        # misses every range-style condition, which would silently undercount
        # how often a feature attributed via a range appears across reruns.
        run_weights = {}
        for condition, weight in weights.items():
            tokens = condition.replace("<", " ").replace(">", " ").replace("=", " ").split()
            matched = next((f for f in feature_names if f in tokens), condition)
            run_weights[matched] = weight
            per_feature_weights.setdefault(matched, []).append(weight)

        top_k_set = set(
            sorted(run_weights, key=lambda f: -abs(run_weights[f]))[:top_k]
        )
        all_top_k_sets.append(top_k_set)

    # Mean pairwise Jaccard similarity across all rerun pairs - 1.0 means
    # every rerun agreed exactly on which features made the top-k, 0.0 means
    # no overlap at all between any pair of reruns.
    pairwise_scores = []
    for i in range(len(all_top_k_sets)):
        for j in range(i + 1, len(all_top_k_sets)):
            union = all_top_k_sets[i] | all_top_k_sets[j]
            intersection = all_top_k_sets[i] & all_top_k_sets[j]
            pairwise_scores.append(len(intersection) / len(union) if union else 1.0)
    mean_jaccard = float(np.mean(pairwise_scores)) if pairwise_scores else np.nan

    feature_variability = pd.DataFrame([
        {
            "feature": feature,
            "n_appearances": len(weights_list),
            "mean_weight": float(np.mean(weights_list)),
            "std_weight": float(np.std(weights_list)),
        }
        for feature, weights_list in per_feature_weights.items()
    ]).sort_values("n_appearances", ascending=False)

    return {
        "n_reruns": n_reruns,
        "top_k": top_k,
        "mean_pairwise_jaccard": mean_jaccard,
        "feature_variability": feature_variability,
    }


def shap_kernel_stability_test(
    predict_fn,
    X_background,
    instance,
    n_reruns: int = 10,
    n_background: int = 50,
    nsamples: int = 100,
) -> pd.DataFrame:
    """Re-run SHAP's KernelExplainer n_reruns times on the SAME instance, each
    time drawing a fresh background sample (as explain.py's
    shap_kernel_explain_rxt does for its one-shot explanation), and report the
    mean and std of each feature's attributed SHAP value across reruns.

    A feature with a small std relative to its mean |value| is a stable
    contributor; a std comparable to or larger than the mean suggests this
    particular attribution shouldn't be treated as a precise number, only as
    a rough direction/magnitude.
    """
    import shap

    all_values = []
    for i in range(n_reruns):
        background = shap.sample(X_background, n_background, random_state=config.RANDOM_SEED + i)
        explainer = shap.KernelExplainer(predict_fn, background)
        values = explainer.shap_values(instance.reshape(1, -1), nsamples=nsamples)
        values = np.asarray(values).flatten()
        all_values.append(values)

    all_values = np.array(all_values)  # shape: (n_reruns, n_features)

    return pd.DataFrame({
        "feature_index": np.arange(all_values.shape[1]),
        "mean_shap_value": all_values.mean(axis=0),
        "std_shap_value": all_values.std(axis=0),
    }).sort_values("mean_shap_value", key=np.abs, ascending=False)


def run_stability_suite(
    lime_explainer_builder,
    lime_predict_proba_fn,
    lime_instance,
    feature_names,
    rxt_predict_fn=None,
    rxt_X_background=None,
    rxt_instance=None,
    model_name: str = "RXT",
) -> dict:
    """Convenience wrapper running both stability tests and saving CSV output.

    rxt_predict_fn/rxt_X_background/rxt_instance are optional - pass them only
    when testing the RXT model's KernelExplainer; omit for tree-based models,
    since TreeExplainer is exact and has nothing to test (see module docstring).
    """
    config.ensure_directories()

    print(f"Running LIME stability test for {model_name} (this may take a minute)...")
    lime_result = lime_stability_test(lime_explainer_builder, lime_predict_proba_fn, lime_instance, feature_names)
    lime_result["feature_variability"].to_csv(
        config.RESULTS_METRICS_DIR / f"lime_stability_{model_name.lower().replace(' ', '_')}.csv", index=False
    )
    print(f"  Mean pairwise Jaccard (top-{lime_result['top_k']} features): {lime_result['mean_pairwise_jaccard']:.3f}")

    shap_result = None
    if rxt_predict_fn is not None:
        print(f"Running SHAP KernelExplainer stability test for {model_name} (this is slow - "
              f"{10} reruns x {100} samples each)...")
        shap_result = shap_kernel_stability_test(rxt_predict_fn, rxt_X_background, rxt_instance)
        shap_result.to_csv(
            config.RESULTS_METRICS_DIR / f"shap_stability_{model_name.lower().replace(' ', '_')}.csv", index=False
        )

    return {"lime": lime_result, "shap": shap_result}


if __name__ == "__main__":
    # Standalone entry point: reload already-persisted models rather than
    # requiring a full run_pipeline.py run (which would mean retraining RXT,
    # ~20-35 minutes on CPU, just to test explanation stability). Matches the
    # pattern every other phase module in this project follows - Phase 3's
    # baseline_models.py and Phase 4's rxt_model.py both persist their models
    # to results/models/ specifically so later phases can reload them.
    import lime.lime_tabular
    import joblib
    import tensorflow as tf

    from src import rxt_model

    train_df = pd.read_csv(config.DATA_PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(config.DATA_PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]
    feature_names = X_test.columns.tolist()

    comparison = pd.read_csv(config.RESULTS_METRICS_DIR / "model_comparison.csv")
    thresholds = dict(zip(comparison["model"], comparison["threshold"]))

    def lime_builder():
        # No random_state - see run_pipeline.py's Phase 6b for why a fixed
        # seed here would make this stability test tautological.
        return lime.lime_tabular.LimeTabularExplainer(
            X_train.values, feature_names=feature_names,
            class_names=["Genuine", "Fraud"], mode="classification",
        )

    rf_path = config.RESULTS_MODELS_DIR / "random_forest.joblib"
    if rf_path.exists():
        rf_model = joblib.load(rf_path)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_pred = (rf_proba >= thresholds["Random Forest"]).astype(int)
        rf_tp_idx = int(np.where((y_test.values == 1) & (rf_pred == 1))[0][0])

        run_stability_suite(
            lime_explainer_builder=lime_builder,
            lime_predict_proba_fn=rf_model.predict_proba,
            lime_instance=X_test.iloc[rf_tp_idx].values,
            feature_names=feature_names,
            model_name="Random Forest",
        )
    else:
        print(f"No persisted Random Forest model at {rf_path} - run Phase 3 first. Skipping.")

    rxt_path = config.RESULTS_MODELS_DIR / "rxt_resnext_gru.keras"
    if rxt_path.exists():
        rxt_trained = tf.keras.models.load_model(rxt_path)

        def rxt_predict_fn(X):
            return rxt_trained.predict(rxt_model.reshape_for_rxt(X), verbose=0).flatten()

        def rxt_predict_proba(X):
            p1 = rxt_predict_fn(X)
            return np.column_stack([1 - p1, p1])

        rxt_proba = rxt_predict_fn(X_test.values)
        rxt_pred = (rxt_proba >= thresholds["RXT (ResNeXt-GRU)"]).astype(int)
        rxt_tp_idx = int(np.where((y_test.values == 1) & (rxt_pred == 1))[0][0])

        run_stability_suite(
            lime_explainer_builder=lime_builder,
            lime_predict_proba_fn=rxt_predict_proba,
            lime_instance=X_test.iloc[rxt_tp_idx].values,
            feature_names=feature_names,
            rxt_predict_fn=rxt_predict_fn,
            rxt_X_background=X_train.values,
            rxt_instance=X_test.iloc[rxt_tp_idx].values,
            model_name="RXT",
        )
    else:
        print(f"No persisted RXT model at {rxt_path} - run Phase 4 first. Skipping.")
