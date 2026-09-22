"""Unit tests for src/explanation_stability.py.

Uses a trivial RandomForest on tiny synthetic data (not a real RXT model) as
a stand-in "black-box predict function" - these tests exist to verify the
stability-testing mechanism itself (Jaccard computation, feature-name
matching, CSV output), not to produce a real stability finding. n_reruns/
nsamples are reduced from the module's production defaults where the test
calls the lower-level functions directly, purely for test speed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import config, explanation_stability


@pytest.fixture
def synthetic_model_and_data():
    rng = np.random.default_rng(0)
    n_samples, n_features = 80, 6
    feature_names = [f"V{i}" for i in range(n_features)]

    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + rng.normal(scale=0.1, size=n_samples) > 0).astype(int)

    X_df = pd.DataFrame(X, columns=feature_names)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_df, y)

    return model, X_df, y, feature_names


def test_top_k_features_selects_largest_magnitude():
    feature_names = ["a", "b", "c", "d"]
    importances = np.array([0.1, -0.9, 0.05, 0.4])

    result = explanation_stability._top_k_features(feature_names, importances, k=2)

    assert result == {"b", "d"}


def test_lime_stability_test_returns_valid_jaccard_and_variability(synthetic_model_and_data):
    import lime.lime_tabular

    model, X_df, y, feature_names = synthetic_model_and_data
    instance = X_df.iloc[0].values

    def builder():
        return lime.lime_tabular.LimeTabularExplainer(
            X_df.values, feature_names=feature_names,
            class_names=["Genuine", "Fraud"], mode="classification",
        )

    result = explanation_stability.lime_stability_test(
        builder, model.predict_proba, instance, feature_names, n_reruns=3, top_k=3, num_features=5
    )

    assert 0.0 <= result["mean_pairwise_jaccard"] <= 1.0
    assert isinstance(result["feature_variability"], pd.DataFrame)
    assert set(result["feature_variability"].columns) == {"feature", "n_appearances", "mean_weight", "std_weight"}
    assert result["feature_variability"]["n_appearances"].max() <= 3


def test_lime_stability_test_matches_range_style_conditions_to_feature_names(synthetic_model_and_data):
    """The real bug this module fixed: LIME's as_list() can key a condition as
    a range like '-0.73 < V2 <= -0.04' (feature name in the middle, not at the
    start) - a plain startswith() check would silently fail to match this back
    to 'V2' and instead leave the raw condition string as the "feature" name.
    Run enough reruns on real LIME output that a range-style condition is
    very likely to appear at least once, and confirm every matched key is a
    real feature name, never a leftover raw condition string."""
    import lime.lime_tabular

    model, X_df, y, feature_names = synthetic_model_and_data
    instance = X_df.iloc[0].values

    def builder():
        return lime.lime_tabular.LimeTabularExplainer(
            X_df.values, feature_names=feature_names,
            class_names=["Genuine", "Fraud"], mode="classification",
        )

    result = explanation_stability.lime_stability_test(
        builder, model.predict_proba, instance, feature_names, n_reruns=5, top_k=3, num_features=6
    )

    matched_features = set(result["feature_variability"]["feature"])
    assert matched_features.issubset(set(feature_names)), (
        f"found unmatched condition string(s) treated as a feature name: {matched_features - set(feature_names)}"
    )


def test_shap_kernel_stability_test_returns_mean_and_std_per_feature(synthetic_model_and_data):
    model, X_df, y, feature_names = synthetic_model_and_data
    instance = X_df.iloc[0].values

    def predict_fn(X):
        return model.predict_proba(X)[:, 1]

    result = explanation_stability.shap_kernel_stability_test(
        predict_fn, X_df.values, instance, n_reruns=2, n_background=20, nsamples=20
    )

    assert list(result.columns) == ["feature_index", "mean_shap_value", "std_shap_value"]
    assert len(result) == len(feature_names)
    assert (result["std_shap_value"] >= 0).all()


def test_run_stability_suite_saves_lime_csv_only_without_rxt_args(tmp_path, monkeypatch, synthetic_model_and_data):
    import lime.lime_tabular

    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    model, X_df, y, feature_names = synthetic_model_and_data
    instance = X_df.iloc[0].values

    def builder():
        return lime.lime_tabular.LimeTabularExplainer(
            X_df.values, feature_names=feature_names,
            class_names=["Genuine", "Fraud"], mode="classification",
        )

    # lime_stability_test's own n_reruns default (20) is used here since
    # run_stability_suite doesn't expose it - fine at this tiny data size.
    result = explanation_stability.run_stability_suite(
        lime_explainer_builder=builder,
        lime_predict_proba_fn=model.predict_proba,
        lime_instance=instance,
        feature_names=feature_names,
        model_name="TestTreeModel",
    )

    assert result["shap"] is None
    assert (tmp_path / "lime_stability_testtreemodel.csv").exists()
    assert not (tmp_path / "shap_stability_testtreemodel.csv").exists()


def test_run_stability_suite_saves_both_csvs_with_rxt_args(tmp_path, monkeypatch, synthetic_model_and_data):
    import lime.lime_tabular

    monkeypatch.setattr(config, "RESULTS_METRICS_DIR", tmp_path)
    model, X_df, y, feature_names = synthetic_model_and_data
    instance = X_df.iloc[0].values

    def builder():
        return lime.lime_tabular.LimeTabularExplainer(
            X_df.values, feature_names=feature_names,
            class_names=["Genuine", "Fraud"], mode="classification",
        )

    def predict_fn(X):
        return model.predict_proba(X)[:, 1]

    result = explanation_stability.run_stability_suite(
        lime_explainer_builder=builder,
        lime_predict_proba_fn=model.predict_proba,
        lime_instance=instance,
        feature_names=feature_names,
        rxt_predict_fn=predict_fn,
        rxt_X_background=X_df.values,
        rxt_instance=instance,
        model_name="TestRXTStandIn",
    )

    assert result["shap"] is not None
    assert (tmp_path / "lime_stability_testrxtstandin.csv").exists()
    assert (tmp_path / "shap_stability_testrxtstandin.csv").exists()
