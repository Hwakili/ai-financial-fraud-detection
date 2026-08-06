"""Phase 6: explainability - SHAP and LIME.

Design decisions:

- Tree-based baselines (Random Forest, XGBoost) use shap.TreeExplainer, which
  computes exact Shapley values efficiently by exploiting tree structure.
- The RXT model uses shap.KernelExplainer, not DeepExplainer. DeepExplainer is
  gradient-based and built for feed-forward/conv architectures; GRU's internal
  control-flow ops are a known source of incompatibility with it in TF2's eager
  execution. KernelExplainer is model-agnostic (treats the model as a black-box
  predict function) and works regardless of layer types, at the cost of being
  much slower - hence it is run on a sampled subset of the test set, as the
  coding roadmap anticipates, rather than the full set.
- LIME is applied to individual instructive cases rather than a random sample:
  one correctly-flagged fraud case, one false negative (missed fraud), and one
  false positive. These three are what a fraud-detection discussion chapter
  actually needs - confirming, missing, and over-flagging cases each tell a
  different story about the model's behaviour.
"""

import matplotlib

matplotlib.use("Agg")

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import shap

from src import config


def shap_tree_explain(model, X_explain, feature_names, model_name: str, save_dir=None):
    """SHAP TreeExplainer for Random Forest / XGBoost. Returns shap_values and
    saves a global feature-importance summary plot."""
    save_dir = save_dir or config.RESULTS_FIGURES_DIR
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Binary classifiers via TreeExplainer sometimes return a list [class0, class1]
    values_for_fraud_class = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig = plt.figure()
    shap.summary_plot(values_for_fraud_class, X_explain, feature_names=feature_names, show=False)
    fig.tight_layout()
    path = save_dir / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return values_for_fraud_class, path


def shap_kernel_explain_rxt(predict_fn, X_background, X_explain, feature_names, save_dir=None, n_background=50, nsamples=100):
    """SHAP KernelExplainer for the RXT model — see module docstring for why
    KernelExplainer (not DeepExplainer) is used here."""
    save_dir = save_dir or config.RESULTS_FIGURES_DIR
    background = shap.sample(X_background, n_background, random_state=config.RANDOM_SEED)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples)

    fig = plt.figure()
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
    fig.tight_layout()
    path = save_dir / "shap_summary_rxt.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return shap_values, path


def select_instructive_cases(y_true, y_pred, y_proba) -> dict:
    """Index of one true-positive, one false-negative, and one false-positive
    case - the three most instructive predictions to explain individually."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cases = {}

    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]

    if len(tp_idx) > 0:
        cases["true_positive_fraud_caught"] = int(tp_idx[0])
    if len(fn_idx) > 0:
        cases["false_negative_fraud_missed"] = int(fn_idx[0])
    if len(fp_idx) > 0:
        cases["false_positive_genuine_flagged"] = int(fp_idx[0])

    return cases


def lime_explain_instance(X_train_array, feature_names, predict_proba_fn, instance, save_path):
    """LIME explanation for a single transaction."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_array,
        feature_names=feature_names,
        class_names=["Genuine", "Fraud"],
        mode="classification",
        random_state=config.RANDOM_SEED,
    )
    explanation = explainer.explain_instance(instance, predict_proba_fn, num_features=10)

    fig = explanation.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return explanation


def run_lime_on_instructive_cases(
    X_train_array, X_test_df, y_test, y_pred, predict_proba_fn, feature_names, model_name: str, save_dir=None
):
    """Run LIME on the true-positive/false-negative/false-positive cases for one model."""
    save_dir = save_dir or config.RESULTS_FIGURES_DIR
    cases = select_instructive_cases(y_test, y_pred, None)

    explanations = {}
    for case_name, idx in cases.items():
        instance = X_test_df.iloc[idx].values
        save_path = save_dir / f"lime_{model_name.lower().replace(' ', '_')}_{case_name}.png"
        explanations[case_name] = lime_explain_instance(
            X_train_array, feature_names, predict_proba_fn, instance, save_path
        )

    return explanations
