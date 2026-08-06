"""Phase 5: comparative evaluation across every model.

Deliberately reads back the persisted predictions/metrics that Phase 3
(baselines) and Phase 4 (RXT) each save independently, rather than re-training
anything - this is what lets every phase's script be run separately (as the
README documents) while Phase 5 still produces a single, consistent
cross-model comparison.
"""

import pandas as pd

from src import config
from src.evaluate import (
    MASTER_METRICS_CSV,
    load_all_predictions,
    plot_grouped_bar_metrics,
    plot_pr_curves,
    plot_roc_curves,
)


def run_comparative_evaluation() -> dict:
    """Load every model's saved predictions + master metrics table, produce
    the overlaid ROC/PR curves and grouped bar chart required by the ToR's
    evaluation plan ("Baseline comparison" + "Comparative Analysis").
    """
    predictions = load_all_predictions()
    if not predictions:
        raise FileNotFoundError(
            "No saved predictions found. Run src/baseline_models.py and "
            "src/rxt_model.py first (each saves its predictions via "
            "evaluate.save_predictions)."
        )

    if not MASTER_METRICS_CSV.exists():
        raise FileNotFoundError(
            f"{MASTER_METRICS_CSV} not found. Run the baseline and RXT "
            "pipelines first - each appends its row via evaluate.evaluate_model."
        )
    metrics_df = pd.read_csv(MASTER_METRICS_CSV)

    roc_path = plot_roc_curves(predictions)
    pr_path = plot_pr_curves(predictions)
    bar_path = plot_grouped_bar_metrics(metrics_df)

    # Emphasise F1 and AUC in the printed summary, per the ToR: with 0.172%
    # fraud, raw accuracy is not a meaningful ranking criterion.
    ranked = metrics_df.sort_values("f1", ascending=False)

    return {
        "metrics_table": metrics_df,
        "ranked_by_f1": ranked,
        "roc_curve_path": roc_path,
        "pr_curve_path": pr_path,
        "grouped_bar_path": bar_path,
    }


if __name__ == "__main__":
    result = run_comparative_evaluation()
    print(result["ranked_by_f1"][["model", "f1", "auc_roc", "mcc", "precision", "recall"]])
    print(f"\nFigures saved to: {config.RESULTS_FIGURES_DIR}")
