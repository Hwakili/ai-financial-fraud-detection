# AI for Fraud Detection in Financial Transactions

MSc Cyber Security dissertation project (Manchester Metropolitan University, module 6G7V0007).
Develops and critically compares AI models — baseline classifiers (Logistic Regression, Random
Forest, XGBoost) and a ResNeXt-embedded GRU deep learning architecture ("RXT") — for detecting
fraudulent credit card transactions under severe class imbalance, with an emphasis on
explainability (SHAP/LIME) and computational efficiency for real-time deployment feasibility.

**Author:** Hamza Wakili (25934110) | **Supervisor:** Safiullah Khan

## Project status

This repository is built incrementally, phase by phase, alongside the dissertation write-up.

| Phase | Description | Status |
|-------|--------------|--------|
| 0 | Environment & repo setup | Done |
| 1 | Data acquisition & EDA | Done |
| 2 | Preprocessing & imbalance handling | Done |
| 3 | Baseline models (LR / RF / XGBoost) | Done |
| 4 | RXT (ResNeXt-embedded GRU) model | Done |
| 5 | Comparative evaluation | Done |
| 6 | Explainable AI (SHAP/LIME) | Done |
| 7 | Computational efficiency benchmarking | Done |
| 8 | Final repo polish | Done |

## Repository structure

```
ai-fraud-detection/
├── data/
│   ├── raw/            # creditcard.csv goes here (not committed - see Setup)
│   └── processed/      # train/val/test splits (64/16/20, scaled)
├── notebooks/          # exploratory notebooks (EDA, ad-hoc analysis)
├── src/
│   ├── config.py               # shared paths, random seed, constants
│   ├── data_acquisition.py     # (Phase 1) Kaggle dataset download
│   ├── eda.py                  # (Phase 1) EDA plots + summary stats
│   ├── preprocessing.py        # (Phase 2) split, scaling, imbalance handling
│   ├── baseline_models.py      # (Phase 3) LR / RF / XGBoost
│   ├── rxt_model.py            # (Phase 4) ResNeXt-embedded GRU
│   ├── evaluate.py             # (Phase 3-5) shared metrics + evaluation pipeline
│   ├── comparative_evaluation.py   # (Phase 5) cross-model ROC/PR/bar charts
│   ├── explain.py              # (Phase 6) SHAP / LIME
│   └── efficiency.py           # (Phase 7) training/inference time, model size
├── run_pipeline.py      # runs every phase end-to-end from a clean checkout
├── results/
│   ├── figures/         # saved plots for the dissertation
│   ├── metrics/         # CSV results tables
│   └── models/          # trained model artefacts (gitignored - regenerate via scripts)
├── tests/               # pytest unit tests
├── requirements.txt
└── README.md
```

## Environment

- Python 3.11
- Kaggle account with API access (for dataset download in Phase 1)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

The dataset (Kaggle/ULB Credit Card Fraud Detection, 284,807 transactions, 492 fraudulent) is
**not committed to this repository** — it is publicly available and should be downloaded locally:

```bash
# via kagglehub - normally requires a Kaggle API token at ~/.kaggle/kaggle.json,
# though kagglehub has been observed to succeed without one for this specific
# public dataset. If it fails for you, create a token at kaggle.com/settings.
python -c "import kagglehub; kagglehub.dataset_download('mlg-ulb/creditcardfraud')"
```

Place the resulting `creditcard.csv` in `data/raw/`. See `src/config.py` for the exact expected
path (`RAW_DATA_FILE`).

## How to run each script

Run the whole pipeline end-to-end (from a clean checkout, once `creditcard.csv` is in
`data/raw/`):

```bash
python run_pipeline.py
```

Or run each phase independently:

```bash
# Phase 1 — download + EDA (writes figures to results/figures/)
python -m src.data_acquisition
python -m src.eda
# or: jupyter notebook notebooks/01_eda.ipynb

# Phase 2 — 64/16/20 train/val/test split, scale, compare SMOTE / class-weighting / undersampling
# on the VALIDATION set (test set stays untouched)
# (writes data/processed/{train,val,test}.csv and results/metrics/phase2_imbalance_comparison.csv)
python -m src.preprocessing

# Phase 3 — Logistic Regression, Random Forest, XGBoost
# (appends to results/metrics/model_comparison.csv, saves models to results/models/)
python -m src.baseline_models

# Phase 4 — RXT (ResNeXt-embedded GRU): held-out test evaluation + 5-fold CV
python -m src.rxt_model

# Phase 5 — cross-model ROC/PR curves + grouped bar chart
# (requires Phase 3 and Phase 4 to have been run first)
python -m src.comparative_evaluation

# Phase 6 / 7 — SHAP, LIME, and efficiency benchmarking are run together
# as part of run_pipeline.py (they need trained model objects in memory);
# see src/explain.py and src/efficiency.py for the underlying functions if
# you want to call them directly on a specific model.
```

### Tests

```bash
pytest tests/ -v
```

Tests use small synthetic datasets (not the real download), so they run in seconds and
without a Kaggle token - they check the pipeline logic (shapes, stratification, metric
correctness), not real-world fraud-detection performance.

## Methodology notes

**Three-way split (64/16/20 train/validation/test).** Every model is trained on `train`,
threshold-tuned on `validation`, and reported on `test` exactly once. See
`src/preprocessing.py::load_and_split()`.

**Classification-threshold tuning.** Every model's final classification threshold is selected
by `src/evaluate.py::select_threshold()`, which maximises F1 on validation-set predictions only.
This matters a great deal here: models trained with `class_weight='balanced'`/`scale_pos_weight`
have probability outputs skewed by that weighting, so the sklearn default of thresholding
`predict_proba >= 0.5` is no longer a principled decision boundary once that weighting is applied.
See the CHANGELOG for the concrete effect this had on the reported numbers.

**Tuned SMOTE/undersampling ratios.** `src/preprocessing.py` uses `sampling_strategy=0.20` for
SMOTE and `sampling_strategy=0.10` for undersampling (raising the fraud class to 20%/10% of the
majority count) rather than a full 1:1 rebalance. This is a design decision documented in-code
(`src/preprocessing.py`, module docstring and `SMOTE_SAMPLING_STRATEGY`/
`UNDERSAMPLE_SAMPLING_STRATEGY`) - a full rebalance was tried first and produced a severe
precision collapse; see `results/metrics/phase2_imbalance_comparison.csv` for the empirical
comparison.

## Results (real run against the full dataset)

> **These numbers supersede an earlier version of this README.** The previous results were
> produced with a hardcoded 0.5 classification threshold, which badly collapsed precision for
> every class-weighted model (Logistic Regression and RXT especially — see CHANGELOG.md for the
> full explanation and before/after numbers). The table below reflects the corrected pipeline:
> a proper 64/16/20 train/validation/test split, with each model's threshold tuned on the
> validation set and applied once to the held-out test set.

Held-out test set (56,962 rows, stratified from the 64/16/20 split), full
`results/metrics/model_comparison.csv`:

| Model | Precision | Recall | F1 | MCC | AUC-ROC | PR-AUC | Threshold |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.910 | 0.827 | **0.866** | **0.867** | 0.957 | 0.843 | 0.500 |
| XGBoost | 0.880 | 0.827 | 0.853 | 0.853 | 0.976 | 0.874 | 0.500 |
| Logistic Regression | 0.852 | 0.765 | 0.806 | 0.807 | 0.974 | 0.712 | 1.000* |
| RXT (ResNeXt-GRU) | 0.817 | 0.684 | 0.744 | 0.747 | 0.972 | 0.707 | 0.999 |

\* Logistic Regression's tuned threshold is 0.9999999993, not literally 1.0 — its
`class_weight='balanced'` training pushes predicted probabilities heavily toward the extremes,
so the F1-maximising cut lands right up against the top of the range.

RXT's 5-fold cross-validation summary (`results/metrics/rxt_kfold_summary.csv`, a supplementary
robustness check — see CHANGELOG.md for why its threshold handling is looser than the table
above): F1 = 0.687 ± 0.084, MCC = 0.691 ± 0.085, AUC-ROC = 0.963 ± 0.019 across folds — consistent
with the held-out test result, i.e. not a lucky single split.

**Headline finding:** all four models are now much closer together than before threshold tuning.
Random Forest and XGBoost still lead on F1/MCC, but Logistic Regression and RXT — the two models
whose `class_weight='balanced'` training most distorted their probability outputs — improved
dramatically once evaluated at their own properly-tuned thresholds rather than an arbitrary 0.5.
This is worth stating plainly in the dissertation: **the earlier "tree ensembles crush RXT"
result was substantially a thresholding artefact, not a genuine architecture gap.** RXT remains
behind the tree ensembles, but the gap is now 0.12 F1 (0.866 vs 0.744), not 0.60. The residual gap
is a more defensible, more interesting finding to discuss critically than the pre-fix numbers
were — e.g. RXT's recall (0.684) trails its precision (0.817) by less than the tree ensembles'
gap, suggesting a different precision/recall balance rather than simply "worse."

**RXT training note:** this repository's RXT numbers were produced on a CPU-only machine
(~3-4 minutes/epoch on the full training set). Your ToR's resource plan specifies Google Colab
Pro / Kaggle Kernels GPU compute for RXT training - re-running `python -m src.rxt_model` (or
`run_pipeline.py`) there will train faster and may find a better-converged model within the same
epoch budget; treat the numbers above as a real, working result rather than the final
GPU-tuned figures for the dissertation.

All figures (confusion matrices, ROC/PR curves, grouped metric comparison, SHAP summaries, LIME
explanations) are in `results/figures/`; the 5-fold CV summary for RXT is in
`results/metrics/rxt_kfold_summary.csv`; efficiency numbers are in
`results/metrics/efficiency_comparison.csv`.
