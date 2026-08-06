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
│   └── processed/      # train/test splits, scaled/resampled arrays
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

# Phase 2 — split, scale, compare SMOTE / class-weighting / undersampling
# (writes data/processed/{train,test}.csv and results/metrics/phase2_imbalance_comparison.csv)
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

## Results (real run against the full dataset)

Held-out test set (56,962 rows, stratified 80/20 split), full `results/metrics/model_comparison.csv`:

| Model | Precision | Recall | F1 | MCC | AUC-ROC |
|---|---|---|---|---|---|
| XGBoost | 0.882 | 0.837 | **0.859** | **0.859** | 0.968 |
| Random Forest | 0.906 | 0.786 | 0.842 | 0.843 | 0.957 |
| RXT (ResNeXt-GRU) | 0.154 | 0.857 | 0.261 | 0.361 | 0.959 |
| Logistic Regression | 0.061 | 0.918 | 0.114 | 0.233 | 0.972 |

**Headline finding:** the gradient-boosted tree ensembles (XGBoost, Random Forest) clearly
outperform both the RXT deep learning model and the linear baseline on F1/MCC, despite RXT and
Logistic Regression achieving comparable or even higher AUC-ROC. The pattern is consistent
across every class-weighted model here (RXT, Logistic Regression): pushing hard on recall to
compensate for 0.172% fraud prevalence collapses precision to ~6-15%, whereas the tree ensembles
handle the same class weighting (`scale_pos_weight`/`class_weight='balanced'`) without that
collapse. This is a genuine, citable finding (tree-based models are well known in recent
literature to be strong, sometimes state-of-the-art, on tabular data relative to deep learning)
and directly informs the Phase 2 imbalance-strategy comparison in
`results/metrics/phase2_imbalance_comparison.csv`, which shows the same precision/recall
trade-off pattern under SMOTE/undersampling/class-weighting.

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
