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
# via kagglehub (requires a Kaggle API token at ~/.kaggle/kaggle.json)
python -c "import kagglehub; kagglehub.dataset_download('mlg-ulb/creditcardfraud')"
```

Place the resulting `creditcard.csv` in `data/raw/`. See `src/config.py` for the exact expected
path (`RAW_DATA_FILE`).

## Running the tests

```bash
pytest tests/ -v
```

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
