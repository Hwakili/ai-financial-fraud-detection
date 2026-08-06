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
| 1 | Data acquisition & EDA | Not started |
| 2 | Preprocessing & imbalance handling | Not started |
| 3 | Baseline models | Not started |
| 4 | RXT (ResNeXt-embedded GRU) model | Not started |
| 5 | Comparative evaluation | Not started |
| 6 | Explainable AI (SHAP/LIME) | Not started |
| 7 | Computational efficiency benchmarking | Not started |
| 8 | Final repo polish | Not started |

## Repository structure

```
ai-fraud-detection/
├── data/
│   ├── raw/            # creditcard.csv goes here (not committed - see Setup)
│   └── processed/      # train/test splits, scaled/resampled arrays
├── notebooks/          # exploratory notebooks (EDA, ad-hoc analysis)
├── src/
│   ├── config.py       # shared paths, random seed, constants
│   ├── preprocessing.py    # (Phase 2) split, scaling, imbalance handling
│   ├── baseline_models.py  # (Phase 3) LR / RF / XGBoost
│   ├── rxt_model.py        # (Phase 4) ResNeXt-embedded GRU
│   ├── evaluate.py         # (Phase 3/5) shared metrics + evaluation pipeline
│   └── explain.py          # (Phase 6) SHAP / LIME
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

Instructions for each phase's scripts/notebooks will be added here as they are built.

- **Phase 0 (this phase):** no runnable script yet — `src/config.py` defines shared paths and
  is imported by every later phase.
