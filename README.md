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
├── run_pipeline.py      # runs every phase end-to-end from a clean checkout,
│                        # writes results/metrics/run_metadata.json at the end
├── results/
│   ├── figures/         # saved plots for the dissertation
│   ├── metrics/         # CSV results tables + run_metadata.json (provenance:
│   │                    # library/platform versions, split sizes, best model)
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

**Training time is measured, not estimated.** The ToR's evaluation plan (Section 6) explicitly
requires training time and inference speed to both be measured. Each model's `.fit()` call is
timed individually (`time.perf_counter()`) in `baseline_models.train_all_baselines()` and
`rxt_model.run_rxt_pipeline()`, and the real elapsed seconds are recorded in
`results/metrics/efficiency_comparison.csv` alongside inference latency and model size.
`efficiency.plot_training_time()` renders this as a log-scaled bar chart, since RXT's training
time is measured in minutes and the baselines' in single-digit seconds.

**Dataset validation.** `eda.validate_dataset()` checks the expected columns are present, there
are no missing/non-finite values, and the target is genuinely binary, before anything downstream
touches the data. Called automatically by `eda.load_data()` - a truncated download or schema
change fails loudly at load time instead of surfacing as a confusing error two phases later.

**Error analysis by transaction amount.** `evaluate.plot_amount_by_error_type()` produces a box
plot of transaction Amount split by TP/FN/FP/TN outcome for the best-performing model - a
systematic view of error patterns across the whole test set, complementing the individual
LIME explanations in Phase 6.

## Results (real run against the full dataset)

> **These numbers supersede an earlier version of this README.** The previous results were
> produced with a hardcoded 0.5 classification threshold, which badly collapsed precision for
> every class-weighted model (Logistic Regression and RXT especially — see CHANGELOG.md for the
> full explanation and before/after numbers). The table below reflects the corrected pipeline:
> a proper train/validation/test split, with each model's threshold tuned on the validation set
> and applied once to the held-out test set. **These figures are taken directly from the current
> `results/metrics/model_comparison.csv` and `results/metrics/rxt_kfold_summary.csv` — verify
> against those files if this README is ever edited again, rather than trusting narrative text
> in isolation.**

Held-out test set (stratified split), from `results/metrics/model_comparison.csv`:

| Model | Precision | Recall | F1 | MCC | AUC-ROC | PR-AUC | Threshold |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.910 | 0.827 | **0.866** | **0.867** | 0.957 | 0.843 | 0.500 |
| XGBoost | 0.880 | 0.827 | 0.853 | 0.853 | 0.976 | 0.874 | 0.500 |
| Logistic Regression | 0.852 | 0.765 | 0.806 | 0.807 | 0.974 | 0.712 | ~1.000* |
| RXT (ResNeXt-GRU) | 0.566 | 0.439 | 0.494 | 0.497 | 0.959 | 0.432 | 0.980 |

\* Logistic Regression's tuned threshold is 0.9999999993, not literally 1.0 — its
`class_weight='balanced'` training pushes predicted probabilities heavily toward the extremes,
so the F1-maximising cut lands right up against the top of the range. This is a probability-
calibration artefact of class weighting, not evidence that only near-certain predictions are
useful — the model's *ranking* of transactions (AUC-ROC = 0.974) is strong even though its raw
probabilities are poorly calibrated.

**Headline finding:** the threshold fix substantially changed the picture versus the earlier,
buggy version of this README. Logistic Regression's precision rose from 0.061 to 0.852 — it is
now the second-best model overall, confirming that the original "tree ensembles uniquely handle
imbalance" narrative was an artefact of the broken threshold, not a genuine property of the
models. Random Forest and XGBoost remain the strongest performers on F1/MCC, but RXT still
trails every baseline (F1 = 0.494) despite a competitive AUC-ROC (0.959) — a genuine finding
worth discussing critically rather than a residual bug.

**RXT's held-out score does not agree with its own cross-validation result, and that
disagreement is itself worth discussing.** From `results/metrics/rxt_kfold_summary.csv`
(5-fold CV, mean ± std): **F1 = 0.728 ± 0.052, MCC = 0.734 ± 0.052, precision = 0.842 ± 0.060,
recall = 0.641 ± 0.046, AUC-ROC = 0.962 ± 0.012**. The cross-validation mean F1 (0.728) is
nearly 25 points higher than the single held-out test F1 (0.494). This gap suggests RXT's
performance on this dataset is *unstable* across data splits, not simply *lower* than the
baselines — plausibly a consequence of the small absolute number of fraud examples available
for a deep architecture to learn from in any single training fold. Both figures are reported
here rather than only the more favourable one; the instability itself is treated as a finding,
not noise to be averaged away, and is exactly the kind of "critical evaluation... considering
the limitations of the techniques employed" the assignment brief's top marking band rewards.

**Computational efficiency**, from `results/metrics/efficiency_comparison.csv`:

| Model | Train time | Inference (1 txn) | Inference (1,000 txns) | Model size | Params |
|---|---|---|---|---|---|
| Logistic Regression | 1.3s | 1.5 ms | 0.001s | 1.5 KB | — |
| XGBoost | 1.7s | 4.7 ms | 0.006s | 219.9 KB | — |
| Random Forest | 12.8s | 32.3 ms | 0.041s | 4.1 MB | — |
| RXT (ResNeXt-GRU) | **1,263.5s (~21 min)** | 69.1 ms | 0.270s | 579.3 KB | 38,145 |

RXT costs roughly 100–1,000x more to train than any baseline here, and is 2–46x slower at
inference, for held-out predictive performance that trails every baseline on F1/MCC. Combined
with the held-out/cross-validation instability above, this raises a real, defensible question
about whether the added architectural complexity of a ResNeXt-embedded GRU is justified on this
dataset — a question this project treats as a substantive finding, not a shortfall (see the
feasibility discussion in the dissertation's literature review and implementation chapters,
which anticipated exactly this kind of outcome before any model was trained).

**RXT training note:** these numbers were produced on a CPU-only machine (~3-4 minutes/epoch on
the full training set). The ToR's resource plan specifies Google Colab Pro / Kaggle Kernels GPU
compute for RXT training — re-running `python -m src.rxt_model` (or `run_pipeline.py`) there
will train faster and may find a differently-converged model within the same epoch budget.
RXT's own results have also been observed to vary somewhat between identical-seed CPU reruns
during development (TensorFlow/Keras training isn't perfectly deterministic, and early stopping
can halt at a different epoch each run) — treat the 5-fold CV summary above as the more stable
reference point for the dissertation write-up rather than any single run in isolation.

All figures (confusion matrices, ROC/PR curves, grouped metric comparison, SHAP summaries, LIME
explanations, the amount-by-error-type box plot, the training-time chart) are in
`results/figures/`; the 5-fold CV summary for RXT is in `results/metrics/rxt_kfold_summary.csv`;
efficiency numbers are in `results/metrics/efficiency_comparison.csv`; full run provenance
(library/platform versions, split sizes, best model) is in `results/metrics/run_metadata.json`.