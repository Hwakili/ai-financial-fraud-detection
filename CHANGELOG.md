# Changelog

## [Unreleased] — ported error-analysis, training-time chart, dataset validation, run provenance

Hamza built a separate, independent version of this project overnight (different codebase, no
RXT - a plain PyTorch GRU only, so it doesn't stand in for this repo). Compared the two and
ported four specific, self-contained improvements worth having here; left the bigger design
differences (a full factorial model x imbalance-strategy sweep, a `--smoke-test` mode) for a
later decision since they change scope/runtime meaningfully.

### Added

- `evaluate.plot_amount_by_error_type()` - box plot of transaction Amount split by
  TP/FN/FP/TN outcome for the best model. Answers a real question the confusion matrix alone
  doesn't: does missed fraud (false negatives) skew toward larger or smaller amounts than caught
  fraud? Directly responds to "Error Analysis" being named as a required evaluation technique in
  the Topic Proposal.
- `efficiency.plot_training_time()` - log-scaled horizontal bar chart of training time across
  models (RXT's ~34 minutes vs the baselines' single-digit seconds would flatten a linear-scale
  chart to nothing).
- `eda.validate_dataset()` - schema/null/finite/binary-label check, run automatically inside
  `eda.load_data()`. Catches a truncated download or schema drift at load time instead of a
  confusing failure two phases downstream.
- `results/metrics/run_metadata.json`, written at the end of `run_pipeline.py` - dataset size,
  split sizes, the best model's headline metrics, and exact library/platform versions for the run
  that produced the committed results. Basic provenance, not tied to any one phase's module.

## [Unreleased] — ToR compliance audit: training time was never actually measured

Cross-checked the repository directly against the signed Terms of Reference (not just the
paraphrased coding guide) and found one concrete gap: Section 6 (Evaluation Plan) explicitly
requires *"Training time and inference speed will be measured to assess suitability for real-time
fraud monitoring environments."* Inference speed was measured correctly, but training time was a
hardcoded `NaN` in every row of `results/metrics/efficiency_comparison.csv` — the timing utility
(`efficiency.measure_training_time()`) existed but was never actually wired up to a real
`model.fit()` call.

### Fixed

- `src/evaluate.py::evaluate_model()` now accepts and records a `train_time_sec` argument
  (defaults to `NaN` if not supplied, so this fails visibly rather than silently reporting a fake
  zero).
- `src/baseline_models.py::train_all_baselines()` now times each of the three `.fit()` calls
  individually with `time.perf_counter()` and returns `(models, train_times)` instead of just
  `models`. `run_baseline_pipeline()` passes each model's real elapsed training time into
  `evaluate_model()`.
- `src/rxt_model.py::run_rxt_pipeline()` now times its `train_rxt()` call the same way and passes
  the real elapsed seconds into `evaluate_model()`.
- `run_pipeline.py`'s Phase 7 efficiency benchmarking now reads the real measured training time
  out of each model's metrics dict instead of passing a hardcoded `float("nan")`.
- Real measured training times (see Results section of README for the current numbers) replace
  the `NaN` placeholders in `results/metrics/efficiency_comparison.csv`.

## [Unreleased] — merge of threshold tuning, three-way split, PR-AUC, tuned resampling ratios

Merged a set of correctness/methodology improvements from a second, parallel version of this
repository into this one (which remains the base — the RXT ResNeXt-embedded GRU architecture is
unchanged). The report-automation feature from that second version (`update_dissertation.py` and
anything resembling it) was deliberately **not** ported: dissertation prose must be written by
the author, not generated.

### Fixed

- **Fixed-threshold bug (the important one).** Every model's classification decision
  (`predict_proba >= threshold`) previously used a hardcoded threshold of 0.5, inherited
  unmodified from scikit-learn's default `.predict()` behaviour. Once a model is trained with
  `class_weight='balanced'` / `scale_pos_weight` (as every model in this project is, to handle
  0.172% fraud prevalence), that weighting skews the model's probability outputs — 0.5 is no
  longer a principled decision boundary. This collapsed precision badly for the linear and
  neural models specifically:
  - **Logistic Regression: F1 0.114 → 0.806**, MCC 0.233 → 0.807 (tuned threshold ≈0.99999999934,
    not 0.5 — the model's raw probabilities were pushed heavily toward the extremes by the class
    weighting).
  - Random Forest and XGBoost were already close to optimal at 0.5 (their `predict_proba` outputs
    are less distorted by class weighting than a linear model's), so their numbers moved less:
    Random Forest F1 0.842 → 0.866, XGBoost F1 0.859 → 0.853 (small movements here are also
    partly attributable to the new three-way split changing the exact train/test partition).
  - RXT's numbers are reported after this fix in the Results section below — see that section for
    the updated figures, since RXT was retrained under the corrected pipeline rather than just
    re-thresholded.

### Added

- `src/evaluate.py::select_threshold(y_true, y_proba)` — selects the threshold that maximises F1
  on the data it's given. Callers are responsible for always passing validation data, never test
  data, into this function; it has no way to enforce that itself.
- `src/evaluate.py::compute_metrics()` now also reports **PR-AUC** (`average_precision_score`)
  alongside accuracy/precision/recall/F1/MCC/AUC-ROC — more informative than ROC-AUC alone under
  this level of class imbalance, since ROC-AUC is dominated by the large true-negative count.
- `src/evaluate.py::evaluate_model()` now records the threshold used in its returned metrics dict
  (and therefore in `results/metrics/model_comparison.csv`), for transparency/reproducibility.
- Three-way stratified 64/16/20 train/validation/test split
  (`src/preprocessing.py::load_and_split()`), replacing the previous 80/20 train/test split. The
  validation split gives threshold tuning (and RXT's early stopping) somewhere to work without
  ever touching the test set.
- `src/preprocessing.py::compare_imbalance_strategies()` now evaluates on the validation set
  instead of the test set, for the same reason.
- Tuned SMOTE/undersampling ratios: `sampling_strategy=0.20` for SMOTE, `sampling_strategy=0.10`
  for undersampling (previously the imblearn default of a full 1:1 rebalance). See
  `src/preprocessing.py`'s module docstring for the full rationale and
  `results/metrics/phase2_imbalance_comparison.csv` for the empirical comparison — the full
  rebalance had produced a severe precision collapse across every resampling strategy.

### Changed

- `src/baseline_models.py::run_baseline_pipeline()` and `src/rxt_model.py::run_rxt_pipeline()`
  now take `(X_train, y_train, X_val, y_val, X_test, y_test)` instead of
  `(X_train, y_train, X_test, y_test)`, and tune each model's threshold on the validation set
  before reporting final metrics on the test set.
- `src/rxt_model.py::run_rxt_pipeline()` no longer carves its own internal validation split out
  of the training set for early stopping — it reuses the project's real validation split for both
  early stopping and threshold tuning, removing a redundant extra split.
- `src/rxt_model.py::train_rxt_kfold()` now also tunes a threshold per fold (via
  `select_threshold`) instead of a hardcoded 0.5, with a documented caveat: each fold's held-out
  partition serves as both the early-stopping validation set and the threshold-tuning/evaluation
  set, which is an accepted simplification for this *supplementary* robustness check — it is not
  the headline result. The properly separated val/test evaluation in `run_rxt_pipeline()` is what
  belongs in the main model-comparison table.

### Not ported (deliberately)

- `update_dissertation.py` and any dissertation-prose-generation functionality from the
  second repository version. Not read, not referenced, not reproduced in any form — dissertation
  writing must be the author's own work for an individually assessed MSc dissertation.
