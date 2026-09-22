# Changelog

## [Unreleased] — pinned dependencies, cross-platform install, statistical robustness & operational analysis

Improvements made after further assessment and investigation into the repository's
reproducibility and the strength of its statistical claims.

### Fixed

- **Phase 2 threshold inconsistency.** `preprocessing.compare_imbalance_strategies()` was
  scoring every imbalance strategy (SMOTE, undersampling, class weighting, no resampling) with
  `model.predict()` - an implicit fixed 0.5 threshold - while every other result in this project
  (Phase 3 baselines, Phase 4 RXT) uses `evaluate.select_threshold()`, tuned on the validation
  set. This meant Phase 2's diagnostic and the project's headline results were not measuring
  like-for-like: a strategy could look artificially strong or weak purely because 0.5 suited its
  probability distribution better or worse, independent of how good the strategy actually is once
  thresholds are chosen properly. Fixed so every strategy is scored at its own validation-tuned
  threshold; `results/metrics/phase2_imbalance_comparison.csv` gains a `threshold` column.

- **LIME stability test was tautological on first integration.** The `lime_builder()` function
  wired into `run_pipeline.py`'s Phase 6 (and `explanation_stability.py`'s own standalone entry
  point) initially passed `random_state=config.RANDOM_SEED` to every "fresh" `LimeTabularExplainer`
  it built - fine for `explain.py`'s one-shot Phase 6 explanation, where reproducibility is the
  goal, but wrong for a *stability* test, where a fixed seed makes every rerun produce identical
  perturbations and therefore guarantees zero measured variance regardless of whether LIME is
  actually stable. Caught by reviewing the first real run's output (Jaccard = 1.000, std ≈ 1e-17 -
  suspiciously perfect) rather than accepting a clean-looking number at face value. Fixed by
  removing the fixed seed from the stability-test builder specifically; the real run afterwards
  showed genuine variability (see Results below).

- **Two new tests were silently overwriting real results with synthetic-data output.**
  `test_repeated_cv_baselines_returns_one_row_per_model` and
  `test_run_chronological_robustness_test_returns_all_baselines` called functions that
  unconditionally write to `config.RESULTS_METRICS_DIR` (`repeated_cv_baselines()` and
  `run_chronological_robustness_test()`) without monkeypatching that path to a temp directory
  first, unlike every other test in this project. Running the test suite therefore clobbered the
  real `baseline_repeated_cv_per_fold.csv`/`_summary.csv` and `chronological_split_robustness.csv`
  with results from a ~440-row synthetic fixture - which produced catastrophic-looking numbers
  (F1 ~0.08, negative MCC) that were briefly mistaken for a real finding before the cause was
  tracked down by manually replicating the CV loop fold-by-fold against the real data (all 15
  folds came back fine, matching the headline results) and then diffing the row count in the
  saved CSV (6 rows - matching the synthetic fixture's `n_splits=2, n_repeats=1` - instead of the
  expected 45). Both tests now monkeypatch `RESULTS_METRICS_DIR` like everywhere else, and the
  real files were regenerated. Worth noting for anyone extending this project: any new test that
  calls a function persisting to `results/` needs this monkeypatch, and there's no test that
  currently guards against forgetting it.

### Added — reproducibility and cross-platform installation

- `requirements.txt` is now **fully pinned** to the exact versions that produced the committed
  results (numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.2.0, tensorflow 2.21.0,
  imbalanced-learn 0.14.2, shap 0.51.0, lime 0.2.0.1, and others) - previously unpinned, so a
  future install could silently resolve different versions and produce different results.
- `jupyter` moved out of `requirements.txt` into a new optional `requirements-dev.txt`. On
  Windows, `jupyter` pulls in `pywinpty` as a transitive dependency (via `terminado`), which has
  no macOS/Linux equivalent and previously broke installation there. `requirements.txt` no longer
  has this problem; `requirements-dev.txt` is only needed to run `notebooks/01_eda.ipynb`
  interactively, not to run the pipeline or test suite.
- `requirements-lock.txt` (a `pip freeze` snapshot that included the same Windows-only packages)
  removed - superseded by the now fully-pinned `requirements.txt`, avoiding two pinned-dependency
  files that could drift out of sync.
- README gains a "Supported platforms" section documenting the above, plus the macOS XGBoost
  prerequisite (`brew install libomp`, needed for OpenMP support that isn't bundled by default).

### Added — statistical robustness layer (`src/robust_evaluation.py`)

- `repeated_cv_baselines()` — 5×3 repeated stratified cross-validation for the three baseline
  models (not RXT — its own 5-fold CV from Phase 4 already serves this purpose, and repeating its
  ~20+ minute CPU training time across dozens of CV folds isn't practical). Each fold tunes its
  threshold on an inner train/validation split, matching the headline protocol rather than a
  looser one.
- `bootstrap_metric_ci()` / `bootstrap_all_models()` — 95% bootstrap confidence intervals for F1
  and PR-AUC on each model's already-saved test predictions (no retraining needed).
- `calibration_report()` / `calibration_report_all()` — reliability diagrams and Brier scores for
  every model, checking whether the class-weighting-driven threshold distortion this project's
  fixed-threshold finding already identified shows up independently as poor calibration.

### Added — operational relevance layer (`src/operational_evaluation.py`)

- `precision_recall_at_k()` — precision/recall at a fixed alert capacity (top 50/100/500 highest-
  scored transactions), independent of the tuned classification threshold.
- `cost_sensitivity_sweep()` — sweeps an assumed false-negative cost against a fixed false-
  positive cost, finding the cost-minimising threshold at each level. The assumed costs are
  explicitly illustrative - this dataset carries no real investigation/recovery-cost data.
- `run_chronological_robustness_test()` — trains and evaluates every baseline on a time-ordered
  split (train on the earlier portion of the two-day collection window, test on the later
  portion) using the identical modelling/thresholding protocol as the headline random-split
  result, as a robustness check rather than a replacement for it.

### Added — explanation stability testing (`src/explanation_stability.py`)

- `lime_stability_test()` — reruns LIME's `explain_instance` on the same transaction multiple
  times (fresh explainer each time) and reports mean pairwise Jaccard similarity of the top-k
  attributed features, plus per-feature weight mean/std, across reruns.
- `shap_kernel_stability_test()` — the same idea for RXT's SHAP KernelExplainer (sampling-based,
  unlike TreeExplainer for the baselines, which is exact and excluded from stability testing since
  its variance is zero by construction).
- Wired into `run_pipeline.py`'s Phase 6 (run for Random Forest and RXT), and independently
  runnable via `python -m src.explanation_stability`, which reloads the already-persisted models
  from `results/models/` rather than requiring a full pipeline run.

### Results (real run against the full dataset)

- **Repeated CV (`baseline_repeated_cv_summary.csv`, 5×3 = 15 folds per model):** F1 mean±std -
  Logistic Regression 0.764±0.046, Random Forest 0.836±0.037, XGBoost 0.842±0.033. Consistent with
  and close to the single-split headline numbers, with tight standard deviations - the baselines'
  performance is stable across resamples, not an artefact of one lucky split.
- **Bootstrap 95% CIs (`bootstrap_confidence_intervals.csv`, 2000 resamples):** F1 - Random Forest
  0.866 [0.814, 0.916], XGBoost 0.853 [0.798, 0.905], Logistic Regression 0.806 [0.739, 0.867],
  RXT 0.494 [0.398, 0.582]. **Random Forest's and RXT's F1 confidence intervals do not overlap at
  all** (RF's lower bound 0.814 is well above RXT's upper bound 0.582) - "the tree ensembles beat
  RXT" is a statistically robust claim under this test, not just a single-split result.
- **Calibration (`calibration_summary.csv`):** Brier scores - Random Forest 0.0004, XGBoost 0.0004,
  Logistic Regression 0.0224, RXT 0.0434. The two models whose fixed-threshold behaviour was worst
  before the threshold-tuning fix (Logistic Regression, RXT) also have the worst calibration here,
  by roughly two orders of magnitude versus the tree ensembles - consistent with, and independent
  evidence for, this project's earlier threshold-tuning finding.
- **Precision at fixed alert capacity (`precision_at_k.csv`):** at k=50, recall is Random Forest
  0.49, XGBoost 0.51, Logistic Regression 0.41, RXT 0.32 - RXT trails every baseline at every
  alert-capacity level tested (k=50/100/500), the same ordering as the headline F1 comparison.
- **Explanation stability (`lime_stability_*.csv`, `shap_stability_rxt.csv`), for one
  correctly-flagged fraud case per model:** LIME top-5-feature Jaccard stability - Random Forest
  0.861, RXT 0.937. **This is the opposite of what might be expected** (a plain GRU-style
  architecture being less stable than a tree ensemble) - on this specific case, RXT's LIME
  explanation was *more* stable, not less. RXT's SHAP KernelExplainer values for its top features
  show coefficient-of-variation around 0.3-0.9 (std is 30-90% of the mean attribution) - real,
  worth-reporting imprecision, but the direction of the LIME finding should be reported honestly
  rather than assumed.
- **Chronological split (`chronological_split_robustness.csv`):** F1 - XGBoost 0.827, Random
  Forest 0.803, Logistic Regression 0.719 (RXT not included - this check only covers the three
  baselines). Lower than the random-split headline numbers for the same models, and with far fewer
  fraud cases in the time-ordered test window (75 vs 98) - some sign that fraud patterns are not
  perfectly stationary across the two-day collection window, worth discussing as a generalisation
  caveat rather than treating the random-split result as the last word.

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
