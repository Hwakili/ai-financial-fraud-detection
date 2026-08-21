## Results (real run against the full dataset, post threshold-fix)

Held-out test set (stratified 80/20 split), full `results/metrics/model_comparison.csv`:

| Model                | Precision | Recall | F1        | MCC       | AUC-ROC | Threshold |
| --------------------- | --------- | ------ | --------- | --------- | ------- | --------- |
| Random Forest         | 0.910     | 0.827  | **0.866** | 0.867     | 0.957   | 0.500     |
| XGBoost                | 0.880     | 0.827  | 0.853     | 0.853     | 0.976   | 0.500     |
| Logistic Regression    | 0.852     | 0.765  | 0.806     | 0.807     | 0.974   | ~1.000    |
| RXT (ResNeXt-GRU)      | 0.566     | 0.439  | 0.494     | 0.497     | 0.959   | 0.980     |

**The threshold fix substantially changed the picture.** Before tuning the classification
threshold on a validation set (see `select_threshold()` in `src/evaluate.py`), a fixed 0.5
cutoff combined with aggressive class weighting collapsed precision for both Logistic
Regression (0.061) and RXT (0.154). With the fix applied, Logistic Regression's precision
rose to 0.852 — now the second-best model overall — confirming that the earlier "tree
ensembles uniquely handle imbalance" narrative was an artefact of the broken threshold,
not a real property of the models. XGBoost and Random Forest remain the strongest
performers, but the margin over a correctly-tuned linear baseline is far smaller than the
original (buggy) results suggested.

**RXT still underperforms — and the reason is worth investigating further, not just
reporting.** Even after the fix, RXT's held-out F1 (0.494) trails every baseline. Notably,
this single held-out score does not agree with the model's own 5-fold cross-validation
result (`results/metrics/rxt_kfold_summary.csv`): mean F1 across folds is **0.728 (± 0.052)**,
nearly 25 points higher than the held-out figure. This gap suggests RXT's performance on
this dataset is *unstable* across data splits, not simply *lower* than the baselines —
plausibly a consequence of the small absolute number of fraud examples (~394 in a typical
training fold) available for a deep architecture to learn from. Both numbers are reported
here rather than only the more favourable one; the instability itself is treated as a
finding, not noise to be averaged away.

**One additional observation:** Logistic Regression's tuned threshold is effectively 1.0
(0.9999999993), indicating its raw probability outputs are heavily compressed toward the
top of the range under `class_weight='balanced'`. This is a probability-calibration
artefact of the class-weighting approach, not evidence that only near-certain predictions
are useful — it means LR's *raw* probabilities aren't well-calibrated, even though its
*ranking* of transactions (reflected in AUC-ROC = 0.974) is strong.

## Computational efficiency

| Model               | Train time | Inference (single transaction) | Parameters |
| -------------------- | ---------- | ------------------------------- | ---------- |
| Logistic Regression   | 1.3s       | 1.5ms                           | —          |
| XGBoost                | 1.7s       | 4.7ms                            | —          |
| Random Forest          | 12.8s      | 32.3ms                          | —          |
| RXT (ResNeXt-GRU)      | **1,263.5s (~21 min)** | 69.1ms          | 38,145     |

RXT costs roughly 100–1,000x more to train than any baseline, for meaningfully worse
held-out predictive performance. Combined with the held-out/cross-validation instability
above, this raises a real question about whether the added architectural complexity of
ResNeXt-embedded GRU is justified on this dataset relative to a correctly-tuned XGBoost
or Random Forest — a question this project treats as a substantive finding in its own
right (see Section 2.5.3 / 4.7 of the dissertation for the feasibility discussion that
anticipated this outcome).

All figures (confusion matrices, ROC/PR curves, grouped metric comparison, SHAP summaries,
LIME explanations) are in `results/figures/`; efficiency numbers are in
`results/metrics/efficiency_comparison.csv`.