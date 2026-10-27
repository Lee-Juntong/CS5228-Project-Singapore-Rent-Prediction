# Boosting model selection (Member C)

How the boosting models (LightGBM, CatBoost) were built, tuned and chosen, with
the results and the reasoning behind each decision. Condensed from the working
log for the whole team.

**Shared setup.** All numbers below are RMSE in SGD on the **time holdout** (fit
Jan 2021–Sep 2024, validate Oct 2024–Mar 2025), the honest forecasting split
from plan decision 1. Seed 5228. Every model goes through the shared
`run_experiment` interface, so these are comparable with the other members' runs.
For context: training rent mean ≈ 2719, std ≈ 750, so RMSE ~450 is about 17% of
mean rent and well under the "always predict the mean" error of ~750.

---

## 1. Baseline models

Default LightGBM and CatBoost (library defaults, native categorical handling),
G0+G1, raw target.

| model | time RMSE | MAE | MAPE |
|---|---|---|---|
| catboost_default | 452.7 | 330.2 | 12.5% |
| lightgbm_default | 457.3 | 334.7 | 12.6% |

For reference on the same split: ridge 574.9, global-median floor 837.1. Both
boosting models clear the linear and naive baselines by a wide margin, and
CatBoost starts ~5 RMSE ahead of LightGBM out of the box.

---

## 2. Hyperparameter tuning

Optuna, TPE sampler, **30 trials per model**, seed 5228. Objective = time-holdout
RMSE, computed through the same `fit_predict` the evaluation uses.

**Why tune on only G0+G1, raw target, time holdout:**

- **Time holdout** — it is the metric we actually report and select on; tuning on
  anything else (e.g. a random split) would optimise for a number we do not trust.
- **raw target** — the headline metric is raw-rent RMSE; `index` is a wrapper
  applied at evaluation time, not a different model, so tuning on raw keeps the
  search aligned with the primary objective.
- **G0+G1** — the Phase 1 feature set. Tuning once on a fixed, representative set
  (rather than per feature set) keeps the 30-trial budget honest and gives one
  canonical config per model. The feature ablation (§4) then reuses that config,
  so each ablation row isolates the *feature* contribution, not a re-tune.

Search spaces (my design choice — conventional ranges, not from a cited source):

- **LightGBM:** n_estimators 200–2000, learning_rate 0.01–0.3 (log), num_leaves
  15–255, min_child_samples 5–100, feature_fraction 0.5–1.0, bagging_fraction
  0.5–1.0, bagging_freq 0–7, reg_alpha / reg_lambda 1e-3–10 (log).
- **CatBoost:** iterations 200–2000, learning_rate 0.01–0.3 (log), depth 4–10,
  l2_leaf_reg 1–20 (log), random_strength 1e-3–10 (log), subsample 0.5–1.0
  (keeps CatBoost's default CPU bootstrap, MVS).

Note: `subsample` replaced `bagging_temperature` after a PR review — the latter
is a no-op under CatBoost's default MVS bootstrap (confirmed empirically), so it
was tuning a dead parameter. Re-tuning with `subsample` improved CatBoost
450.2 → 448.8.

Best params are saved to `configs/<model>_tuned.json`; `make_lightgbm_tuned` /
`make_catboost_tuned` read them back. Caveat for limitations: tree count is tuned
directly (no early stopping), so a trial can pick many trees at a low learning rate.

---

## 3. Tuned model results

The tuned models were run across **feature sets and both target modes** (not
confined to G0+G1/raw — that restriction applied only to the tuning search in
§2). Tuning cut LightGBM 457.3 → 449.9 and CatBoost 452.7 → 448.8 on G0+G1/raw.

Split comparison (G0+G1, raw) — see `optimism` for generalization behaviour:

| model | time | random5 all | random5 window | optimism |
|---|---|---|---|---|
| catboost_tuned | 448.8 | 475.4 | 451.0 | -2.1 |
| lightgbm_tuned | 449.9 | 478.8 | 451.9 | -2.0 |
| catboost_default | 452.7 | 476.4 | 451.6 | 1.2 |
| lightgbm_default | 457.3 | 478.9 | 455.3 | 2.0 |
| ridge_default | 574.9 | 509.2 | 501.6 | 73.3 |
| global_median | 837.1 | 752.0 | 794.6 | 42.5 |

`optimism` = time − random-CV-on-the-same-holdout-rows. It measures how much a
random (leaky) split flatters the model; since the real test is strictly future,
low optimism means the honest score will hold up. The tuned boosting models sit
at ≈ 0 (even slightly negative) — they do not rely on seeing future rows, unlike
Ridge (+73), which extrapolates a straight trend and overshoots.

---

## 4. Feature-group ablation

Tuned hyperparameters held fixed at their G0+G1 values; only the feature set
varies, so each row isolates one feature group. Raw target, time holdout.

| feature set | features | catboost_tuned | lightgbm_tuned |
|---|---|---|---|
| G0 (core) | 13 | 462.7 | 460.2 |
| G0+G1 (+location) | 39 | 448.8 | 449.9 |
| G0+G1+G2 (+block) | 44 | 448.5 | 449.8 |
| ALL (+market) | 54 | 450.1 | 452.4 |

- **G1 location** is the only clear win (~12–14 RMSE).
- **G2 block** adds almost nothing (0.1–0.3).
- **G3 market** (current-month COE, CPI, stocks) *hurts* both models (+1.6–2.6),
  even though those values would not be known at prediction time — supports plan
  decision 6: drop the leaky macro features.

**Why the full 4-way ablation was run only on `raw`, and `index` only on G0+G1
and G0+G1+G2:** the ablation's job is to rank *feature groups*, and the trend
handling (`raw` vs `index`) is an independent axis — running both on all four
feature sets would be 8 configs/model for no extra insight. `index` was added
only on the two feature sets still in contention for the final model (G0+G1 and
G0+G1+G2), since G0 and ALL were already losing on `raw`. So the trend-handling
comparison is a focused 2×2, not a full grid.

Trend handling (raw vs index) on the contending feature sets:

| model | feature set | raw | index |
|---|---|---|---|
| lightgbm_tuned | G0+G1 | 449.9 | **445.8** |
| lightgbm_tuned | G0+G1+G2 | 449.8 | 446.2 |
| catboost_tuned | G0+G1 | 448.8 | 449.0 |
| catboost_tuned | G0+G1+G2 | 448.5 | 448.4 |

`index` (train on rent ÷ monthly rent index, multiply back) helps LightGBM ~4
RMSE but barely moves CatBoost. The index is the training median rent per month,
extrapolated to future months by a line through the last 12 monthly medians
(leakage-safe). It matters because tree models cannot extrapolate a trend on
their own — the index supplies it.

---

## 5. Final model choice

**lightgbm_tuned, G0+G1, index target — time-holdout RMSE 445.8** (the best
single result across all configs above).

Reasoning:

- **Lowest honest RMSE.** 445.8 beats every other config, including tuned
  CatBoost (448.4 best) and tuned LightGBM on raw (449.9).
- **Trend handling fits the task.** The Kaggle test spans Mar 2025–Jul 2026,
  further into the future than our holdout. Index mode is the only mechanism that
  lets a tree model follow a continuing rent trend, so its edge should be at
  least as large on the real horizon as on the holdout.
- **G0+G1, not G0+G1+G2.** Adding block features (G2) did not help (it slightly
  hurt index-mode LightGBM: 445.8 → 446.2), so the simpler feature set wins.
- **Low optimism.** Tuned LightGBM's optimism ≈ −2 says the honest score is not
  an artifact of the split, so we expect it to generalize.

Submission: `submissions/lightgbm_tuned_G0G1_index.csv` (50k rows, Id/Predicted),
logged in `results/kaggle_log.csv`. Weights saved for replay at
`models/lightgbm_tuned_G0G1_index.joblib` (gitignored; only on member C's computer).

Rebuild with:
```sh
python -m scripts.save_final_model --model lightgbm_tuned --features G0+G1 --target-mode index
```
