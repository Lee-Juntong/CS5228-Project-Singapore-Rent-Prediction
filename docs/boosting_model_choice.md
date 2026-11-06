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

Default LightGBM, CatBoost and XGBoost (library defaults, native categorical
handling), G0+G1, raw target.

| model | time RMSE | MAE | MAPE |
|---|---|---|---|
| catboost_default | 452.7 | 330.2 | 12.5% |
| lightgbm_default | 457.3 | 334.7 | 12.6% |
| xgboost_default | 457.2 | 335.4 | 12.6% |

For reference on the same split: ridge 574.9, global-median floor 837.1. All
three boosting families clear the linear and naive baselines by a wide margin.
LightGBM and XGBoost start about level (~457); CatBoost is ~5 RMSE ahead out of
the box.

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
§2). Tuning cut LightGBM 457.3 → 449.9, CatBoost 452.7 → 448.8, and XGBoost
457.2 → 449.9 on G0+G1/raw.

Split comparison (G0+G1, raw) — see `optimism` for generalization behaviour:

| model | time | random5 all | random5 window | optimism |
|---|---|---|---|---|
| catboost_tuned | 448.8 | 475.4 | 451.0 | -2.1 |
| lightgbm_tuned | 449.9 | 478.8 | 451.9 | -2.0 |
| xgboost_tuned | 449.9 | 477.4 | 450.8 | -0.9 |
| catboost_default | 452.7 | 476.4 | 451.6 | 1.2 |
| lightgbm_default | 457.3 | 478.9 | 455.3 | 2.0 |
| xgboost_default | 457.2 | 479.9 | 456.5 | 0.7 |
| ridge_default | 574.9 | 509.2 | 501.6 | 73.3 |
| global_median | 837.1 | 752.0 | 794.6 | 42.5 |

`optimism` = time − random-CV-on-the-same-holdout-rows. It measures how much a
random (leaky) split flatters the model; since the real test is strictly future,
low optimism means the honest score will hold up. All three tuned boosting
families sit at ≈ 0 (even slightly negative) — they do not rely on seeing future
rows, unlike Ridge (+73), which extrapolates a straight trend and overshoots.

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

The feature-group ablation was run for **LightGBM and CatBoost only**. XGBoost
was added later and run only on the best LightGBM/CatBoost combination (G0+G1),
to compare model families — not swept across feature groups — so we make no
claim about which feature groups matter for XGBoost. The G1-helps / G2-G3-don't
pattern is consistent across the two families we did ablate.

**Why the full 4-way ablation was run only on `raw`, and only G0+G1 and
G0+G1+G2 for `index`:** the ablation's job is to rank *feature groups*, and the
trend handling (`raw` vs `index`) is an independent axis — running both on all four
feature sets would be 8 configs/model for no extra insight. `index` was added
only on the two feature sets still in contention for the final model (G0+G1 and
G0+G1+G2), since G0 and ALL were already losing on `raw`. So the trend-handling
comparison is a focused 2×2, not a full grid.

Trend handling (raw vs index) on the contending feature sets:

| model | feature set | raw | index |
|---|---|---|---|
| xgboost_tuned | G0+G1 | 449.9 | **445.6** |
| lightgbm_tuned | G0+G1 | 449.9 | 445.8 |
| lightgbm_tuned | G0+G1+G2 | 449.8 | 446.2 |
| catboost_tuned | G0+G1 | 448.8 | 449.0 |
| catboost_tuned | G0+G1+G2 | 448.5 | 448.4 |

`index` (train on rent ÷ monthly rent index, multiply back) helps LightGBM and
XGBoost ~4 RMSE but barely moves CatBoost. The index is the **training median
rent per month**, extrapolated to future months by a line through the last 12
monthly medians (leakage-safe). It matters because tree models cannot
extrapolate a trend on their own — the index supplies it. XGBoost on index
(445.6) edges LightGBM (445.8) by 0.2 — a tie within noise (see §5).

---

## 5. Final single model choice

The team's final model is now the five model repeat-rent blend (holdout 431.0,
public LB 415.01), which includes this LightGBM as one of its members. See
[repeat_rent_blend.md](repeat_rent_blend.md). This section records the choice of
the best single boosting model.

**lightgbm_tuned, G0+G1, index target — time-holdout RMSE 445.8.**

Reasoning:

- **Effectively the best honest RMSE.** The top three index configs are
  xgboost_tuned 445.6, lightgbm_tuned 445.8, catboost_tuned (G0+G1+G2) 448.4.
  XGBoost edges LightGBM by 0.2 — a tie well inside the ~0.4 leaderboard noise
  band — so there is no meaningful accuracy difference between the two best.
- **Why LightGBM over the fractionally-better XGBoost.** 0.2 RMSE is not a real
  improvement, and LightGBM was already chosen, submitted (public LB 434.4),
  weight-saved and documented. Switching for a within-noise difference adds risk
  without benefit. XGBoost's value is as a *third family that confirms the ~445
  result*, not as a reason to re-pick. (We did submit XGBoost as a comparison
  data point — see §7.)
- **Trend handling fits the task.** The Kaggle test spans Mar 2025–Jul 2026,
  further into the future than our holdout. Index mode is the only mechanism that
  lets a tree model follow a continuing rent trend, so its edge should be at
  least as large on the real horizon as on the holdout.
- **G0+G1, not G0+G1+G2.** Adding block features (G2) did not help (it slightly
  hurt index-mode LightGBM: 445.8 → 446.2), so the simpler feature set wins.
- **Low optimism.** Tuned LightGBM's optimism ≈ −2 says the honest score is not
  an artifact of the split, so we expect it to generalize.

The bigger takeaway: **all three boosting families converge to ~445–450 on the
index target**. This shows the result is not specific to one library — the
choice of boosting implementation no longer moves the score — so further gains
would have to come from features or data, not from swapping boosting models. (It
does not by itself establish an absolute error floor; the noise-floor analysis
speaks to that separately.)

Submission: `submissions/lightgbm_tuned_G0G1_index.csv` (50k rows, Id/Predicted),
logged in `results/kaggle_log.csv`. Weights saved for replay at
`models/lightgbm_tuned_G0G1_index.joblib` (gitignored; only on member C's computer).

Rebuild with:
```sh
python -m scripts.save_final_model --model lightgbm_tuned --features G0+G1 --target-mode index
```

---

## 6. Suspicious-row ablation

EDA (`preprocessing/findings.md`, `decisions.md`, `correlations.ipynb`) flags
rows that are individually plausible but inconsistent with each other, using two
methods. The team decided to keep + flag them and let a with/without RMSE test
decide (`decisions.md`, "Outlier handling"); this section runs that test.

| flag | definition | catches | count (our data) |
|---|---|---|---|
| **M1** (`suspect_m1`) | rent extreme *within* its FLAT_TYPE and FLAT_TYPE×TOWN group (IQR fence AND \|z\|>3) | "wrong price for this *type*" (cheap executive, pricey 1-room) | 331 |
| **M2 low** (`suspect_rent_low_for_area`) | `rent ~ FLOOR_AREA_SQM` degree-1 fit, residual z < −3 | "big flat, tiny rent" — error candidates | 49 |
| **M2 high** (`suspect_rent_high_for_area`) | same fit, residual z > +3 | "expensive for area" — usually legit prime-location premiums | 506 |
| **both** (`suspect_both`) | M1 ∩ M2 low | flagged by both → high-confidence errors | 12 |

> **Update 2026-11-06.** The runs below estimated the flag thresholds from all labelled rows, so validation rents shaped which fit rows were dropped. `scripts/suspicious_row_ablation.py` now estimates them from the fit window only. Fit-only results (36 / 374 / 9 rows dropped for M2 low / M1 / both, val RMSE 446.2 / 449.1 / 446.2) are in `results/member_b/extensions/outliers_fit_only.csv`. The public LB runs train on all labelled rows, so their thresholds are unaffected.

We drop only the flagged rows in the **fit window** (≤ Sep 2024), so the Oct
2024–Mar 2025 validation set is identical to the baseline runs and the RMSE delta
reflects only the cleaner training data. All runs use the final single model
(lightgbm_tuned, G0+G1). Reproduce with
`python -m scripts.suspicious_row_ablation --flag {m2,both,m1}`.

### M2 low — "big flat, tiny rent" (49 rows)

The `suspect_rent_low_for_area` subset (e.g. a 176 sqm executive flat at $750).
Reproducing the flag on our 149,575-row training frame (the EDA ran on the raw
150k) gives **49 rows**, of which **44 fall in the fit window** and 5 in the
validation window; we drop the 44 fit-window rows.

| target | with suspect rows (baseline) | without (44 dropped) | Δ |
|---|---|---|---|
| raw | 449.9 | 450.1 | +0.2 (worse) |
| index | 445.8 | 445.7 | −0.1 (better) |

**Verdict: no meaningful effect** (±0.1–0.2 RMSE, within run-to-run noise, and
it even slightly *hurts* the raw target). This confirms the EDA hypothesis that
tree models are robust to these ~50 outliers, so removal buys nothing.
**Decision: keep the rows** — consistent with the preprocessing decision, now
backed by the A/B. We also submitted the without-suspect model to Kaggle: public
LB 434.0 vs 434.4 with the rows (§7) — the leaderboard confirms the holdout verdict.

### M1 — rent extreme within its group (331 rows)

The `suspect_m1` subset: rent beyond the IQR fence *and* |z|>3 within its
FLAT_TYPE or FLAT_TYPE×TOWN group. Unlike M2 low, this is *symmetric* — it flags
both too-cheap and too-expensive rows. 331 rows, 276 in the fit window.

| target | with suspect rows (baseline) | without (276 dropped) | Δ |
|---|---|---|---|
| raw | 449.9 | 452.6 | +2.7 (worse) |
| index | 445.8 | 447.9 | +2.1 (worse) |

**Removing M1 rows clearly hurts** (+2–3 RMSE on both targets) — the most
decisive of the three ablations. M1 sweeps in *legitimate* extremes, notably the
`central` 5-room premiums the EDA flagged as real signal (not errors), so
dropping them throws away information the model uses. This is the cautionary
case: blunt within-group outlier removal is actively harmful here. The Kaggle
leaderboard confirms it — the M1-removed model scores 438.6 vs 434.4 with the
rows (+4.2, ~1%; §7), worse on the true test horizon too, and ~10× the ~0.5
noise band seen in the M2-low submission.

### both — high-confidence rows (M1 ∩ M2 low, 12 rows)

Rows flagged by **both** methods (wrong for their *type* and wrong for their
*size*, in the error direction). Our data has **12** such rows (10 in the fit
window). The notebook's "243" counts M1 ∩ (M2 low ∪ M2 high), but ~230 of those
are M1 ∩ *high* — the "expensive for area" premiums the EDA itself flags as
legitimate ("keep") — so the 12 low-side rows are the genuine high-confidence
error candidates.

| target | with suspect rows (baseline) | without (10 dropped) | Δ |
|---|---|---|---|
| raw | 449.9 | 449.9 | ~0 |
| index | 445.8 | 446.1 | +0.3 (worse) |

**Same verdict — no benefit; keep the rows.** Two independent outlier
definitions both say the data's statistical outliers are not worth removing for
a tree model.

**Summary across the three ablations** (lightgbm_tuned, G0+G1 index target):

| dropped | rows | val RMSE | public LB | vs baseline |
|---|---|---|---|---|
| — (baseline) | 0 | 445.8 | 434.4 | — |
| M2 low | 49 | 445.7 | 434.0 | ~0 (noise) |
| both (M1 ∩ M2) | 12 | 446.1 | — | +0.3 val |
| M1 | 331 | 447.9 | 438.6 | **+2.2 val / +4.2 LB** |

Removing flagged rows ranges from harmless (M2 low, both) to clearly harmful
(M1), and never helps — so we **keep all rows**. The M1 result shows *why*: its
symmetric within-group rule also removes legitimate premium rents, which the
tree model relies on. The harm is modest in absolute terms (~1% RMSE) but
consistent across *both* the holdout and the Kaggle test, and ~10× the ~0.5
leaderboard noise band — so it is a real effect, not run-to-run variation.

---

## 7. Kaggle Results

### Tue 27 Oct 2026 11:09:34 — lightgbm_tuned, G0+G1, index

| field | value |
|---|---|
| model | lightgbm_tuned, G0+G1, index target |
| submission file | `submissions/lightgbm_tuned_G0G1_index.csv` |
| time-holdout RMSE (val) | 445.8 |
| public LB RMSE | **434.4** |

Public LB (434.4) comes in about 11 better than the 445.8 validation estimate.
That is reassuring rather than suspicious: the validation window (Oct 2024–Mar
2025) and the public test rows are different months, and the model generalized at
least as well on the test data. It also confirms the model is not overfit to the
validation split (consistent with the near-zero optimism in §3).

### Tue 27 Oct 2026 — lightgbm_tuned, G0+G1, index, without M2-low rows

| field | value |
|---|---|
| model | lightgbm_tuned, G0+G1, index, trained without the 49 M2-low (`suspect_rent_low_for_area`) rows |
| submission file | `submissions/lightgbm_tuned_G0G1_index_nosuspect.csv` |
| time-holdout RMSE (val) | 445.7 |
| public LB RMSE | **434.0** |

Dropping the M2-low rows gives public LB 434.0 vs 434.4 with them — 0.45 apart,
i.e. noise. The leaderboard confirms the §6 ablation: the ~50 "big flat, tiny
rent" rows do not matter either way, so we keep them (simpler, matches the
preprocessing decision).

### Wed 28 Oct 2026 — lightgbm_tuned, G0+G1, index, without M1 rows

| field | value |
|---|---|
| model | lightgbm_tuned, G0+G1, index, trained without the 331 M1 rows |
| submission file | `submissions/lightgbm_tuned_G0G1_index_nom1.csv` |
| time-holdout RMSE (val) | 447.9 |
| public LB RMSE | **438.6** |

Removing the within-group rent extremes scores 438.6 vs 434.4 with them — +4.2
(~1%), worse on the test horizon just as on the holdout (+2.2), and ~10× the
noise band the no-suspect submission established (0.45). Modest in absolute terms
but a real, consistent degradation — the strongest evidence that blunt extreme-row
removal hurts, because M1 discards legitimate premium rents (e.g. central 5-room).
The prediction range also compressed (max 5637 → 5073), the model losing its grip
on high rents.

### Sat 1 Nov 2026 — xgboost_tuned, G0+G1, index (third-family comparison)

| field | value |
|---|---|
| model | xgboost_tuned, G0+G1, index target |
| submission file | `submissions/xgboost_tuned_G0G1_index.csv` |
| time-holdout RMSE (val) | 445.6 |
| public LB RMSE | **434.5** |

Not the final model — submitted only to place the third boosting family on the
leaderboard next to LightGBM (434.4). XGBoost scores 434.5 vs LightGBM's 434.4 —
**0.05 apart**, so the two best models are indistinguishable on the test set, not
just the holdout. This confirms the §5 takeaway on real data: all three boosting
families land in the same place (~445 holdout / ~434 LB), so the choice of
boosting library is noise — further gains would need better features or data,
not a different boosting model.
