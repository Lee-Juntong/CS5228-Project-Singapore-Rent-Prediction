# Repeat-rent blend

Rents in the same block and flat type line up more strongly at gaps of 12, 24 and 36 months than at nearby gaps, most of all at 24 months (`gap_correlation.csv`). This looks like flats renewing, although the data has no unit number to confirm it. This method adds that signal as lag features, then blends models with and without them. It is the final model.

## Features

`lag_features` in `scripts/lag_blend_submission.py` works on rent divided by the monthly rent index. For each row it takes earlier rows in the same block, street and flat type, and only uses rows that are in the training period. It records the mean, count, minimum and maximum at gaps of 11 to 13, 23 to 25 and 35 to 37 months, plus one comparison set of other gaps between 6 and 30 months, which gives 16 features. Test rows see training rows only, so the 12 month set is empty for the last three test months (`test_lag_coverage.csv`), while the 24 month set still covers about 75% of rows.

## Results

| Model | Oct 2024 to Mar 2025 holdout | 16 month backtest | Public LB |
|---|---|---|---|
| Tuned LightGBM, index | 445.8 | 511.0 | 434.43 |
| Two model blend: tuned LightGBM + regularised LightGBM with lags | 432.6 | 495.3 | 419.71 |
| Five model blend (adds XGBoost, CatBoost, random forest with lags) | 431.0 | 492.8 | 415.01 |
| Base and lag LightGBM on index and raw targets | 431.6 | 485.6 | 419.59 |

The 16 month window fits to November 2023 and scores December 2023 to March 2025. The blend was chosen on these same windows and on the public score, so its gains are slightly optimistic. The index and raw mix has the better 16 month score and the worse public score, so we keep the index target.

## Limit on any model

Rows with the same block, flat type and month have identical features, so any model gives them the same prediction. Even with perfect means per group, RMSE stays at 178 on all training rows, 165 on the holdout and 187 on the 16 month window (`limit.csv`). The bound counts the 74% of rows with unique inputs as perfect. Rents within a group have a pooled standard deviation of 449 on the holdout and 468 on all rows, so an honest score near that bound is not reachable. Leaderboard scores near 160 to 180 sit at this bound, and we infer that those teams look up rents in the public data.gov.sg table. We did not use that table.

## Run

```bash
python -m scripts.lag_blend_submission
python -m scripts.lag_blend_submission --validate
```

The first command writes `submissions/lag_blend5_G0G1_index.csv`. The second rebuilds every number on this page in [results/member_a/repeat_rent/](../results/member_a/repeat_rent/) and `report/figures/lag_correlation.pdf`.
