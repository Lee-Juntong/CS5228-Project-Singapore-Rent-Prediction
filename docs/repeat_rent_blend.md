# Repeat-rent blend

Rents in the same block and flat type line up more strongly at gaps of 12, 24 and 36 months, which looks like flats renewing. This method adds that signal as lag features, then blends models with and without them.

## Features

`lag_feats` in `scripts/lag_blend_submission.py` works on rent divided by the monthly rent index. For each row it takes earlier rows in the same block, street and flat type, and only uses rows that are in the training period. It records the mean, count, minimum and maximum at gaps of 11 to 13, 23 to 25 and 35 to 37 months, plus one comparison set of other gaps between 6 and 30 months.

## Results

| Model | Oct 2024 to Mar 2025 holdout | 16 month backtest | Public LB |
|---|---|---|---|
| Tuned LightGBM, index | 445.8 | 511.0 | 434.43 |
| Tuned LightGBM + regularised LightGBM with lags | 432.6 | 495.3 | 419.71 |
| Five model blend (adds XGBoost, CatBoost, random forest with lags) | 431.0 | 492.8 | 415.01 |
| Base and lag LightGBM on index and raw targets | 431.6 | 485.6 | 419.59 |

The blend was chosen on these same two windows, so its gains are slightly optimistic. The index and raw mix had the better 16 month score but the worse public score, so we keep the index target.

## Limit on any model

Rows with the same block, flat type and month have identical features, so any model gives them the same prediction. Even with perfect means per group, RMSE stays at about 178 on the training data (165 to 187 by window). Rents within such a group differ by about 450, so an honest score near that bound is not reachable.

## Run

```bash
python scripts/lag_blend_submission.py
```

This writes `submissions/lag_blend5_G0G1_index.csv`.
