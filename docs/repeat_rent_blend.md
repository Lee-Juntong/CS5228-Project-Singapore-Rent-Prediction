# Repeat-rent blend

Rents in the same block and flat type line up more strongly at gaps of 12, 24 and 36 months than at nearby gaps, most of all at 24 months (`gap_correlation.csv`). This looks like flats renewing, although the data has no unit number to confirm it. This method adds that signal as lag features, then blends models with and without them. It is the final model, and `lag_blend5_G0G1_index.csv` is the final Kaggle submission.

## Features

`lag_features` in `scripts/lag_blend_submission.py` works on rent divided by the monthly rent index. For each row it takes earlier rows in the same block, street and flat type, and only uses rows that are in the training period. It records the mean, count, minimum and maximum at gaps of 11 to 13, 23 to 25 and 35 to 37 months, plus one comparison set of other gaps between 6 and 30 months, which gives 16 features. Test rows see training rows only, so the 12 month set is empty for the last three test months (`test_lag_coverage.csv`), while the 24 month set still covers about 75% of rows.

## Results

| Model | Oct 2024 to Mar 2025 holdout | 16 month backtest | Public LB |
|---|---|---|---|
| Tuned LightGBM, index | 445.8 | 511.0 | 434.43 |
| Two model blend: tuned LightGBM + regularised LightGBM with lags | 432.6 | 495.3 | 419.71 |
| Five model blend (adds XGBoost, CatBoost, random forest with lags) | 431.0 | 492.8 | 415.01 |
| Base and lag LightGBM on index and raw targets | 431.6 | 485.6 | 419.59 |

The 16 month window fits to November 2023 and scores December 2023 to March 2025. We chose the blend by its time holdout score, where it is best, and the public score agrees. The blend was designed on these same windows, so its gains are slightly optimistic. The index and raw mix has the better 16 month score but slightly worse holdout and public scores, so we keep the index target.

## Error analysis on the blend

On the holdout the blend has fewer large errors than tuned LightGBM: absolute errors above 1,000 fall from 3.9% to 3.3% of rows, and from 38% to 34% of squared error (`error_analysis.csv`). RMSE falls from 445.8 to 427.8 on rows with unique inputs and from 445.8 to 440.6 on rows that share inputs, because lag features cannot separate rows inside a group. Bias is larger, at -47.0 against -24.5.

The 24 month lags are the second largest input in permutation importance (+43, flat type ordinal +44). The 12 month lags add +5 and the 36 month lags +0.4 (`permutation_importance.csv`). The case review and the comparison with Ridge in [analysis.md](analysis.md) stay on tuned LightGBM.

## Limit on any model

Rows with the same block, flat type and month have identical features, so any model gives them the same prediction. Even with perfect means per group, RMSE stays at 178 on all training rows, 165 on the holdout and 187 on the 16 month window (`limit.csv`). The bound counts the 74% of rows with unique inputs as perfect. Rents within a group have a pooled standard deviation of 449 on the holdout and 468 on all rows, so an honest score near that bound is not reachable.

## Run

Use Python 3.10 with the full root `requirements.txt`. Both raw CSVs and the processed feature files are required. Run from the repository root. The three modes below are separate commands and each trains fresh models:

```bash
python -m scripts.lag_blend_submission
python -m scripts.lag_blend_submission --validate
python -m scripts.lag_blend_submission --analyse
```

The first command writes `submissions/lag_blend5_G0G1_index.csv`. The other two rebuild every number on this page except the public leaderboard scores. They write the numbers to [results/member_a/repeat_rent/](../results/member_a/repeat_rent/) and the figures to `report/figures/lag_correlation.pdf` and `report/figures/blend_importance.pdf`.

The commands perform 5, 14 and 5 fits respectively. All three overwrite committed files: the first rewrites the committed final submission, and the other two rewrite the CSVs and figures above. Use a separate checkout with its own copy of the data, or restore the files afterwards as shown in [the README](../README.md#kaggle-submission). A rerun of the submission is not byte identical, so compare it with the committed file numerically (maximum difference below 1e-9) and do not commit it. The script does not save model weights, upload predictions to Kaggle or update the Kaggle log.

After regenerating results, run `python -m scripts.report_tables` to rebuild the results, ablation and blend tables, and follow [the report build instructions](../report/README.md). Public leaderboard values are read from the recorded Kaggle log, not recomputed locally.
