# Final blend reproduction

Completed on 5 November 2026 against main commit `17d942963204a44fec97a4985eb405860bd0b09e` (PRs #6 and #8 included), in a separate Windows checkout. The blend implementation, hyperparameters and input data were unchanged.

## Environment and commands

Python 3.10.22, with every version in the root `requirements.txt` verified, including LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.8 and scikit-learn 1.5.1. Additional analysis dependencies matched `requirements-analysis.txt`. `pip check` passed. Exact versions and input SHA-256 hashes are in [verification.json](../results/member_b/blend_reproduction/verification.json).

From the repository root, after placing the raw competition files as described in [data setup](../data/README.md):

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s checks -v
python -c "from src.data_preprocessing import run_pipeline; run_pipeline()"
python -m scripts.lag_blend_submission --validate
python -m scripts.lag_blend_submission
python -m scripts.lag_blend_submission --analyse
python -m scripts.trend_analysis
python -m scripts.horizon_backtest --table-only
python -m scripts.report_tables
```

Then run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from `report/`.

All 39 unit tests passed. Fresh preprocessing produced 149,575 labelled rows and 50,000 test rows. No tuning was run and no baseline or backtest models were refitted. The backtest command above rebuilds its table from saved results.

## Numerical agreement

| Five model blend | Reference RMSE | Reproduced RMSE |
| --- | ---: | ---: |
| October 2024 to March 2025 holdout | 430.973558 | 430.973558 |
| December 2023 to March 2025 backtest | 492.767291 | 492.767291 |

All keys, row counts and text fields matched across the six generated CSVs. Maximum absolute differences over their numeric cells were:

| File in `results/member_a/repeat_rent/` | Maximum difference |
| --- | ---: |
| `validation.csv` | 7.11e-15 |
| `limit.csv` | 0 |
| `gap_correlation.csv` | 0 |
| `test_lag_coverage.csv` | 0 |
| `error_analysis.csv` | 2.27e-13 |
| `permutation_importance.csv` | 3.55e-14 |

Every difference was below 1e-10. The committed reference results are kept, since differences at this scale are floating point rounding.

The three blend modes performed 24 fits in total. On this machine, validation took 264.4 seconds, prediction generation 108.2 seconds, and analysis 135.4 seconds. These are command wall times, including feature construction, prediction and scoring, not individual model fit times.

## Submission file

The local `submissions/lag_blend5_G0G1_index.csv` has columns `Id,Predicted` and finite predictions throughout. Predicted rent ranges from 1,527.26 to 5,335.60 SGD, with mean 3,180.48 SGD. All three summary values match the uploaded submission to full precision.

PR #10 added the uploaded CSV to main. We compared the saved output from this run (`17d9429`) with the Git copy at `547d891`, matching rows by `Id` and parsing `Predicted` as double precision. Both files contain the same 50,000 unique IDs. Of those, 7,152 predictions differ numerically. The largest absolute difference is 9.09e-13 SGD, so all 50,000 pass the 1e-9 SGD tolerance. The verification JSON records both file hashes.

On another Windows PC, a second run at `547d891` also passed this comparison, with 12.2% of predictions differing by at most 1.8e-12 SGD. The public score of 415.01 comes from the original Kaggle submission.

## Report checks

In the run above, the seven regenerated tables (`results`, `ablation`, `blend`, `trend`, `split`, `hyperparameters`, `horizon`) are byte identical to their committed references. The report compiled without unresolved references or overfull boxes. Evaluation tables and regenerated figures were visually checked.

After the merge with main and the figure fix (`31828a9`), the PDF has 11 pages in total. The conclusion ends on page 6. The remaining main text figures are on pages 7 and 8. References and appendices start on page 9, so the main content is within the eight page limit.

After later report changes, the build on 6 November 2026 has 12 pages. The conclusion is on page 8 and references start on page 9, so the main content is still within the limit.
