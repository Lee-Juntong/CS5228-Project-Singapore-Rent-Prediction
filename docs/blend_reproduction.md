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

All 39 unit tests passed. Fresh preprocessing produced 149,575 labelled rows and 50,000 test rows. No tuning or historical baseline/backtest refitting was performed; the backtest command above rebuilds its table from saved results.

## Numerical agreement

| Five-model blend | Reference RMSE | Reproduced RMSE |
| --- | ---: | ---: |
| October 2024-March 2025 holdout | 430.973558 | 430.973558 |
| December 2023-March 2025 backtest | 492.767291 | 492.767291 |

All keys, row counts and text fields matched across the six generated CSVs. Maximum absolute differences over their numeric cells were:

| File in `results/member_a/repeat_rent/` | Maximum difference |
| --- | ---: |
| `validation.csv` | 7.11e-15 |
| `limit.csv` | 0 |
| `gap_correlation.csv` | 0 |
| `test_lag_coverage.csv` | 0 |
| `error_analysis.csv` | 2.27e-13 |
| `permutation_importance.csv` | 3.55e-14 |

Every difference was below 1e-10. The committed reference results are retained; differences at this scale are floating-point rounding.

The three blend modes performed 24 fits in total. On this machine, validation took 264.4 seconds, prediction generation 108.2 seconds, and analysis 135.4 seconds. These are command wall times, including feature construction and prediction/scoring, not individual model fit times.

## Submission file

The local `submissions/lag_blend5_G0G1_index.csv` has columns `Id,Predicted`, exactly 50,000 complete unique test IDs in order, and finite predictions throughout. Predicted rent ranges from 1,527.26 to 5,335.60 SGD, with mean 3,180.48 SGD. Its SHA-256 is recorded in the verification JSON.

The original uploaded submission file was unavailable, so row-by-row equality with that file was not checked. No Kaggle upload was performed; the recorded public score of 415.01 was not independently remeasured. The blend script does not save its fitted estimators.

## Report checks

The seven regenerated tables (`results`, `ablation`, `blend`, `trend`, `split`, `hyperparameters`, `horizon`) are byte-identical to their committed references. The report compiles without unresolved references or overfull boxes. Evaluation tables and regenerated figures were visually checked.

The PDF has 10 pages in total: the conclusion ends on page 6, remaining main-text figures are on page 7, and references/appendices follow. The main content is within the eight-page limit excluding references and appendices. Three student-ID placeholders and the workload/preprocessing-attribution TODO still require the team's confirmation.

The README now gives the final blend commands and distinguishes the single-LightGBM checkpoint example. Evaluation wording identifies the suspicious-row experiment as row-removal ablation and marks logged training times as hardware dependent.
