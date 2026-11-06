# Baseline extensions and alternative rent index

Run from the repository root in the shared Python 3.10 environment after the standard data and preprocessing steps.

```bash
python -m scripts.baseline_extensions features outliers outliers_fit_only lags horizon
python -m scripts.median_lag_extension
python -m scripts.macro_rent_index
python -m scripts.summarise_baseline_extensions
```

[RESULTS.md](RESULTS.md) contains the comparison tables. Each ablation is also independently runnable by passing its name. A completed row or a saved holdout prediction vector is reused on subsequent runs. These caches are specific to this dataset, split, model registry and experiment settings. Use a fresh output directory when changing any of them. No commands write the original shared results or submission files.

To verify saved results and regenerate the report tables, use the following commands. The first command also needs the local processed training data to audit the outlier thresholds.

```bash
python -m scripts.summarise_baseline_extensions
python -m scripts.report_tables
python -m scripts.report_appendix_tables
```

The per-rental cache `macro_index/ratios.parquet` stays local under the repository's prediction-artifact policy. Running the macro experiment in a fresh checkout reconstructs missing row predictions once. To rerun an ablation with changed data or settings, first move its saved CSV out of this directory. The output directory is defined by `OUT` in `scripts/baseline_extensions.py`.

| File | Protocol |
| --- | --- |
| `features.csv` | Raw rent; 132,538 fit rows, 17,037 validation rows. G0 (13 candidate columns), G0+G1 (39), +G2 (44), then +G3 (54). KNN uses the intersection with its four inputs; G0 therefore uses area and month. Lookup models use the same keys for every group. Later groups do not add KNN inputs. Identical-input rows reference the archived run. G3 contains contemporaneous market indicators that may be unavailable in an operational forecast. |
| `outliers.csv` | Index target and G0+G1. The existing published LightGBM experiment computes flags on all labelled records, then removes only flagged fit-period rows. We reproduce that retrospective rule for comparability. It uses validation-label distribution information in the flag thresholds. Interpret this as a sensitivity analysis, not a prospective cleaning policy. The index and estimator are refitted after each removal. M1 is within-type or within-type-and-town extremes, M2 is an unusually low area-adjusted rent, and both is their intersection. |
| `lags.csv` | Index target, fixed dates. Sixteen features from earlier records of the same block/street/type are computed with the team's lag function. Linear, Ridge, tree and forest use the added columns. KNN expands from 4 to 20 dimensions with the original scaling, imputation, distance weighting and k. Fixed median models ignore additional columns. |
| `outliers_fit_only.csv` | Prospective counterpart to the original ablation. Estimate thresholds from the fit period alone, drop its flagged rows, and retain every validation row. Includes the seven baselines and a matched tuned LightGBM. `flag_threshold_audit.csv` compares flag counts and row disagreements. |
| `median_lag_variants.csv` | Feature-aware alternatives for the two median controls. With n matching rents 23–25 months earlier, predict a weighted mean of the historical mean rent ratio and the original median model ratio, with weights n and 5. A missing match returns the original prediction. No weight search. |
| `horizon.csv` | Seven models × five expanding training origins × two targets × two horizon bands. Observed fit-period medians and a training-only linear forecast for index mode. Reuse the forest rows from Member A; the source records RMSE and bias only. Repeated calendar months across origins are intentional. |

## Official forecasts

[sg_forecast_vintages.csv](../../../data/official_forecasts/sg_forecast_vintages.csv) records the annual real GDP growth forecast issued by Singapore's trade ministry and the headline CPI-All Items inflation forecast from the Monetary Authority of Singapore, in percent. Each row gives the public release date, target year, lower/upper limits and source URL. Single-number forecasts have equal limits. Two MAS governor speeches are sourced from their BIS republications using the speech dates. The series records changes to numerical forecasts and selected reaffirmations. Repeated publications of the same range do not change its midpoint. Qualitative assessments within an unchanged interval are not encoded.

The monthly lookup uses the midpoint of the latest available range for the requested year, with a release date **strictly earlier than the first day of the forecast month**. `macro_index/asof_monthly_forecasts.csv` covers January 2021 through July 2026. The issued-at-origin records in `monthly_holdout.csv` use the additional origin restriction below. An unreleased next-year projection stays missing in the published input fields; each model uses its training mean for this missing input. No realised GDP or inflation values enter the experiment.

## Monthly index protocols

- **Fixed origin:** Stop at September 2024. Recursively forecast the October 2024–March 2025 medians and freeze the available macro releases at that origin. Earlier forecasts replace missing future medians, including across the January year boundary. This is the direct comparison with the report's original index.
- **Monthly update:** Retain the fitted model coefficients, refresh the previous months' observed medians and available official releases before each forecast. The comparable linear12 and last-month controls also update. This assumes the preceding rent median is observable by the next month and represents a monthly forecasting service.
- **Ridge (`macro`):** Use all 12 preceding calendar-year medians and the current year's available monthly medians in calendar order, relative to a trailing three-month anchor. Pad unreached current-year slots with zero log difference. Use month sine/cosine and the two annual forecast midpoints. Fit log rent-relative-to-anchor with a Ridge penalty selected on expanding six-month windows wholly before October 2024. `no_macro` uses the same settings without GDP/CPI. Only 33 monthly training targets exist because the first year supplies lag history.
- **Anchored drift (`anchored_macro`):** An exploratory follow-up after observing the first Ridge holdout result. Age each earlier median forward by the target month and an assumed annual drift of headline inflation plus half real GDP growth. Average all eligible months with exponentially declining weights; choose the half-life among 1.5, 3 and 6 months using only pre-holdout origins. The drift is a modelling assumption rather than a measured relationship between GDP, CPI and rent. `anchored_history` sets the drift to zero with the same weights. Holdout improvement for this follow-up needs confirmation on another period.
- **Controls:** `linear12` fits a line to the most recent 12 medians; `last` predicts the last available median.

The seven baselines and six boosting configurations retain the same G0+G1 setup and original observed fit-period index. The rent-ratio models are therefore identical across future index choices. Saved per-row rent ratios allow all methods/protocols to share the same fit. Summary-only archived RMSE cannot be rescaled month by month. The script reconstructs missing index predictions once and records the locally reproduced linear12 control beside the archived score. For example, tuned XGBoost reproduced at 446.03 versus the archived 445.57. Use 446.03 for the paired index comparison. Monthly median errors give each month equal weight; individual rent errors weight each validation rental equally.

`macro_index/fixed_origin_2025_03_future_index.csv` contains monthly forecasts from April 2025 through July 2026 made as of March 2025. Unlabelled future months are not scored. March 2025 itself already has observed training rents. `settings.json` and the two training CV tables document hyperparameters and development windows.

## Findings for the report

- Location reduced raw-target forest RMSE from 468.77 to 458.70. Contemporaneous market data substantially helped raw linear models, but those features depend on the approval month's economic releases.
- Under the original retrospective low-rent-for-area rule, tuned LightGBM improved from 445.78 to 445.69. With strictly fit-only thresholds it worsened to 446.24, while the forest improved from 444.26 to 443.92. The direction of small effects depends on the model and flag protocol.
- The 16 history features reduced Ridge RMSE from 467.89 to 448.97 and forest RMSE from 444.26 to 440.16. The single tree deteriorated from 473.30 to 491.95. This paired experiment isolates adding lag inputs for each of these estimators. The archived five-member blend scored 430.97.
- Averaged over the five origins at the longer horizon, the index reduced group-median RMSE from 998.7 to 700.6 and raised forest RMSE from 650.2 to 668.2. The last origin has four fewer scored months.
- For the six fixed-origin monthly medians, the macro Ridge scored 120.53 SGD RMSE. The exploratory anchored model improved from 52.28 without macro inputs to 45.53 with them. The original 12-month line still had the lowest RMSE of the six methods (43.10). The monthly-update comparison likewise favored the line (46.08 versus 47.50). These six months and 33 training targets give limited evidence for a production change.
