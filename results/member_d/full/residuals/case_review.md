# Residual review: executed Member D holdout

**Verified result, 2 November 2026.** Tuned LightGBM, G0+G1 (39 inputs), index target, seed 5228; fit January 2021–September 2024 and validate October 2024–March 2025. Reference configuration: `20261025-210800_lightgbm_tuned_G0+G1_time_index`. These cases describe the local refit, whose aggregate metrics reproduce the reference; the original reference predictions were unavailable for row-level comparison. Sources are [top 20](top20_errors.csv), [underpredictions](largest_underpredictions.csv), [segments](segment_metrics.csv), [matched comparison](boosting_vs_ridge_segments.csv), and [run manifest](../manifest.json).

## Findings suitable for the report

The model has RMSE SGD **445.78**, MAE **323.91**, MAPE **12.25%** and bias **−24.51** across 17,037 rows. Residual means prediction minus truth. Although 60.20% of observations are underpredicted, positive errors account for 59.91% of total squared error: the overprediction tail is less frequent but more severe. The 20 largest absolute errors are all overpredictions and together contribute 3.91% of squared error; both signs are therefore reviewed separately below.

The clearest concentration is in retrospectively defined low true rents. The 585 observations below SGD 2,000 represent **3.43% of rows but 25.99% of squared error**, with RMSE SGD 1,226.50 and bias +1,112.23. Within them, the 41 rents below SGD 1,000 contribute 4.59% of total squared error. This diagnoses a low-target tail; neither a price threshold nor a large residual confirms a recording error, room rental or housing programme. True-rent bands cannot be used at inference time.

Large flats have somewhat higher per-row error but do not carry most total error. The 5-room and executive categories comprise 27.80% of rows and 31.32% of squared error; 3-room and 4-room flats contribute **66.12%**. Executive RMSE is SGD 524.32 on 851 observations. Jurong West is the largest town contributor (9.57%, n=1,174), followed by Bukit Merah (7.47%, n=972). The town labelled `central` contributes only 1.81% (n=266). Defining proximity explicitly as less than 5 km from Raffles Place gives 2,066 rows (12.13%) and 15.75% of squared error, with RMSE SGD 507.97. These results do not support a claim that central locations account for most error.

Monthly RMSE is 431.49, 458.65, 440.85, 460.25, 443.81 and 439.05 SGD at horizons 1–6. There is no monotonic deterioration across this window. Bias remains slightly negative in every month (−33.28 to −14.30 SGD). Longer-horizon extrapolation remains a limitation rather than the demonstrated dominant failure here. Predicted-rent decile means remain close to the calibration diagonal (bin biases −39.13 to −9.49 SGD), but averaging hides substantial errors at both true-rent tails.

With identical fit rows, input columns and index target handling, Ridge has RMSE SGD 467.13 and bias −57.54; boosting lowers RMSE by SGD 21.34 and has lower absolute error on 9,723/17,037 rows (57.07%). Boosting improves RMSE in every reported MRT-distance bin: 456.12 versus 481.43 SGD below 0.5 km (n=7,686), and 438.99 versus 456.91 SGD at 0.5–1 km (n=7,649). Town/type gains vary: Bukit Merah 3-room improves from 544.87 to 479.36 SGD (n=414), while Jurong West executive worsens from 580.39 to 600.65 SGD (n=99). The varying gains support a conditional model comparison; they do not by themselves prove which learned interaction or nonlinear distance response caused it. Consult the PDPs alongside these results.

## Review of the 20 largest absolute errors

All 20 source rents are between SGD 600 and 2,000, nine are below SGD 1,000, and the available area ranges from 65 to 146 m². There are eight 5-room, seven 4-room, three 3-room and two executive observations across 13 towns. Ten occur in November 2024. None has the available invalid-lease flag, nonpositive/missing area, or missing town/type flag. **No source-record error is confirmed.** One case belongs to a small observed town/type holdout segment; the others remain unexplained. A small holdout segment is a coverage flag, not a cause.

| ROW_ID | Town / flat type | True SGD | Prediction SGD | Residual SGD | Evidence / classification |
| --- | --- | ---: | ---: | ---: | --- |
| 147330 | Bukit Merah / 5-room | 650 | 3921.34 | +3271.34 | Low observed rent; unexplained. |
| 137245 | Bukit Timah / executive | 1400 | 4577.74 | +3177.74 | Town/type holdout n=3; fit n=32. Rare observed segment; cause unexplained. |
| 137603 | Ang Mo Kio / 5-room | 750 | 3896.88 | +3146.88 | Low observed rent; unexplained. |
| 138124 | Toa Payoh / 4-room | 800 | 3738.70 | +2938.70 | Low observed rent; unexplained. |
| 137920 | Kallang/Whampoa / 3-room | 800 | 3526.06 | +2726.06 | Same recorded profile also has rent 3550; origin of spread unverified. |
| 137697 | Tampines / 4-room | 700 | 3277.40 | +2577.40 | Low observed rent; unexplained. |
| 137580 | Bukit Merah / 5-room | 1500 | 4077.11 | +2577.11 | Unexplained. |
| 135829 | Yishun / 4-room | 600 | 3167.99 | +2567.99 | Low observed rent; unexplained. |
| 142449 | Jurong West / 5-room | 1200 | 3681.84 | +2481.84 | Unexplained. |
| 144886 | Jurong West / 5-room | 600 | 3079.28 | +2479.28 | Low observed rent; unexplained. |
| 149220 | Bishan / 4-room | 1200 | 3615.24 | +2415.24 | Unexplained. |
| 136288 | Jurong West / 5-room | 700 | 3101.74 | +2401.74 | Low observed rent; unexplained. |
| 137509 | Bedok / executive | 1500 | 3889.31 | +2389.31 | Town/type holdout n=38; unexplained. |
| 143686 | Jurong West / 4-room | 1000 | 3377.40 | +2377.40 | Unexplained; not in the strictly below-1000 band. |
| 133225 | Central / 3-room | 1008 | 3363.69 | +2355.69 | Unexplained. |
| 142555 | Bukit Merah / 3-room | 700 | 3008.64 | +2308.64 | Same recorded profile also has rent 2880; origin of spread unverified. |
| 135867 | Queenstown / 4-room | 2000 | 4239.80 | +2239.80 | Unexplained; not in the strictly below-2000 band. |
| 144281 | Jurong West / 5-room | 1350 | 3567.29 | +2217.29 | Unexplained. |
| 142088 | Clementi / 5-room | 2000 | 4215.92 | +2215.92 | Unexplained. |
| 133155 | Jurong West / 4-room | 1000 | 3212.19 | +2212.19 | Unexplained. |

Two paired examples directly illustrate recorded-profile dispersion. ROW_IDs **136125/137920** describe Kallang/Whampoa block 8 Boon Keng Road, November 2024, 3-room, 68 m², DBSS, lease year 2011, at rents **3550/800 SGD**. ROW_IDs **141383/142555** describe Bukit Merah block 130 Kim Tian Road, January 2025, 3-room, 65 m², improved, lease year 1976, at **2880/700 SGD**. All four values occur in the raw training file. Different units within a block, unrecorded tenancy details and possible misrecording remain alternative hypotheses; these records alone do not determine the explanation.

## Review of underpredictions

The ten largest underpredictions contribute 0.95% of total squared error. Their true rents range from SGD 4,400 to 6,000. All have valid available lease flags and positive recorded area. No data error is established.

| ROW_ID | Town / flat type | True SGD | Prediction SGD | Residual SGD | Evidence / classification |
| --- | --- | ---: | ---: | ---: | --- |
| 139592 | Kallang/Whampoa / executive | 6000 | 3895.92 | −2104.08 | Town/type holdout n=9, fit n=47; adjoined flat. Rare observed segment; cause unexplained. |
| 136856 | Kallang/Whampoa / 4-room | 5600 | 3578.89 | −2021.11 | Adjoined flat, 125 m², age 55; unexplained. |
| 139895 | Kallang/Whampoa / 3-room | 5000 | 3120.31 | −1879.69 | Terrace flat model has only 14 fit rows across all towns; rare observed model category. |
| 144939 | Bukit Merah / 3-room | 6000 | 4206.98 | −1793.02 | Unexplained. |
| 146848 | Bukit Merah / 4-room | 5000 | 3222.46 | −1777.54 | Unexplained. |
| 145090 | Bukit Merah / 5-room | 5800 | 4028.69 | −1771.31 | Unexplained. |
| 148111 | Bukit Merah / 3-room | 5900 | 4185.39 | −1714.61 | Unexplained. |
| 137892 | Yishun / executive | 5300 | 3648.70 | −1651.30 | Town/type holdout n=31; unexplained. |
| 142396 | Bukit Merah / 3-room | 5700 | 4105.57 | −1594.43 | Unexplained. |
| 138300 | Central / 3-room | 4400 | 2838.57 | −1561.43 | Same block/type/area/month/model/lease profile also has rent 2500 (ROW_ID 139068); spread unexplained. |

Unrecorded floor/storey, condition, renovation, facing, furnishing detail and tenancy context are plausible omitted information for the dataset. Their contribution is a hypothesis, not an established explanation assigned to any row above. Age, size or centrality alone does not explain these cases.

## Independent verification and reproduction

All 56 reported segment groups were independently checked against the saved row predictions and processed metadata: counts, RMSE, MAE, MAPE, bias, bin edges, squared-error sums/shares and aggregation back to overall RMSE passed. The four date boundaries were checked in synthetic tests; real holdout IDs exactly match the 17,037 metadata IDs. Every joined target agrees with the original raw row at ROW_ID. The actual top-20 absolute-error order and top-10 underprediction order were independently reconstructed. All six segmentations account for every row; the real holdout has no missing age values. All three residual PNGs were inspected for labels, units and clipping, including the distinct RMSE and squared-error-share axes in the town figure.

Generation command: `& .venv/Scripts/python.exe -m scripts.run_member_d --member D --output results/member_d/full` (use a fresh output directory when rerunning). Residual and outlier calculations are implemented in `src/member_d_residuals.py`; checks run with `& .venv/Scripts/python.exe -m unittest discover -s checks -v`. The raw/processed data, row predictions and models remain local ignored inputs. Shared-run provenance and hashes are retained in the manifest. Full addresses, months, area, model, age, MRT distance and every unrounded error are retained in the case CSVs.
