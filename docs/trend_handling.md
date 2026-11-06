# Trend handling and split comparison (Member A)

Plan decision 5 and the random CV check from decision 1. All runs use G0+G1 features, seed 5228 and the shared `run_experiment`. Rebuild every table and figure below with `python -m scripts.trend_analysis`. The CSV files are in [results/member_a/trend/](../results/member_a/trend/).

## What was run

* Index target on the time holdout for group median, linear regression, Ridge, KNN, decision tree, random forest, default CatBoost and default XGBoost. Member C already had the tuned boosting runs, including tuned XGBoost, and the global median and default LightGBM were run in Phase 0.
* Raw target on the time holdout for the six Member B baselines again, in the shared Python 3.10 setup, so each raw and index pair comes from one environment. The scores match Member B's Python 3.14 run to within 0.001.
* Random 5 fold CV for group median, linear regression, Ridge, KNN, decision tree, random forest with index target and tuned LightGBM with index target.

## Verdicts

1. **The index target lowers the RMSE of 11 of the 13 models.** For KNN and default CatBoost the gain is under 1 SGD, which is too small to matter. Tuned CatBoost gets 0.2 SGD worse. Default XGBoost gets 2.4 SGD worse, although its bias falls from −36 to −22 SGD. Tuned XGBoost gains 4.3 SGD, so the loss does not carry over to the tuned model.

   | Model | Raw | Index | Change |
   |---|---:|---:|---:|
   | Global median | 837.1 | 641.5 | −195.6 |
   | Group median | 754.9 | 483.1 | −271.8 |
   | Linear regression | 575.4 | 467.6 | −107.8 |
   | Ridge | 576.1 | 467.9 | −108.2 |
   | KNN | 482.3 | 481.9 | −0.4 |
   | Decision tree | 482.9 | 473.3 | −9.7 |
   | Random forest | 458.7 | **444.3** | −14.4 |
   | LightGBM (default) | 457.3 | 454.3 | −3.0 |
   | CatBoost (default) | 452.7 | 452.2 | −0.5 |
   | XGBoost (default) | 457.2 | 459.6 | +2.4 |
   | LightGBM (tuned) | 449.9 | 445.8 | −4.1 |
   | CatBoost (tuned) | 448.8 | 449.0 | +0.2 |
   | XGBoost (tuned) | 449.9 | 445.6 | −4.3 |

2. **The untuned random forest with the index target (444.3) is the best single result.** It beats tuned XGBoost (445.6) and the final single model, tuned LightGBM (445.8), both with the index target. The three are within 1.5 SGD, and the difference comes from one six month holdout, so it does not by itself justify a change of the final single model. Member C already treats XGBoost and LightGBM as a tie (public leaderboard 434.5 and 434.4) and keeps LightGBM. The same reasoning covers the forest, and the final model is now the blend, which uses all three. LightGBM fits in about 18 seconds and the forest in about 97 seconds on Juntong's PC.

3. **Most of the gap between simple and complex models is the trend, not the model.** On raw rent Ridge is 126 SGD behind tuned LightGBM. With the index target it is 22 SGD behind. Ridge on raw rent over-predicts by 330 SGD on average, because its month coefficient extends the average rise from 2021 to 2024 into a flat period. The medians on raw rent under-predict by more than 500 SGD.

4. **The holdout cannot show trend drift for tree models.** Rents rose quickly through 2022 and were nearly flat from mid 2023 (median about SGD 3,100), so the extrapolated index rises only about 2.4 SGD per month and stays within 73 SGD of the observed median in the holdout months. Tree model errors do not grow from October 2024 to March 2025. Only the linear models on raw rent show strongly growing bias (Ridge 239 to 381 SGD). On raw rent the tree models under-predict by 28 to 50 SGD, probably because leaves mix old and recent rows. The index target shrinks this bias for every tree model except the two CatBoost models, where it grows (default −28 to −36, tuned −30 to −50).

5. **Random CV flatters the lookup and linear models, not the trees.** KNN goes the other way: random CV makes it look 26 SGD worse. Gap is time holdout RMSE minus random CV RMSE on the same October 2024 to March 2025 rows.

   | Model | Time | Random CV | Gap |
   |---|---:|---:|---:|
   | Global median | 837.1 | 794.6 | +42.5 |
   | Group median | 754.9 | 490.5 | +264.4 |
   | Linear regression | 575.4 | 502.2 | +73.2 |
   | Ridge | 576.1 | 502.7 | +73.3 |
   | KNN | 482.3 | 508.6 | −26.3 |
   | Decision tree | 482.9 | 480.6 | +2.3 |
   | Random forest | 458.7 | 461.1 | −2.4 |
   | Random forest + index | 444.3 | 451.9 | −7.6 |
   | LightGBM (default) | 457.3 | 455.3 | +2.0 |
   | CatBoost (default) | 452.7 | 451.6 | +1.2 |
   | XGBoost (default) | 457.2 | 456.5 | +0.7 |
   | LightGBM (tuned) | 449.9 | 451.9 | −2.0 |
   | LightGBM (tuned) + index | 445.8 | 450.7 | −4.9 |
   | CatBoost (tuned) | 448.8 | 451.0 | −2.1 |
   | XGBoost (tuned) | 449.9 | 450.8 | −0.9 |

   A model choice made with random CV would have ranked the group median (490.5) above KNN (508.6), although on the time holdout it is 270 SGD worse. The time holdout is the right split for this task.

6. **The index target is a bet that the recent slope continues.** The holdout has flat rents, so we moved the end of the fit window back and forecast up to 16 months ahead from five earlier dates. Only 12 months of data follow the last date. `python -m scripts.horizon_backtest` takes about 15 minutes and writes [horizon_backtest.csv](../results/member_a/trend/horizon_backtest.csv). RMSE for tuned LightGBM:

   | Fit to | Slope (SGD per month) | Months 1 to 6, raw | Months 1 to 6, index | Months 7 to 16, raw | Months 7 to 16, index |
   |---|---:|---:|---:|---:|---:|
   | 2022-03 | 21.5 | 535 | 517 | 909 | 762 |
   | 2022-09 | 42.7 | 698 | 652 | 727 | 622 |
   | 2023-03 | 69.8 | 640 | 668 | 569 | 955 |
   | 2023-09 | 25.7 | 557 | 577 | 491 | 598 |
   | 2024-03 | 9.8 | 496 | 492 | 455 | 451 |

   While rents kept rising, the index target won by up to 147 SGD. Once rents flattened, it lost by up to 386 SGD, because the line through the last 12 medians was still steep. Fitted to March 2023 it reaches SGD 4,171 after 16 months, when the observed median stayed near SGD 3,100, so bias is +743 SGD on months 7 to 16. XGBoost and the forest show the same pattern.

   The March 2024 fit is the closest case to the final single model. Its slope is 9.8 SGD per month, and the index target is slightly better there (451 vs 455 on months 7 to 12, XGBoost 450 vs 452, forest 449 vs 462). At the end of the training data the slope is 5.3 SGD per month, so the index reaches SGD 3,246 at the end of the test period against a last median of SGD 3,200. This points to the low risk case, although five dates are too few to prove it. The public leaderboard agrees for now: tuned LightGBM scores 434.43 with the index target and 437.15 raw. That score covers only the public part of the test set.

   With the index target, the forest beats LightGBM in 9 of 10 rows (by up to 25 SGD), and XGBoost beats LightGBM in all 10 (by up to 10 SGD), although the two low slope rows are within about 2 SGD. The rows overlap in time, so this is weak evidence, but it does not support preferring LightGBM over the other two. The final model is now the blend, which uses all three.

## Caveats

* The final single model was tuned and selected on the same holdout, so its holdout RMSE is slightly optimistic.
* The test period runs up to 16 months past the training data, against 6 months for the holdout. Verdict 6 tests this directly.
* The backtest in verdict 6 uses hyperparameters tuned on the later holdout, so it borrows some information from later data.

## Report

The Methods section in [report/sections/04_methods.tex](../report/sections/04_methods.tex) uses these results. See [report/README.md](../report/README.md) to build the PDF.
