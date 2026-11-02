# Trend handling and split comparison (Member A)

Plan decision 5 and the random CV check from decision 1. All runs use G0+G1 features, seed 5228 and the shared `run_experiment`. Rebuild every table and figure below with `python -m scripts.trend_analysis`. The CSV files are in [results/member_a/trend/](../results/member_a/trend/).

## What was run

* Index target on the time holdout for group median, linear regression, Ridge, KNN, decision tree, random forest and default CatBoost. Member C already had the boosting runs, and the global median and default LightGBM were run in Phase 0.
* Raw target on the time holdout for the six Member B baselines again, in the shared Python 3.10 setup, so each raw and index pair comes from one environment. The scores match Member B's Python 3.14 run exactly.
* Random 5 fold CV for group median, linear regression, Ridge, KNN, decision tree, random forest with index target and tuned LightGBM with index target.

## Verdicts

1. **The index target helps every model, or makes no difference.** It never hurts by more than 0.2 SGD.

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
   | LightGBM (tuned) | 449.9 | 445.8 | −4.1 |
   | CatBoost (tuned) | 448.8 | 449.0 | +0.2 |

2. **The untuned random forest with the index target (444.3) is now the best single result.** It beats the chosen final model, tuned LightGBM with the index target (445.8), by 1.5 SGD. This difference is small and comes from one six month holdout, so it does not by itself justify a change of the final model. Member C should decide whether to keep LightGBM. LightGBM fits in about 18 seconds and the forest in about 97 seconds on Juntong's PC.

3. **Most of the gap between simple and complex models is the trend, not the model.** On raw rent Ridge is 126 SGD behind tuned LightGBM. With the index target it is 22 SGD behind. Ridge on raw rent over-predicts by 330 SGD on average, because its month coefficient extends the steep rise of 2022 and 2023 into a flat period. The medians on raw rent under-predict by more than 500 SGD.

4. **The holdout cannot show trend drift for tree models.** Rents were nearly flat from February 2024 (median about SGD 3,100), so the extrapolated index rises only about 2.4 SGD per month and stays within 73 SGD of the observed median in the holdout months. Tree model errors do not grow from October 2024 to March 2025. Only Ridge on raw rent shows growing bias (239 to 381 SGD). The index still helps trees, because it removes a constant under-prediction of 28 to 50 SGD that comes from leaves mixing old and recent rows.

5. **Random CV flatters the simple models, not the trees.** Gap is time holdout RMSE minus random CV RMSE on the same October 2024 to March 2025 rows.

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
   | LightGBM (tuned) | 449.9 | 451.9 | −2.0 |
   | LightGBM (tuned) + index | 445.8 | 450.7 | −4.9 |
   | CatBoost (tuned) | 448.8 | 451.0 | −2.1 |

   A model choice made with random CV would have ranked the group median (490.5) above KNN (508.6), although on the time holdout it is 270 SGD worse. The time holdout is the right split for this task.

## Caveats

* The final model was tuned and selected on the same holdout, so its holdout RMSE is slightly optimistic.
* The test period runs up to 16 months past the training data, against 6 months for the holdout. The index target is still the only option that lets a tree model predict a rent level above the training range.

## Report

The Methods section in [report/sections/04_methods.tex](../report/sections/04_methods.tex) uses these results. See [report/README.md](../report/README.md) to build the PDF.
