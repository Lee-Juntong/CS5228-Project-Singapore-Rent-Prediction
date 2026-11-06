# Member B experiment results

RMSE is in SGD per rental unless the table explicitly scores monthly medians. Lower is better. Each experiment CSV records the source of reused rows. Figures are rounded here only.

## Feature groups, raw target

| model | G0 | G0+G1 | G0+G1+G2 | ALL |
| --- | --- | --- | --- | --- |
| global_median | 837.11 | 837.11 | 837.11 | 837.11 |
| group_median | 754.95 | 754.95 | 754.95 | 754.95 |
| linear_regression | 582.83 | 575.42 | 575.13 | 479.58 |
| ridge | 583.61 | 576.07 | 575.77 | 479.51 |
| knn | 545.55 | 482.34 | 482.34 | 482.34 |
| decision_tree | 480.60 | 482.95 | 483.91 | 484.87 |
| random_forest | 468.77 | 458.70 | 458.62 | 458.86 |
| lightgbm_tuned (archived) | 460.24 | 449.90 | 449.77 | 452.37 |
| catboost_tuned (archived) | 462.74 | 448.83 | 448.54 | 450.11 |

## Dropping flagged fitting rows, index target

| model | all | m1 | m2 | both |
| --- | --- | --- | --- | --- |
| global_median | 641.54 | 641.50 | 641.54 | 641.54 |
| group_median | 483.10 | 483.29 | 483.10 | 483.11 |
| linear_regression | 467.59 | 468.89 | 467.14 | 467.51 |
| ridge | 467.89 | 469.10 | 467.48 | 467.81 |
| knn | 481.94 | 482.37 | 482.12 | 481.95 |
| decision_tree | 473.30 | 472.92 | 472.57 | 472.57 |
| random_forest | 444.26 | 446.92 | 444.09 | 443.96 |
| lightgbm_tuned (archived) | 445.78 | 447.94 | 445.69 | 446.14 |

## Repeat-rent features, index target

| model | base | plus16 |
| --- | --- | --- |
| global_median | 641.54 | 641.54 |
| group_median | 483.10 | 483.10 |
| linear_regression | 467.59 | 448.91 |
| ridge | 467.89 | 448.97 |
| knn | 481.94 | 451.22 |
| decision_tree | 473.30 | 491.95 |
| random_forest | 444.26 | 440.16 |

## Outlier sensitivity with strictly training-only thresholds

All rules are estimated again on fitting rows only. The tuned LightGBM control uses the same flags.

| model | all | m1 | m2 | both |
| --- | --- | --- | --- | --- |
| decision_tree | 473.30 | 472.58 | 471.90 | 471.41 |
| global_median | 641.54 | 641.52 | 641.54 | 641.54 |
| group_median | 483.10 | 484.04 | 483.11 | 483.10 |
| knn | 481.94 | 482.99 | 482.09 | 481.94 |
| lightgbm_tuned | 445.78 | 449.13 | 446.24 | 446.18 |
| linear_regression | 467.59 | 467.79 | 467.19 | 467.53 |
| random_forest | 444.26 | 447.19 | 443.92 | 444.23 |
| ridge | 467.89 | 468.01 | 467.53 | 467.83 |

The fixed median definitions ignore added columns. Separate lookup variants give the L24 mean a weight of n/(n+5), with n earlier matching transactions and the ordinary median model providing the remainder.

| model | rmse | with_history |
| --- | --- | --- |
| global_median | 572.18 | 12314 |
| group_median | 454.87 | 12314 |

Previously scored holdout comparisons. The lagged LightGBM also changes regularisation, and the five-model blend changes the estimator combination.

| model | rmse |
| --- | --- |
| lightgbm_tuned_index | 445.78 |
| lag_lightgbm_index | 435.86 |
| lag_blend2 | 432.60 |
| lag_blend5 | 430.97 |
| lag_blend_index_raw | 431.60 |

## Earlier origins

Equal-weight mean of the five window RMSEs, so overlapping windows are not independent observations. Near covers months 1–6. Far covers months 7–16 except the last origin, where labels end at month 12. Per-origin counts, bias and scores: horizon.csv.

| model | near_index | near_raw | far_index | far_raw |
| --- | --- | --- | --- | --- |
| decision_tree | 602.18 | 640.39 | 695.63 | 683.27 |
| global_median | 716.93 | 942.83 | 810.04 | 1045.91 |
| group_median | 606.06 | 893.02 | 700.57 | 998.67 |
| knn | 603.51 | 603.24 | 697.28 | 655.70 |
| lightgbm_tuned | 581.07 | 585.28 | 677.52 | 630.36 |
| linear_regression | 583.02 | 610.45 | 670.27 | 677.36 |
| random_forest | 576.71 | 602.21 | 668.19 | 650.20 |
| ridge | 583.22 | 610.89 | 670.60 | 678.08 |
| xgboost_tuned | 577.27 | 584.20 | 671.96 | 630.55 |

## Macro index: monthly median RMSE

| method | fixed | monthly_update |
| --- | --- | --- |
| anchored_history | 52.28 | 50.13 |
| anchored_macro | 45.53 | 47.50 |
| last | 62.08 | 62.92 |
| linear12 | 43.10 | 46.08 |
| macro | 120.53 | 68.90 |
| no_macro | 120.58 | 68.92 |

## Macro index: per-rental RMSE on the same 17,037 holdout rows

| model | archived_raw | archived_index | anchored_history | anchored_macro | last | linear12 | macro | no_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_default | 452.73 | 452.20 | 453.76 | 451.26 | 455.63 | 452.20 | 461.88 | 461.88 |
| catboost_tuned | 448.83 | 449.04 | 451.11 | 446.68 | 453.42 | 449.04 | 454.57 | 454.58 |
| decision_tree | 482.95 | 473.30 | 474.12 | 474.14 | 475.30 | 473.30 | 487.86 | 487.87 |
| global_median | 837.11 | 641.54 | 641.96 | 642.59 | 642.65 | 641.54 | 653.68 | 653.69 |
| group_median | 754.95 | 483.10 | 482.46 | 487.85 | 482.35 | 483.10 | 509.09 | 509.11 |
| knn | 482.34 | 481.94 | 482.13 | 484.52 | 482.81 | 481.94 | 501.32 | 501.33 |
| lightgbm_default | 457.31 | 454.27 | 455.47 | 454.26 | 457.00 | 454.27 | 466.77 | 466.78 |
| lightgbm_tuned | 449.90 | 445.78 | 446.94 | 445.93 | 448.46 | 445.78 | 458.99 | 459.00 |
| linear_regression | 575.42 | 467.59 | 469.87 | 464.53 | 472.31 | 467.60 | 470.54 | 470.55 |
| random_forest | 458.70 | 444.26 | 445.63 | 443.79 | 447.32 | 444.26 | 455.72 | 455.73 |
| ridge | 576.07 | 467.89 | 470.16 | 464.82 | 472.60 | 467.89 | 470.81 | 470.82 |
| xgboost_default | 457.18 | 459.57 | 460.48 | 460.31 | 461.77 | 459.57 | 474.13 | 474.14 |
| xgboost_tuned | 449.87 | 445.57 | 447.17 | 446.22 | 448.67 | 446.03 | 459.37 | 459.38 |

The paired linear12 column scores the same local model and predictions as the alternative indices. Compare each alternative with that column to isolate the index change. Monthly-update scores by rental and all metrics are in macro_index/rent_metrics.csv.

Environment: Python 3.10.22, numpy 1.26.4, pandas 2.2.1, scikit-learn 1.5.1, lightgbm 4.6.0, xgboost 3.2.0, catboost 1.2.8, pyarrow 19.0.1.
