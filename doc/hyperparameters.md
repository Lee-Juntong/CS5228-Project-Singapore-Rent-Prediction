# Baseline hyperparameters

All models use raw monthly rent and the shared chronological holdout. Parameters were fixed before evaluating the reported models; no hyperparameter search or early stopping was used. The random seed is `5228`.

| Model | Main settings |
| --- | --- |
| Global median | Median of all training rents |
| Group median | Group medians; fallback: town + flat type + calendar quarter, then town + flat type, flat type, town, global |
| Linear regression | `fit_intercept=True`, `positive=False`; remaining estimator defaults |
| Ridge | `alpha=10`, `solver="lsqr"`, `tol=1e-6`, `max_iter=10000`, `fit_intercept=True` |
| KNN | `n_neighbors=50`, `weights="distance"`, `algorithm="kd_tree"`, `leaf_size=30`, Euclidean distance (`p=2`), `n_jobs=8` |
| Decision tree | `criterion="squared_error"`, `max_depth=16`, `min_samples_leaf=20`, `min_samples_split=2`, `max_features=None`, `random_state=5228` |
| Random forest | `n_estimators=300`, `criterion="squared_error"`, `max_depth=20`, `min_samples_leaf=5`, `min_samples_split=2`, `max_features=0.7`, `bootstrap=True`, `max_samples=None`, `oob_score=False`, `random_state=5228`, `n_jobs=8` |

Ridge provides a regularized linear reference. KNN averages a moderate number of nearby transactions. Tree depth and leaf-size limits reduce overfitting, while the forest averages 300 trees. `max_features=0.7` applies to encoded columns. These are fixed baseline settings, not claimed optima.

## Features and preprocessing

The shared candidate set `G0+G1` has 39 columns. Model-specific inputs are:

- **Global median:** no predictors.
- **Group median:** `TOWN`, `FLAT_TYPE`, `MONTH_INDEX`. Quarter is `floor(MONTH_INDEX / 3)`, where January 2021 is month zero; years are distinguished. No minimum group size is imposed.
- **Linear regression and Ridge:** retain categorical flat type and drop `FLOOR_AREA_SQM`, `LOG_FLOOR_AREA_SQM`, `FLOOR_AREA_PER_ROOM_PROXY`, `FLAT_TYPE_ORDINAL` and `LEASE_COMMENCE_DATE`, leaving 34 inputs. One-hot encode town, flat type, flat model and subzone with `drop="first"`; median-impute and standardize numeric columns.
- **KNN:** only latitude, longitude, floor area and month index, with median imputation and standard scaling.
- **Decision tree and random forest:** all 39 inputs; median-impute numeric columns and one-hot encode categories without dropping a level. No scaling. Random forest receives a dense encoded matrix.

Preprocessing is fitted within each training partition. Encoders use `handle_unknown="ignore"`; imputers use `keep_empty_features=True`.

## Learning curves

Training fractions are 10%, 25%, 50%, 75% and 100%, using a deterministic nested sample within each training month. Every point uses the same model parameters and the same future holdout. The forest also evaluates prefixes of 10, 25, 50, 100, 200 and 300 trees from the saved ensemble. The final tree count remains fixed at 300.

Complete estimator and preprocessing parameters are in [hyperparameters.json](../results/member_b/baseline_ladder/hyperparameters.json). Package pins are in [requirements-baselines.txt](requirements-baselines.txt).
