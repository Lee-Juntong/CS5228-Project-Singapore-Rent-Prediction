# Pipeline guide: reusable functions

This page lists every function you can reuse for your models. All of them live in `src/`. Run code from the project root so `from src...` imports work.

The short version for a new model:

```python
from sklearn.linear_model import Ridge
from src.evaluation import run_experiment, compare_splits
from src.features import load_modeling_data
from src.models import sklearn_tabular

data = load_modeling_data()                      # load once, reuse for every run
make_model = lambda: sklearn_tabular(Ridge(alpha=10))
run_experiment("ridge_a10", make_model, feature_set="G0+G1", member="B", data=data)
```

That one call fits on Jan 2021 to Sep 2024, scores Oct 2024 to Mar 2025 and logs a row to `results/results.csv`. Commit the new file in `results/runs/` with your code.

## The model contract

A model is any function with no arguments that returns an object with `fit(X, y)` and `predict(X)`.

* `X` is a pandas DataFrame holding only the feature columns of the chosen set.
* `TOWN`, `FLAT_TYPE`, `FLAT_MODEL` and `HDB_SUBZONE` arrive as pandas `category`. Category codes are the same in train and test.
* Numeric columns can hold NaN. Your model handles its own encoding and imputation, or you wrap it in `sklearn_tabular`.
* `y` is a NumPy array of monthly rent. In `target_mode="index"` it is rent divided by the monthly rent index, and the pipeline multiplies your predictions back. You do not need to change anything in your model for this.

Pass the function itself, not a fitted model. The pipeline calls it once per fold so every fold starts fresh.

## `src/features.py`

### `load_modeling_data() -> (train, test)`

Reads the processed parquet files, adds the external features and sets the categorical dtypes. Returns two DataFrames. `train` has `MONTHLY_RENT`, `test` does not. Both have `ROW_ID`, `RENT_APPROVAL_DATE` and `MONTH_INDEX`. It takes a few seconds, so call it once and pass the result as `data=` to the functions below.

```python
from src.features import load_modeling_data
train, test = load_modeling_data()
```

### `feature_columns(feature_set) -> list[str]`

Returns the column names for a feature set in a fixed order. Named sets are `"G0"`, `"G0+G1"`, `"G0+G1+G2"` and `"ALL"`. Any `+` join of groups works too, for example `"G0+G2"`.

```python
from src.features import feature_columns
cols = feature_columns("G0+G1")      # 39 columns
X = train[cols]
```

### `FEATURE_GROUPS`, `FEATURE_SETS`, `CATEGORICAL_COLUMNS`

The dicts behind `feature_columns`. Read them to see what each group holds. Do not edit them for your own experiments, since every member must compare on the same inputs. If you need a narrower column list (for example a linear model that keeps only one of area and flat type), drop columns inside your model, as in the example under `sklearn_tabular` below.

## `src/evaluation.py`

### `run_experiment(model_name, make_model, feature_set="G0+G1", split="time", target_mode="raw", member="", notes="", data=None, save_predictions=True, log=True) -> dict`

The one function every model goes through.

* `split`: `"time"` (main) or `"random5"` (shuffled 5 fold CV).
* `target_mode`: `"raw"` trains on rent. `"index"` trains on rent divided by a monthly rent index.
* `log=False` prints the scores but writes nothing. Use it for quick trials.

Returns the logged row: `rmse`, `mae`, `mape`, `bias`, `rmse_std` (spread over folds), `rmse_window` (see `compare_splits`), timings and the git commit. When logging it writes:

* `results/runs/<run_id>.csv`, one row. Commit it.
* `results/results.csv`, rebuilt from all run files.
* `results/breakdowns/<run_id>.csv`, RMSE per flat type and per month (time split only).
* `results/predictions/<run_id>.parquet`, columns `ROW_ID`, `fold`, `y_true`, `y_pred`. Not in git. Use it for error analysis.

```python
from src.evaluation import run_experiment
from src.models import make_lightgbm
row = run_experiment("lightgbm_default", make_lightgbm, feature_set="G0+G1",
                     target_mode="index", member="C", data=data)
print(row["rmse"])
```

### `compare_splits(model_name, make_model, feature_set="G0+G1", target_mode="raw", member="", notes="", data=None, log=True) -> DataFrame`

Runs the time holdout and random 5 fold CV for one model, logs both runs and returns one row:

| column | meaning |
|---|---|
| `rmse_time` | RMSE on the time holdout (Oct 2024 to Mar 2025) |
| `rmse_random5` | mean RMSE over the 5 random folds, all months |
| `rmse_random5_std` | spread of that RMSE over folds |
| `rmse_random5_window` | random CV RMSE on the Oct 2024 to Mar 2025 rows only |
| `optimism` | `rmse_time` minus `rmse_random5_window` |

Compare `rmse_time` with `rmse_random5_window`, not with `rmse_random5`. Both score the same rows, so the gap is only the effect of the split. `rmse_random5` covers all of 2021 to 2025, and rents in 2023 vary far more than in the holdout, so its headline number is not comparable. A positive `optimism` means random CV makes the model look better than it would on future months.

Use it on your best model once you have one:

```python
from src.evaluation import compare_splits
compare_splits("catboost_tuned", make_catboost, member="C", data=data)
```

### `split_comparison(results=None) -> DataFrame`

Builds the same comparison from `results/results.csv` without refitting. It pairs the latest time and random5 runs for each model, feature set and target mode. Use it to make the report table.

```python
from src.evaluation import split_comparison
split_comparison().round(1)
```

### `fit_predict(make_model, fit_frame, predict_frame, columns, target_mode="raw") -> (prediction, fit_seconds, predict_seconds)`

One fit and one prediction with the shared seed and trend handling, without logging. Use it for custom loops such as tuning trials.

```python
from src.evaluation import fit_predict, time_holdout_masks, regression_metrics
fit, valid = time_holdout_masks(train)
pred, _, _ = fit_predict(make_model, train[fit], train[valid], feature_columns("G0+G1"))
regression_metrics(train.loc[valid, "MONTHLY_RENT"], pred)
```

### `time_holdout_masks(train) -> (fit_mask, valid_mask)`

Boolean arrays for the time holdout. Fit is Jan 2021 to Sep 2024, valid is Oct 2024 to Mar 2025. Tune on these masks so your tuning matches the reported split.

### `iter_folds(train, split)`

Yields `Fold(name, fit_index, valid_index)` with row positions for `"time"` (one fold) or `"random5"` (five folds, seed 5228).

### `in_time_valid_window(frame) -> mask`

True for rows dated Oct 2024 to Mar 2025.

### `regression_metrics(y_true, y_pred) -> dict`

Returns `rmse`, `mae`, `mape` (percent) and `bias` (mean of prediction minus truth, so positive means too high).

### `segment_metrics(frame, prediction) -> DataFrame`

RMSE, MAE and MAPE per flat type and per month. `frame` needs `FLAT_TYPE`, `RENT_APPROVAL_DATE` and `MONTHLY_RENT`.

### `MonthlyRentIndex(window=12)`

The rent index used by `target_mode="index"`. `fit(month_index, rent)` stores the median rent per month and a straight line through the last `window` medians. `transform(month_index)` returns the median for seen months and the line for later months.

### `collect_results() -> DataFrame`

Rebuilds `results/results.csv` from `results/runs/`. Run it after a merge conflict in `results.csv`.

### `make_submission(make_model, feature_set, target_mode="raw", path=None, data=None) -> DataFrame`

Fits on all training rows, predicts the test set and writes a Kaggle file with columns `Id` and `Predicted`.

```python
from src.evaluation import make_submission
make_submission(make_lightgbm, "G0+G1", path="submissions/lightgbm.csv", data=data)
```

## `src/models.py`

### `sklearn_tabular(estimator, scale=True) -> Pipeline`

Wraps any scikit-learn regressor so it accepts the shared X. Categorical columns are one hot encoded (unseen levels become all zeros). Numeric columns get median imputation and, if `scale` is true, standard scaling. Set `scale=False` for trees.

To use fewer columns than the feature set, put a column selector in front:

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

def make_linear_no_area():
    drop = FunctionTransformer(lambda X: X.drop(columns=["FLOOR_AREA_SQM", "LOG_FLOOR_AREA_SQM"]))
    return make_pipeline(drop, sklearn_tabular(LinearRegression()))
```

### Ready made models

* `make_global_median()`: predicts the training median. Sanity floor.
* `make_lightgbm()`: LightGBM with default settings and native categories.
* `make_ridge()`: Ridge (alpha 1) on all columns through `sklearn_tabular`. A simple linear reference.

### `MODELS`

Name to factory dict used by the command line runner. Add your factory here to run it from the shell.

## `src/config.py`

* `SEED` (5228) and `set_seed()`. The pipeline calls `set_seed()` before every fit. Use `SEED` for any `random_state`.
* `TIME_FIT_END`, `TIME_VALID_START`, `TIME_VALID_END`, `RANDOM_CV_FOLDS`: the split settings.
* `TARGET` (`"MONTHLY_RENT"`) and the results paths.

## Command line

```bash
python -m scripts.run_experiment --model lightgbm_default --features "G0+G1" --member A
python -m scripts.run_experiment --model ridge_default lightgbm_default --compare-splits --member A
```

`--features`, `--split` and `--target-mode` each take several values and run every combination. `--compare-splits` runs both splits for each model and prints the comparison table. `--no-log` writes nothing.
