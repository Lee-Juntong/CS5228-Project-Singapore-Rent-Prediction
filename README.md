# CS5228 Team 55: Singapore HDB rent prediction

Predict `MONTHLY_RENT` for HDB rentals. Training covers Jan 2021 to Mar 2025 and the test set covers Mar 2025 to Jul 2026, so the task is forecasting future rents. The team plan is in [docs/final-report-plan.md](docs/final-report-plan.md).

## Setup

Use Python 3.10. Python 3.13 fails to install the pinned numpy.

```bash
git clone https://github.com/Lee-Juntong/CS5228-Project-Singapore-Rent-Prediction.git
cd CS5228-Project-Singapore-Rent-Prediction
python3.10 -m venv .venv
```

Activate the environment. On Windows run `.venv\Scripts\activate`. On macOS or Linux run `source .venv/bin/activate`. Then install the packages:

```bash
pip install -r requirements.txt
```

This includes torch (CPU build) for neural models such as TabM.

Then put the Kaggle files in `data/` and build the processed data as described in [data/README.md](data/README.md). Preprocessing takes about 20 seconds on Juntong's PC.

### Working together

* Run `git pull` before you push.
* Commit only your own files in `results/runs/` and `results/breakdowns/`.
* Pass your member letter as `--member` (or `member=` in Python) so every run shows who made it.

## Shared experiment setup (Phase 0)

Every model goes through one function, so all results are comparable. [docs/pipeline_guide.md](docs/pipeline_guide.md) lists every reusable function with its inputs, outputs and an example.

* **Split.** `time` (the main one): fit on Jan 2021 to Sep 2024, validate on Oct 2024 to Mar 2025. `random5`: shuffled 5 fold CV, reported once to show how much a random split overstates performance.
* **Metrics.** RMSE (primary, as on Kaggle), MAE, MAPE and mean bias. For the time split, RMSE per flat type and per validation month is saved too.
* **Feature groups.** Fixed in `src/features.py`. G0 core, G1 location, G2 block, G3 market. Named sets are `G0`, `G0+G1`, `G0+G1+G2` and `ALL`. Phase 1 uses `G0+G1`.
* **Trend handling.** `target_mode="raw"` trains on rent directly. `target_mode="index"` divides rent by a monthly rent index (training median per month, extended by a straight line fitted to the last 12 months), then multiplies back.
* **Seed.** 5228, set before every fit.
* **Feature handling table.** [docs/feature_handling.md](docs/feature_handling.md).

### Run a registered model

```bash
python -m scripts.run_experiment --model lightgbm_default --features "G0+G1" --member A
```

`--features`, `--split` and `--target-mode` each take several values and run every combination. Add `--no-log` for a quick trial that should not enter the results table.

### Add your own model

Write a function with no arguments that returns an object with `fit(X, y)` and `predict(X)`. X is a DataFrame with categorical columns as pandas `category` and possible NaN in numeric columns. For scikit-learn models, `sklearn_tabular` in `src/models.py` adds one hot encoding, imputation and scaling.

```python
from sklearn.linear_model import Ridge
from src.evaluation import run_experiment
from src.features import load_modeling_data
from src.models import sklearn_tabular

data = load_modeling_data()
run_experiment("ridge", lambda: sklearn_tabular(Ridge(alpha=1.0)),
               feature_set="G0+G1", member="B", data=data)
```

Register it in `MODELS` in `src/models.py` if you want it on the command line.

### Random CV against the time holdout

`compare_splits` runs both splits for one model and scores the random CV on the same Oct 2024 to Mar 2025 rows as the time holdout. Run it once on your best model.

```python
from src.evaluation import compare_splits
compare_splits("ridge", lambda: sklearn_tabular(Ridge(alpha=1.0)), member="B", data=data)
```

Or from the shell: `python -m scripts.run_experiment --model lightgbm_default --compare-splits`. `split_comparison()` rebuilds the table from `results/results.csv`.

Current numbers on G0+G1, raw target, sorted by time holdout RMSE:

| model | time holdout RMSE | random CV RMSE, all rows | random CV RMSE, holdout rows | optimism |
|---|---|---|---|---|
| catboost_tuned | 448.8 | 475.4 | 451.0 | -2.1 |
| lightgbm_tuned | 449.9 | 478.8 | 451.9 | -2.0 |
| catboost_default | 452.7 | 476.4 | 451.6 | 1.2 |
| lightgbm_default | 457.3 | 478.9 | 455.3 | 2.0 |
| ridge_default | 574.9 | 509.2 | 501.6 | 73.3 |
| global_median | 837.1 | 752.0 | 794.6 | 42.5 |

`global_median` (predict the training median for every row) is the sanity floor; every learned model clears it by a wide margin. Compare each model's time holdout RMSE with its random CV holdout rows number: both score the same Oct 2024 to Mar 2025 rows, so the gap (optimism) is only the effect of the split. Random CV flatters Ridge by 73 RMSE because Ridge extends a straight month trend into the holdout and overpredicts by 330 on average there. The boosting models have optimism near zero: they keep the last level they saw, rents were flat in late 2024, so the random split barely helps them, and the tuned ones even score marginally better on the honest future window than on the random folds. The all rows random CV number is not comparable to the other two columns because it covers all of 2021 to 2025, including 2023 when rents varied most.

Tuned boosting models (30 Optuna trials each on the time holdout, G0+G1). Time holdout RMSE:

| model | raw target | index target |
|---|---|---|
| catboost_tuned | 448.8 | 449.0 |
| lightgbm_tuned | 449.9 | 445.8 |

Tuning cuts LightGBM from 457.3 to 449.9 and CatBoost from 452.7 to 448.8, so the two land close together. The index target mode (dividing rent by a monthly rent index, decision 5) helps LightGBM most: tuned LightGBM on index is the best single result at 445.8, while it barely moves CatBoost. Build the configs with `python -m scripts.tune_boosting`; the best parameters are saved to `configs/<model>_tuned.json`.

### Where results go

* `results/runs/<run_id>.csv`: one row per run. Commit these.
* `results/results.csv`: all runs combined, rebuilt after every run. If it conflicts in a merge, take either side and rerun `python -c "from src.evaluation import collect_results; collect_results()"`.
* `results/breakdowns/<run_id>.csv`: RMSE, MAE and MAPE per flat type and per month.
* `results/predictions/<run_id>.parquet`: validation predictions for error analysis. Not in git.

### Kaggle submission

```python
from src.evaluation import make_submission
from src.models import make_lightgbm
make_submission(make_lightgbm, "G0+G1", path="submissions/lightgbm.csv")
```

The `submissions/` folder is created if it does not exist. It is not in git.

## Layout

* `src/data_preprocessing.py`: cleaning, block join, geospatial and macro features
* `src/external_features.py`: CPI, rail line and tertiary features
* `src/features.py`, `src/evaluation.py`, `src/models.py`: shared experiment interface
* `scripts/run_experiment.py`: command line runner
* `notebooks/`: EDA
* `templates/latex/`: course report template
