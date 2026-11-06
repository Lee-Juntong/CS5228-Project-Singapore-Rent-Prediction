# CS5228 Team 55: Singapore HDB rent prediction

Predict `MONTHLY_RENT` for HDB rentals. Training covers Jan 2021 to Mar 2025 and the test set covers Mar 2025 to Jul 2026, so the task is forecasting future rents. The team plan is in [docs/final-report-plan.md](docs/final-report-plan.md).

## Setup

Use Python 3.10. Python 3.13 fails to install the pinned numpy.

```bash
git clone https://github.com/Lee-Juntong/CS5228-Project-Singapore-Rent-Prediction.git
cd CS5228-Project-Singapore-Rent-Prediction
python3.10 -m venv .venv
```

If `python3.10` is not found, as with Anaconda on Windows, check that `python --version` reports 3.10 and run `python -m venv .venv` instead.

Activate the environment. On Windows run `.venv\Scripts\activate`. On macOS or Linux run `source .venv/bin/activate`. Then install the packages:

```bash
pip install -r requirements.txt
```

This includes torch, which no current model uses. For analysis only, use `python -m pip install -r requirements-analysis.txt` instead.

On macOS, LightGBM also needs OpenMP (`brew install libomp`).

Then put the Kaggle files in `data/` and build the processed data as described in [data/README.md](data/README.md). Preprocessing takes about 20 seconds on Juntong's PC.

### Working together

* Run `git pull` before you push.
* Commit only your own files in `results/runs/` and `results/breakdowns/`.
* Pass your member letter as `--member` (or `member=` in Python) so every run shows who made it.

## Final model: repeat-rent blend

The final model is a five model blend, and `submissions/lag_blend5_G0G1_index.csv` is the final Kaggle submission (time holdout RMSE 431.0, public score 415.01). The prediction is the plain mean of five models on G0+G1 with the index target: tuned LightGBM, plus LightGBM, XGBoost, CatBoost and random forest with repeat-rent lag features. The lag features use earlier rents in the same block, street and flat type at gaps of about 12, 24 and 36 months.

Blend results, all on G0+G1. The time holdout is Oct 2024 to Mar 2025, and the 16 month backtest scores Dec 2023 to Mar 2025. The final model is in bold.

| model | target | time holdout RMSE | 16 month backtest RMSE | public score |
|---|---|---|---|---|
| lightgbm_tuned (no blend, for comparison) | index | 445.8 | 511.0 | 434.43 |
| two model blend: lightgbm_tuned + LightGBM with lags | index | 432.6 | 495.3 | 419.71 |
| **five model blend: the two above + XGBoost, CatBoost, random forest with lags** | index | **431.0** | **492.8** | **415.01** |
| index and raw mix: base and lag LightGBM on both targets | index + raw | 431.6 | 485.6 | 419.59 |

To reproduce it, finish [Setup](#setup) with the full `requirements.txt`, then run from the repository root:

```bash
python -c "from src.data_preprocessing import run_pipeline; run_pipeline()"
python -m scripts.lag_blend_submission
```

The second command takes about a minute and a half on Juntong's PC, and overwrites the committed CSV with a numerically matching one. See [Kaggle submission](#kaggle-submission) for validation, how to compare and restore the files, and [docs/repeat_rent_blend.md](docs/repeat_rent_blend.md) for the method and results.

## Baseline ladder (Phase 1)

Seven baselines are registered in the shared model registry. Run `python -m scripts.run_baselines --output results/member_b/new_run` for seven holdout fits and comparison plots. Learning curves, random CV and one selected final refit are optional. See the [pipeline guide](docs/pipeline_guide.md#baseline-ladder) for commands and hyperparameters, and the [baseline results](docs/baseline_results.md) for the recorded scores and figure locations.

## Analysis

[Analysis findings and reproduction](docs/analysis.md) covers EDA, residuals and model interpretation for selected LightGBM plus matched Ridge. Reference evidence is in `results/analysis/reference/`; WebP previews and PDF report figures are in `docs/figures/analysis/`. Full runs and models are regenerated locally and stay ignored.

Run the checks without Kaggle data: `python -m unittest discover -s checks -v`.

## Shared experiment setup (Phase 0)

Every model goes through one function, so all results are comparable. [docs/pipeline_guide.md](docs/pipeline_guide.md) lists every reusable function with its inputs, outputs and an example.

* **Split.** `time` (the main one): fit on Jan 2021 to Sep 2024, validate on Oct 2024 to Mar 2025. `random5`: shuffled 5 fold CV, reported once to show how much a random split overstates performance.
* **Metrics.** RMSE (primary, as on Kaggle), MAE, MAPE and mean bias. For the time split, RMSE per flat type and per validation month is saved too.
* **Feature groups.** Fixed in `src/features.py`. G0 core, G1 location, G2 block, G3 market. Named sets are `G0`, `G0+G1`, `G0+G1+G2` and `ALL`. Phase 1 uses `G0+G1`.
* **Trend handling.** `target_mode="raw"` trains on rent directly. `target_mode="index"` divides rent by a monthly rent index (training median per month, extended by a straight line fitted to the last 12 months), then multiplies back.
* **Seed.** 5228, set before every fit. The four added blend members have their own fixed seeds.
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
run_experiment("ridge_example", lambda: sklearn_tabular(Ridge(alpha=1.0)),
               feature_set="G0+G1", member="B", data=data)
```

Register it in `MODELS` in `src/models.py` if you want it on the command line.

### Random CV against the time holdout

`compare_splits` runs both splits for one model and scores the random CV on the same Oct 2024 to Mar 2025 rows as the time holdout. Run it once on your best model.

```python
from src.evaluation import compare_splits
compare_splits("ridge_example", lambda: sklearn_tabular(Ridge(alpha=1.0)), member="B", data=data)
```

Or from the shell: `python -m scripts.run_experiment --model lightgbm_default --compare-splits`. `split_comparison()` rebuilds the table from `results/results.csv`.

Current numbers on G0+G1, raw target, sorted by time holdout RMSE:

| model | time holdout RMSE | random CV RMSE, all rows | random CV RMSE, holdout rows | optimism |
|---|---|---|---|---|
| catboost_tuned | 448.8 | 475.4 | 451.0 | -2.1 |
| xgboost_tuned | 449.9 | 477.3 | 450.8 | -0.9 |
| lightgbm_tuned | 449.9 | 478.8 | 451.9 | -2.0 |
| catboost_default | 452.7 | 476.4 | 451.6 | 1.2 |
| xgboost_default | 457.2 | 479.9 | 456.5 | 0.7 |
| lightgbm_default | 457.3 | 478.9 | 455.3 | 2.0 |
| ridge_default | 574.9 | 509.2 | 501.6 | 73.3 |
| global_median | 837.1 | 752.0 | 794.6 | 42.5 |

`global_median` (predict the training median for every row) is the sanity floor; every learned model clears it by a wide margin. Compare each model's time holdout RMSE with its random CV holdout rows number: both score the same Oct 2024 to Mar 2025 rows, so the gap (optimism) is only the effect of the split. Random CV flatters Ridge by 73 RMSE because Ridge extends a straight month trend into the holdout and overpredicts by 330 on average there. The boosting models have optimism near zero: they keep the last level they saw, rents were flat in late 2024, so the random split barely helps them, and the tuned ones even score marginally better on the honest future window than on the random folds. The all rows random CV number is not comparable to the other two columns because it covers all of 2021 to 2025, including 2023 when rents varied most.

Tuned boosting models (30 Optuna trials each on the time holdout). Time holdout RMSE by feature set and target mode:

| model | feature set | raw target | index target |
|---|---|---|---|
| lightgbm_tuned | G0+G1 | 449.9 | 445.8 |
| xgboost_tuned | G0+G1 | 449.9 | 445.6 |
| lightgbm_tuned | G0+G1+G2 | 449.8 | 446.2 |
| catboost_tuned | G0+G1 | 448.8 | 449.0 |
| catboost_tuned | G0+G1+G2 | 448.5 | 448.4 |

Tuning cuts LightGBM from 457.3 to 449.9 and CatBoost from 452.7 to 448.8. The index target mode (dividing rent by a monthly rent index, decision 5) helps LightGBM by about 4 RMSE but barely moves CatBoost. **lightgbm_tuned on G0+G1 with index mode (445.8)** is the chosen single model. The final model and final Kaggle submission is the five model repeat-rent blend, `lag_blend5_G0G1_index.csv` (holdout 431.0, public 415.01, [docs/repeat_rent_blend.md](docs/repeat_rent_blend.md)). Tuned XGBoost (445.6) and the untuned random forest (444.3) are within 1.5 RMSE of it, which is a tie on one holdout. Index mode lets a tree model predict above the training range, but [docs/trend_handling.md](docs/trend_handling.md) shows on earlier periods that it hurts when rents stop rising. Adding the block group (G2) does not help either model here. Build the configs with `python -m scripts.tune_boosting`; the best parameters are saved to `configs/<model>_tuned.json`.

Feature-group ablation (tuned models, time holdout, raw target). The tuned hyperparameters are held fixed at the `G0+G1` values while the feature set varies, so each row isolates the contribution of a feature group:

| feature set | features | catboost_tuned | lightgbm_tuned |
|---|---|---|---|
| G0 (core) | 13 | 462.7 | 460.2 |
| G0+G1 (+location) | 39 | 448.8 | 449.9 |
| G0+G1+G2 (+block) | 44 | 448.5 | 449.8 |
| ALL (+market) | 54 | 450.1 | 452.4 |

Location (G1) is the only group that clearly helps: it cuts about 10 to 14 RMSE for both models. Block attributes (G2) add almost nothing (0.1 to 0.3). The market group (G3: current-month COE, CPI and stock values) makes both models worse, by 1.6 to 2.6 RMSE, even though those values would not be known at prediction time. The final selected recipe is `G0+G1` with index targets; the tiny raw-target improvement from G2 does not carry over to that LightGBM setup.

### Where results go

* `results/runs/<run_id>.csv`: one row per run. Commit these.
* `results/results.csv`: all runs combined, rebuilt after every run. If it conflicts in a merge, take either side and rerun `python -c "from src.evaluation import collect_results; collect_results()"`.
* `results/breakdowns/<run_id>.csv`: RMSE, MAE and MAPE per flat type and per month.
* `results/predictions/<run_id>.parquet`: validation predictions for error analysis. Not in git.

### Kaggle submission

The final model is the five model repeat-rent blend. It needs the full `requirements.txt`, including LightGBM, XGBoost and CatBoost, because `requirements-analysis.txt` alone is not enough. Keep the raw `data/train.csv` and `data/test.csv` next to the processed files. The blend rebuilds the block, street and flat type key from the raw row order.

Run these commands separately from the repository root:

```bash
python -m scripts.lag_blend_submission --validate
python -m scripts.lag_blend_submission
python -m scripts.lag_blend_submission --analyse
```

Each mode trains fresh models. Validation performs 14 fits: five index target models and two raw target LightGBM models in each of two time windows. The default command performs five fits on all training rows and writes `submissions/lag_blend5_G0G1_index.csv` with columns `Id,Predicted`. Analysis performs five holdout fits plus permutation scoring. None of them uploads to Kaggle or updates `results/kaggle_log.csv`.

All three modes overwrite committed files. Validation and analysis rewrite `results/member_a/repeat_rent/` and two report figures. The default command rewrites the committed final submission. Other steps rewrite committed files too: preprocessing rewrites `artifacts/data_audit/audit_summary.json`, `python -m scripts.trend_analysis` rewrites `report/figures/rent_index.pdf` and `report/figures/monthly_rmse.pdf`, and `python -m scripts.run_baselines` adds rows to `results/results.csv` and new files under `results/runs/` and `results/breakdowns/`. A rerun is not byte identical, so compare it with the committed file numerically (maximum difference below 1e-9) and do not commit it. A separate checkout needs its own copy of the data, because `data/` is mostly not in git (see [data/README.md](data/README.md)). Otherwise restore the generated files afterwards, which leaves other uncommitted work alone: `git checkout -- results/member_a/repeat_rent submissions/lag_blend5_G0G1_index.csv report/figures/lag_correlation.pdf report/figures/blend_importance.pdf report/figures/rent_index.pdf report/figures/monthly_rmse.pdf artifacts/data_audit/audit_summary.json results/results.csv`. Delete any new files under `results/runs/` and `results/breakdowns/` yourself.

`python -m scripts.report_tables` rebuilds `results.tex`, `ablation.tex` and `blend.tex` in `report/tables/`. In the same directory, `python -m scripts.trend_analysis` writes the trend, split and hyperparameter tables, and `python -m scripts.horizon_backtest --table-only` writes the horizon table, and `python -m scripts.report_appendix_tables` writes the baseline backtest, rent history and monthly index tables. Run all four commands from the repository root. See [the report build instructions](report/README.md), [the blend guide](docs/repeat_rent_blend.md) and [the reproduction check](docs/blend_reproduction.md).

The `submissions/` folder is created if it does not exist. Only the final submission, `submissions/lag_blend5_G0G1_index.csv`, is in git. Each submission is logged in `results/kaggle_log.csv` (date, member, model, feature set, target mode, validation RMSE, public leaderboard score, notes). Record the validation RMSE before reading the public score so the leaderboard does not drive model selection. The final model is the one with the lowest time holdout RMSE. The five model blend scores 431.0, against 431.6 for the index and raw mix and 432.6 for the two model blend, and its public score agrees with that ranking. Reproducing the CSV locally does not verify the public score.

### Saved model weights

The following command saves the **single LightGBM reference**, not the final blend:

```bash
python -m scripts.save_final_model --model lightgbm_tuned --features G0+G1 --target-mode index
```

It fits on all training rows and saves the fitted model to `models/` (gitignored) with a JSON provenance sidecar. See `models/README.md` to reload one. The blend script writes predictions only. It does not save its five estimators or the rent history behind the lag features, so reproducing the blend means running its script again.

## Layout

* `src/data_preprocessing.py`: cleaning, block join, geospatial and macro features
* `src/external_features.py`: CPI, rail line and tertiary features
* `src/features.py`, `src/evaluation.py`, `src/models.py`: shared experiment interface
* `scripts/run_experiment.py`: command line runner
* `scripts/trend_analysis.py`, `scripts/horizon_backtest.py`: trend tables, report figures and the 16 month backtest ([docs/trend_handling.md](docs/trend_handling.md))
* `scripts/lag_blend_submission.py`: repeat-rent features, final five model blend and its validation ([docs/repeat_rent_blend.md](docs/repeat_rent_blend.md))
* `scripts/report_tables.py`: Evaluation tables of the report
* `scripts/report_appendix_tables.py`: Appendix tables of the report
* `notebooks/`: EDA
* `templates/latex/`: course report template
