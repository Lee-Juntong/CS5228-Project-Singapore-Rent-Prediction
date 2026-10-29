# Baseline training and evaluation

Phase 1 includes seven models: global median, group median, linear regression, Ridge, KNN, decision tree and random forest.

- [Evaluation interface](evaluation.md)
- [Hyperparameters](hyperparameters.md)
- [Results and figures](results_report.md)

## Environment

Use Python 3.14. The reference run used Python 3.14.7; the accompanying requirements pin its core package versions and include Jinja2 for the LaTeX table export. Use this requirements file for the baseline workflow; the repository-root `requirements.txt` targets the original Python 3.10 environment.

Run all commands from the repository root. These instructions assume this documentation is located in `doc/`.

Check the Python version and install the baseline dependencies:

```bash
python --version
python -m pip install --upgrade pip
python -m pip install -r doc/requirements-baselines.txt
```

These dependencies cover preprocessing, the seven baselines, evaluation and plots. Install additional packages separately for other models such as LightGBM, CatBoost or TabM. Package versions match the recorded run; installation through pip in a fresh environment has not been independently validated.

## Prepare the data

Download the competition files from [Kaggle](https://www.kaggle.com/competitions/cs-5228-2610-project/data) and arrange them as follows:

```text
data/
  train.csv
  test.csv
  auxiliary/
    sg-hdb-block.csv
    sg-mrt-stations.csv
    sg-schools.csv
    sg-shopping-malls.csv
    sg-coe-prices.csv
    sg-stock-prices.csv
```

Keep the repository's `data/auxiliary/sg-cpi.csv`, `sg-tertiary-institutions.csv` and `data/derived/block_extra_features.csv`. The cached block table avoids downloading the rail geometry again. `example-submission.csv` is optional.

```bash
python -c "from src.data_preprocessing import run_pipeline; run_pipeline(audit_dir='artifacts/member_b_data_audit')"
```

This creates `data/processed/train_features.parquet` and `test_features.parquet`. If these files already exist for the same data version, skip preprocessing.

## Run the baseline ladder

```bash
python -u -m scripts.run_baselines --output results/member_b/new_run
```

Choose a new output directory. The default run produces all seven holdout models, learning curves, full-data refits, test predictions, comparison plots, a results table and random five-fold CV for the best baseline. It also logs results through the shared evaluation interface with member `B`.

To run selected models:

```bash
python -u -m scripts.run_baselines --models ridge random_forest --output results/member_b/selected_run
```

Optional flags: `--skip-curves`, `--skip-refit` and `--skip-random-cv`. Use `--resume` with the same output directory only after an interrupted run and without changing the model code or parameters. Feature ablation is optional via `--ablation` and is separate from the reported Phase 1 results.

## Saved models and predictions

Within each run directory:

| Path | Contents |
| --- | --- |
| `holdout/<model>/time/model.joblib` | Model fitted only through September 2024, including preprocessing |
| `holdout/<model>/time/validation_predictions.csv` | `ROW_ID,y_true,y_pred` for the holdout |
| `full_fit/<model>/model.joblib` | Model refitted on all labelled rows |
| `full_fit/<model>/test_predictions.csv` | `ROW_ID,y_pred` for all 50,000 test rows |
| `results_table.csv`, `split_comparison.csv` | Holdout and CV results |
| `curves/`, `figures/` | Learning curves and metric comparisons, with PNG and SVG outputs |

Model bundles and row-level predictions are generated locally and are excluded from Git by the current ignore rules. After training, reload a full-fit model to predict without retraining:

```bash
python -m scripts.predict_saved_model --model results/member_b/new_run/full_fit/random_forest/model.joblib --output results/member_b/reloaded_test.csv
```

Use a new output filename. Bundles expect the engineered DataFrame returned by `load_modeling_data()`, not raw Kaggle rows. Load them with matching package versions. Use holdout models for validation; full-fit models have already seen the holdout labels. No command above uploads predictions to Kaggle.

## Checks

```bash
python -m unittest discover -s checks -v
python -m checks.verify_run results/member_b/new_run
```

The artifact check requires a complete seven-model run with curves, refits and random CV enabled.
