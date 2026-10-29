# Evaluation interface

`src/evaluation.py` provides shared splits, metrics, training and result logging. `src/evaluation_artifacts.py` provides learning curves, comparison plots and evaluation of prediction files. Follow the [setup instructions](README.md) and run commands from the repository root.

## Shared protocol

- Fit: January 2021 to September 2024, 132,538 rows.
- Validate: October 2024 to March 2025, 17,037 rows.
- Default features: `G0+G1`; target: raw `MONTHLY_RENT`; seed: `5228`.
- Fit all preprocessing and target-derived statistics on training rows only.

RMSE is the primary metric. RMSE, MAE and bias are in SGD per month; MAPE is a percentage. Bias is mean prediction minus observed rent, so positive values indicate overprediction. R-squared is also reported. Predictions must be finite one-dimensional arrays matching the targets; rent targets must be positive. Constant targets produce an undefined R-squared (`NaN`). Segment tables include sample counts and metrics by flat type and validation month.

## Evaluate a model factory

Provide a zero-argument factory returning a fresh object with `fit(X, y)` and `predict(X)`. `X` is a DataFrame with categorical columns stored as pandas `category`; numeric columns may contain missing values. This runnable example uses the published Ridge baseline; replace the factory and name for another model.

```python
from src.baselines import make_baseline_ridge
from src.evaluation import run_experiment
from src.features import load_modeling_data

data = load_modeling_data()
row = run_experiment(
    "ridge_example", make_baseline_ridge,
    feature_set="G0+G1", split="time", target_mode="raw",
    member="B", data=data,
    artifact_dir="results/custom/ridge_run",
)
```

The shared logger writes `results/runs/<run_id>.csv`, `results/breakdowns/<run_id>.csv` and validation predictions under `results/predictions/`. It rebuilds `results/results.csv`. When `artifact_dir` is supplied, it additionally saves the fitted model, fit IDs, training metrics and validation predictions, and verifies prediction equality after reloading the model. Use a new artifact directory.

Omit `artifact_dir` when saving checkpoints through a model-specific API. Custom classes used with joblib should live in importable Python modules. `log=False` disables file output and cannot be combined with `artifact_dir`.

## Evaluate predictions from any model

Export a CSV or Parquet file containing `ROW_ID,y_pred`. IDs must exactly match the shared holdout rows, with no duplicates, missing IDs or extra rows. Use the processed data's `ROW_ID`, not a reset DataFrame index. Row order may differ: the evaluator aligns predictions by ID.

```bash
python -m scripts.evaluate_predictions --predictions results/custom/holdout.csv --output results/custom/evaluation_run
```

This creates `metrics.csv` and `segment_metrics.csv` in a new directory. It checks alignment and numeric validity; the caller remains responsible for preventing training leakage. Test-set predictions cannot be scored without test labels.

## Learning curves and plots

```python
import pandas as pd
from src.evaluation_artifacts import learning_curve, plot_comparison

# Continue from the factory example above.
learning_curve("ridge_example", make_baseline_ridge, data[0],
               "results/custom/curves")
plot_comparison(pd.DataFrame([row]), "results/custom/figures")
```

Add result dictionaries for other models to the DataFrame for a multi-model comparison. Each metric gets a separate PNG and SVG. Use the same data version, holdout IDs, feature set and target mode. The comparison chart is intended for the shared time holdout.

Learning curves use raw targets, nested month-stratified samples of the training window and a fixed future holdout. They show training and validation error versus sample size, not epochs. An optional `fitted_full` argument must be the matching holdout bundle, never a model refitted on all labelled data.

## Random CV and feature ablation

Set `split="random5"` to use shuffled five-fold CV with seed 5228. Headline metrics are fold averages; `rmse_std` is the standard deviation of fold RMSEs, not a confidence interval. `rmse_window` pools out-of-fold predictions on the same October 2024 to March 2025 rows used by the time holdout. Compare time-holdout `rmse` against random-CV `rmse_window`, not against the all-month CV score.

For ablation, repeat `run_experiment` with `feature_set` set to `G0`, `G0+G1`, `G0+G1+G2` and `ALL`, keeping the model parameters fixed. Use separate artifact directories. The model must support each feature set; the fixed four-input KNN requires location features. Report ablation separately. `ALL` includes contemporaneous market variables that may not be available at forecast time.
