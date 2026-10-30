"""Evaluate registered baselines; opt in to curves, CV or one final refit."""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.baselines import BASELINE_MODELS, MODEL_NOTES, GroupMedian
from src.config import PROJECT_ROOT, SEED, TARGET
from src.evaluation import fit_predict, run_experiment, time_holdout_masks, _git_commit
from src.evaluation_artifacts import learning_curve, forest_curve, plot_comparison, plot_segments
from src.features import feature_columns, load_modeling_data
from src.models import MODELS



def markdown_table(table):
    def fmt(value):
        return f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join(["| " + " | ".join(table.columns) + " |",
                       "| " + " | ".join(["---"] * len(table.columns)) + " |"] +
                      ["| " + " | ".join(fmt(x) for x in row) + " |" for row in table.itertuples(index=False, name=None)])


def export_parameters(output, names):
    parameters = {}
    for name in names:
        model = MODELS[name]()
        params = model.get_params(deep=True) if hasattr(model, "get_params") else {"strategy": "median"}
        parameters[name] = {"notes": MODEL_NOTES[name], "parameters": {k: repr(v) for k, v in params.items()}}
    (output / "hyperparameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")


def write_report(output, table, comparison=None, refit_model=None, learning_curves=False):
    ordered = table.sort_values("rmse")
    best = ordered.iloc[0]
    lookup = table.set_index("model")
    paragraph = (f"On the chronological holdout, {best['model']} achieved the lowest RMSE "
                 f"among the evaluated baselines ({best['rmse']:.2f} SGD). ")
    if {"global_median", "group_median", "ridge", "random_forest", "decision_tree"} <= set(lookup.index):
        paragraph += (
            f"The global and group median RMSEs were {lookup.loc['global_median', 'rmse']:.2f} "
            f"and {lookup.loc['group_median', 'rmse']:.2f} SGD, respectively. "
            "All validation calendar quarters are unseen in training, so the quarter-specific lookup "
            "falls back to coarser historical groups, limiting its ability to track recent rents. "
            f"Ridge obtained {lookup.loc['ridge', 'rmse']:.2f} SGD, while the random forest obtained "
            f"{lookup.loc['random_forest', 'rmse']:.2f} SGD, a reduction of "
            f"{lookup.loc['ridge', 'rmse'] - lookup.loc['random_forest', 'rmse']:.2f} SGD relative to Ridge "
            f"and {lookup.loc['decision_tree', 'rmse'] - lookup.loc['random_forest', 'rmse']:.2f} SGD "
            "relative to the single tree. "
            f"Ridge's positive mean bias of {lookup.loc['ridge', 'bias']:.2f} SGD shows systematic "
            "overprediction, not merely noisy individual errors. The comparison supports evaluating nonlinear interactions "
            "and averaging trees, but linear and KNN models use the plan's restricted inputs, so their "
            "gaps do not isolate algorithm choice alone. Boosting's incremental benefit over random "
            "forest should be compared under the same protocol.")
    (output / "gaps.txt").write_text(paragraph + "\n", encoding="utf-8")
    columns = ["model", "rmse", "mae", "mape", "r2", "bias", "train_seconds", "predict_seconds"]
    text = ("# Member B baseline evaluation\n\n"
            "Training: Jan 2021-Sep 2024. Validation: Oct 2024-Mar 2025. "
            f"Training rows: {int(best['n_fit']):,}; validation rows: {int(best['n_valid']):,}. "
            "Raw rent target; seed 5228; fixed hyperparameters without a tuning search. "
            "RMSE and MAE are SGD/month, MAPE is percent, bias is prediction minus observed rent.\n\n"
            + markdown_table(ordered[columns]) + "\n\n" + paragraph + "\n\n"
            "## Comparison protocol\n\n"
            f"The shared candidate feature set is G0+G1 ({len(feature_columns('G0+G1'))} columns). Global median uses no inputs; "
            "group median uses town, flat type and calendar quarter; KNN uses latitude, longitude, "
            "area and month. Linear/Ridge retain flat type and remove the three area-derived columns, "
            "ordinal flat type and redundant lease year. Both tree models use all G0+G1 columns. "
            "All imputers, scalers, category encoders and target statistics are fitted on training rows only.\n\n"
            "## Saved models and optional diagnostics\n\n"
            "The results table evaluates models fitted only through September 2024. "
            "Holdout bundles include preprocessing. Test labels are unavailable, so no test-set "
            "metrics are reported. The six-month validation horizon does not establish accuracy "
            "at the end of the longer Kaggle test horizon.\n")
    if refit_model:
        text += f"\nOnly `{refit_model}` was refitted on all labelled rows for local test predictions.\n"
    else:
        text += "\nNo full-data refit or test prediction was requested.\n"
    if learning_curves:
        text += ("\nSample-size curves are in `curves/` (CSV, PNG and SVG). They use nested "
                 "month-stratified training samples and a fixed future holdout. Training errors "
                 "are in-sample; these are diagnostics, not repeated-CV confidence intervals. "
                 "The forest tree-count curve scores prefixes of the same ensemble.\n")
    if comparison is not None:
        text += "\n## Random CV comparison\n\n" + markdown_table(comparison) + "\n\n"
        split_row = comparison.iloc[0]
        text += (f"The random-CV RMSE on the same validation-window rows is "
                 f"{split_row['random5_rmse_same_window']:.2f} SGD versus {split_row['time_rmse']:.2f} "
                 f"for the time holdout (optimism = {split_row['optimism']:.2f} SGD). ")
        text += ("This run does not show an optimism advantage for random CV. " if split_row["optimism"] <= 0 else
                 "Random CV gives a lower error on those same rows in this run. ")
        text += "The all-month random-CV RMSE covers a different target distribution and is not directly comparable.\n\n"
        text += ("Compare time RMSE with random-CV RMSE on the same holdout rows. The CV training folds "
                 "include later dates and may include near-identical property records. A positive optimism "
                 "gap measures this split effect, not a deployable forecast gain. Model selection and the "
                 "reported holdout score use the same validation window; the best score is exploratory.\n")
    (output / "evaluation_report.md").write_text(text, encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "member_b" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--models", nargs="+", choices=BASELINE_MODELS, default=list(BASELINE_MODELS))
    parser.add_argument("--learning-curves", action="store_true", help="Add sample-size curves (four extra fits per model)")
    parser.add_argument("--random-cv", action="store_true", help="Add five-fold CV for the best holdout model")
    parser.add_argument("--refit", choices=["best", *BASELINE_MODELS], help="Refit one evaluated model on all labelled rows; best selects by holdout RMSE")
    args = parser.parse_args(argv)
    args.models = list(dict.fromkeys(args.models))
    if args.refit and args.refit != "best" and args.refit not in args.models:
        parser.error("--refit must name a model included in --models (or use best)")
    return args


def refit_final(name, train, test, output):
    print(f"REFIT ALL LABELLED ROWS: {name}", flush=True)
    pred, fit_s, predict_s, final = fit_predict(MODELS[name], train, test,
                                               feature_columns("G0+G1"), return_fitted=True)
    destination = output / "full_fit" / name
    destination.mkdir(parents=True)
    joblib.dump(final, destination / "model.joblib", compress=3)
    np.testing.assert_allclose(joblib.load(destination / "model.joblib").predict(test), pred, rtol=1e-12, atol=1e-8)
    pd.DataFrame({"ROW_ID": test.ROW_ID.to_numpy(), "y_pred": pred}).to_csv(destination / "test_predictions.csv", index=False)
    pd.DataFrame([{"n_fit": len(train), "n_predict": len(test), "fit_seconds": fit_s,
                   "predict_seconds": predict_s, "serialization_verified": True}]).to_csv(destination / "metadata.csv", index=False)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT: {output}", flush=True)
    data = load_modeling_data()
    train, test = data
    fit, valid = time_holdout_masks(train)
    if train.ROW_ID.duplicated().any() or test.ROW_ID.duplicated().any() or np.any(fit & valid):
        raise ValueError("Invalid IDs or overlapping train/validation masks")
    packages = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow", "joblib", "threadpoolctl")}
    manifest = {"seed": SEED, "git_commit": _git_commit(), "python": platform.python_version(), "packages": packages,
                "n_train": len(train), "n_fit": int(fit.sum()), "n_valid": int(valid.sum()), "n_test": len(test),
                "feature_columns": feature_columns("G0+G1"), "arguments": vars(args) | {"output": str(output)},
                "completed": False, "refit_model": None}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    export_parameters(output, args.models)
    train.loc[valid, ["ROW_ID", "RENT_APPROVAL_DATE", "FLAT_TYPE", TARGET]].to_csv(output / "holdout_truth.csv", index=False)
    rows, segments = [], []
    with threadpool_limits(limits=8):
        for name in args.models:
            print(f"FIT: {name}", flush=True)
            folder = output / "holdout" / name
            row = run_experiment(name, MODELS[name], member="B", data=data,
                                 notes=MODEL_NOTES[name], artifact_dir=folder)
            fitted = joblib.load(folder / "time" / "model.joblib")
            row["effective_n_features"] = len(fitted.model.named_steps["columns"].columns_) if hasattr(fitted.model, "named_steps") and "columns" in fitted.model.named_steps else (0 if name == "global_median" else 3 if name == "group_median" else len(fitted.columns))
            train_metrics = pd.read_csv(folder / "time" / "train_metrics.csv").iloc[0]
            row.update({f"train_{k}": train_metrics[k] for k in ("rmse", "mae", "r2")})
            rows.append(row)
            table = pd.DataFrame(rows)
            table.to_csv(output / "results_table.csv", index=False)
            segment = pd.read_csv(folder / "segment_metrics.csv")
            segment["model"] = name
            segments.append(segment)
            if isinstance(fitted.model, GroupMedian):
                _, levels = fitted.model.predict_with_levels(train.loc[valid, fitted.columns])
                pd.Series(levels).value_counts().rename_axis("fallback_level").to_csv(folder / "fallback_counts.csv", header=["n"])
            if args.learning_curves:
                learning_curve(name, MODELS[name], train, output / "curves", fitted_full=fitted)
                if name == "random_forest":
                    forest_curve(fitted, train, output / "curves")
        table = pd.DataFrame(rows)
        plot_comparison(table, output / "figures")
        all_segments = pd.concat(segments, ignore_index=True)
        all_segments.to_csv(output / "segment_metrics.csv", index=False)
        plot_segments(all_segments, output / "figures")
        comparison = None
        best = table.sort_values("rmse").iloc[0]
        if args.random_cv:
            name = best["model"]
            print(f"RANDOM FIVE-FOLD CV: {name}", flush=True)
            cv = run_experiment(name, MODELS[name], split="random5", member="B", data=data,
                                notes="Best Phase 1 baseline; same fixed hyperparameters; all OOF predictions saved.")
            shutil.copy2(PROJECT_ROOT / "results" / "predictions" / f"{cv['run_id']}.parquet", output / "random5_predictions.parquet")
            comparison = pd.DataFrame([{"model": name, "time_rmse": best["rmse"], "random5_rmse_all": cv["rmse"],
                                       "random5_rmse_std": cv["rmse_std"], "random5_rmse_same_window": cv["rmse_window"],
                                       "optimism": best["rmse"] - cv["rmse_window"]}])
            comparison.to_csv(output / "split_comparison.csv", index=False)
        if args.refit:
            name = best["model"] if args.refit == "best" else args.refit
            refit_final(name, train, test, output)
            manifest["refit_model"] = name
        write_report(output, table, comparison, manifest["refit_model"], args.learning_curves)
    manifest["completed"] = True
    manifest["serialization_checks"] = "Every saved holdout and requested full-fit model reloaded; predictions matched."
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table[["model", "rmse", "mae", "mape", "r2"]].to_string(index=False), flush=True)
    print(f"COMPLETE: {output}", flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
