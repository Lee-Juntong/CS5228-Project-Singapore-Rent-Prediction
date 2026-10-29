"""Reproduce Member B models, holdout evaluation, curves and local predictions."""

import argparse
from datetime import datetime
import hashlib
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
from src.config import PROJECT_ROOT, PROCESSED_DIR, SEED, TARGET
from src.evaluation import fit_predict, run_experiment, time_holdout_masks, _git_commit
from src.evaluation_artifacts import learning_curve, forest_curve, plot_comparison, plot_segments
from src.features import feature_columns, load_modeling_data


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(table):
    def fmt(value):
        return f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join(["| " + " | ".join(table.columns) + " |",
                       "| " + " | ".join(["---"] * len(table.columns)) + " |"] +
                      ["| " + " | ".join(fmt(x) for x in row) + " |" for row in table.itertuples(index=False, name=None)])


def export_parameters(output, names):
    parameters = {}
    for name in names:
        model = BASELINE_MODELS[name]()
        params = model.get_params(deep=True) if hasattr(model, "get_params") else {"strategy": "median"}
        parameters[name] = {"notes": MODEL_NOTES[name], "parameters": {k: repr(v) for k, v in params.items()}}
    (output / "hyperparameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")


def write_report(output, table, comparison=None):
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
            "forest remains to be measured using Member C's models under the same protocol.")
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
            "## Learning curves and saved models\n\n"
            "The sample-size curves use a single deterministic, nested, month-stratified sample sequence "
            "from the training window and a fixed future holdout. They are diagnostics, not repeated-CV "
            "confidence intervals. Training errors are in-sample; KNN with distance weighting and group "
            "lookups can have particularly optimistic training errors. The forest curve evaluates prefixes "
            "of the saved forest; it is not gradient-descent convergence or early stopping.\n\n"
            "The results table evaluates models fitted only through September 2024. Each final model "
            "is separately refitted on all labelled rows for local test predictions. Test labels are "
            "unavailable, so no test-set metrics are reported. The six-month validation horizon does "
            "not establish accuracy at the end of the much longer Kaggle test horizon.\n")
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
    ordered[columns].to_latex(output / "results_table.tex", index=False, float_format="%.2f")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "member_b" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--models", nargs="+", choices=BASELINE_MODELS, default=list(BASELINE_MODELS))
    parser.add_argument("--skip-curves", action="store_true")
    parser.add_argument("--skip-refit", action="store_true")
    parser.add_argument("--skip-random-cv", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume a run using completed model results and full-fit artifacts")
    parser.add_argument("--ablation", action="store_true", help="Optional Phase 2 feature ablation for selected best two baselines")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume and not (output / "manifest.json").exists():
        raise FileNotFoundError("--resume requires an existing run manifest")
    output.mkdir(parents=True, exist_ok=args.resume)
    print(f"OUTPUT: {output}", flush=True)
    data = load_modeling_data()
    train, test = data
    fit, valid = time_holdout_masks(train)
    if train.ROW_ID.duplicated().any() or test.ROW_ID.duplicated().any() or np.any(fit & valid):
        raise ValueError("Invalid IDs or overlapping train/validation masks")
    packages = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow", "joblib", "threadpoolctl")}
    sources = list((PROJECT_ROOT / "src").glob("*.py")) + list((PROJECT_ROOT / "scripts").glob("*.py"))
    manifest = {"seed": SEED, "git_commit": _git_commit(), "python": platform.python_version(), "packages": packages,
                "n_train": len(train), "n_fit": int(fit.sum()), "n_valid": int(valid.sum()), "n_test": len(test),
                "feature_columns": feature_columns("G0+G1"), "arguments": vars(args) | {"output": str(output)},
                "source_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sources},
                "data_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in
                                list(PROCESSED_DIR.glob("*.parquet")) + list((PROJECT_ROOT / "data").glob("*.csv")) +
                                list((PROJECT_ROOT / "data" / "auxiliary").glob("*.csv")) +
                                list((PROJECT_ROOT / "data" / "derived").glob("*.csv"))}}
    if args.resume:
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for key in ("data_sha256", "packages", "feature_columns", "seed"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Cannot resume with changed {key}")
        manifest["previous_execution"] = previous
    snapshot = output / "source_snapshot" / datetime.now().strftime("%Y%m%d-%H%M%S")
    for source in sources:
        target = snapshot / source.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest["source_snapshot"] = str(snapshot.relative_to(output))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "requirements-lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in packages.items()) + "\n", encoding="utf-8")
    export_parameters(output, args.models)
    train.loc[valid, ["ROW_ID", "RENT_APPROVAL_DATE", "FLAT_TYPE", TARGET]].to_csv(output / "holdout_truth.csv", index=False)
    rows, segments = [], []
    with threadpool_limits(limits=8):
        for name in args.models:
            print(f"FIT: {name}", flush=True)
            folder = output / "holdout" / name
            if args.resume and (folder / "result.csv").exists():
                print(f"  using completed holdout: {name}", flush=True)
                row = pd.read_csv(folder / "result.csv").iloc[0].to_dict()
            else:
                row = run_experiment(name, BASELINE_MODELS[name], member="B", data=data,
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
            if not args.skip_curves:
                curve_file = output / "curves" / f"{name}.csv"
                curve_complete = (args.resume and curve_file.exists() and len(pd.read_csv(curve_file)) == 5
                                  and curve_file.with_suffix(".png").exists() and curve_file.with_suffix(".svg").exists())
                if not curve_complete:
                    learning_curve(name, BASELINE_MODELS[name], train, output / "curves", fitted_full=fitted)
                if name == "random_forest":
                    forest_curve(fitted, train, output / "curves")
            if not args.skip_refit and not (args.resume and (output / "full_fit" / name / "metadata.csv").exists()):
                print(f"REFIT ALL LABELLED ROWS: {name}", flush=True)
                pred, fit_s, predict_s, final = fit_predict(BASELINE_MODELS[name], train, test,
                                                          feature_columns("G0+G1"), return_fitted=True)
                destination = output / "full_fit" / name
                destination.mkdir(parents=True)
                joblib.dump(final, destination / "model.joblib", compress=3)
                np.testing.assert_allclose(joblib.load(destination / "model.joblib").predict(test), pred, rtol=1e-12, atol=1e-8)
                pd.DataFrame({"ROW_ID": test.ROW_ID.to_numpy(), "y_pred": pred}).to_csv(destination / "test_predictions.csv", index=False)
                pd.DataFrame([{"n_fit": len(train), "n_predict": len(test), "fit_seconds": fit_s,
                               "predict_seconds": predict_s, "serialization_verified": True}]).to_csv(destination / "metadata.csv", index=False)
        table = pd.DataFrame(rows)
        plot_comparison(table, output / "figures")
        all_segments = pd.concat(segments, ignore_index=True)
        all_segments.to_csv(output / "segment_metrics.csv", index=False)
        plot_segments(all_segments, output / "figures")
        comparison = None
        best = table.sort_values("rmse").iloc[0]
        if not args.skip_random_cv:
            name = best["model"]
            print(f"RANDOM FIVE-FOLD CV: {name}", flush=True)
            cv = run_experiment(name, BASELINE_MODELS[name], split="random5", member="B", data=data,
                                notes="Best Phase 1 baseline; same fixed hyperparameters; all OOF predictions saved.")
            shutil.copy2(PROJECT_ROOT / "results" / "predictions" / f"{cv['run_id']}.parquet", output / "random5_predictions.parquet")
            comparison = pd.DataFrame([{"model": name, "time_rmse": best["rmse"], "random5_rmse_all": cv["rmse"],
                                       "random5_rmse_std": cv["rmse_std"], "random5_rmse_same_window": cv["rmse_window"],
                                       "optimism": best["rmse"] - cv["rmse_window"]}])
            comparison.to_csv(output / "split_comparison.csv", index=False)
        if args.ablation:
            ablations = []
            eligible = table[table.model.isin(["linear_regression", "ridge", "decision_tree", "random_forest"])]
            for name in eligible.sort_values("rmse").head(2).model:
                for features in ("G0", "G0+G1", "G0+G1+G2", "ALL"):
                    if features == "G0+G1":
                        ablations.append(table[table.model == name].iloc[0].to_dict())
                    else:
                        ablations.append(run_experiment(name, BASELINE_MODELS[name], feature_set=features,
                            member="B", data=data, notes="Optional baseline ablation; ALL includes contemporaneous market data.",
                            artifact_dir=output / "ablation" / name / features))
                    pd.DataFrame(ablations).to_csv(output / "ablation_results.csv", index=False)
        write_report(output, table, comparison)
    manifest["completed"] = True
    manifest["serialization_checks"] = "Every saved holdout and full-fit model reloaded; predictions matched."
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table[["model", "rmse", "mae", "mape", "r2"]].to_string(index=False), flush=True)
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
