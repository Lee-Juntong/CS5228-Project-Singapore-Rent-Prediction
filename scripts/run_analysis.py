"""Generate analysis report assets; never retune or submit models."""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
import platform
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT, RUNS_DIR, PREDICTIONS_DIR, SEED, TARGET
from src.evaluation import fit_predict, regression_metrics, time_holdout_masks
from src.features import feature_columns, load_modeling_data
from src.analysis_report import (
    REFERENCE_RUN, file_hash, write_full_report, write_json,
)
from src.models import make_lightgbm_tuned, make_ridge


def verify_reference(train, predictions, reference):
    """Check reproducibility before describing a refit as the selected run."""
    from src.analysis_residuals import align_holdout_predictions

    aligned = align_holdout_predictions(train, predictions)
    fit, _ = time_holdout_masks(train)
    protocol = {"model": "lightgbm_tuned", "feature_set": "G0+G1", "split": "time",
                "target_mode": "index", "seed": SEED, "n_features": len(feature_columns("G0+G1")),
                "n_fit": int(fit.sum()), "n_valid": len(aligned)}
    for key, expected in protocol.items():
        if reference[key] != expected:
            raise ValueError(f"Source-run protocol differs at {key}: {reference[key]} != {expected}")
    metrics = regression_metrics(aligned.y_true, aligned.y_pred)
    for metric in ("rmse", "mae", "mape", "bias"):
        if not np.isclose(metrics[metric], reference[metric], rtol=0, atol=.05):
            raise ValueError(f"Refit {metric}={metrics[metric]:.6f} disagrees with source run {reference[metric]:.6f}; investigate data/configuration before interpretation")
    saved = PREDICTIONS_DIR / f"{REFERENCE_RUN}.parquet"
    agreement = {"level": "aggregate metrics only", "absolute_metric_tolerance": .05,
                 "source_predictions_available": saved.exists(), "metrics": metrics,
                 "limitation": "Matching aggregate metrics does not establish identical row-level predictions."}
    if saved.exists():
        original = align_holdout_predictions(train, pd.read_parquet(saved))
        paired = aligned[["ROW_ID", "y_pred"]].merge(original[["ROW_ID", "y_pred"]],
                    on="ROW_ID", validate="one_to_one", suffixes=("_refit", "_source"))
        np.testing.assert_allclose(paired.y_pred_refit, paired.y_pred_source, rtol=1e-7, atol=1e-4)
        agreement.update(level="ROW_ID-aligned prediction agreement", prediction_atol_sgd=1e-4,
                         prediction_rtol=1e-7, limitation=None)
    return agreement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Fresh output directory; existing directories are never overwritten")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 10):
        parser.error("Use Python 3.10; activate the project environment described in README.md")
    if args.sample_size < 2 or args.permutation_repeats < 2:
        parser.error("Sample size and permutation repeats must each be at least 2")
    data_paths = [PROCESSED_DIR / f"{part}_features.parquet" for part in ("train", "test")]
    config_path = PROJECT_ROOT / "configs/lightgbm_tuned.json"
    source_path = RUNS_DIR / f"{REFERENCE_RUN}.csv"
    required_paths = [*data_paths, config_path, source_path,
                      PROJECT_ROOT / "data/derived/block_extra_features.csv",
                      PROJECT_ROOT / "data/auxiliary/sg-cpi.csv"]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required_paths if not p.is_file()]
    if missing:
        parser.error("Missing " + ", ".join(missing) + ". Prepare the competition inputs first; see docs/analysis.md.")
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
    output = args.output or PROJECT_ROOT / "results/analysis" / stamp
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / "results/analysis").resolve()):
        parser.error("Full-run output must be under results/analysis so large row-level files remain ignored")
    if output.is_relative_to((PROJECT_ROOT / "results/analysis/reference").resolve()):
        parser.error("reference/ is reserved for compact recorded results; choose another output directory")
    if output.exists():
        parser.error(f"Output already exists; choose a fresh directory: {output}")
    output.mkdir(parents=True)
    command = [sys.executable, "-m", "scripts.run_analysis",
               *(sys.argv[1:] if argv is None else argv)]
    manifest = {
        "status": "running", "created_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "command": subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command),
        "command_args": command,
        "python": platform.python_version(), "source_run_id": REFERENCE_RUN,
        "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow", "matplotlib", "seaborn", "joblib")},
        "requirements_sha256": file_hash(PROJECT_ROOT / "requirements-analysis.txt"),
        "code_sha256": {str(p.relative_to(PROJECT_ROOT).as_posix()): file_hash(p) for p in
            [Path(__file__).resolve(), *(PROJECT_ROOT / "src").glob("analysis_*.py"),
             *(PROJECT_ROOT / "src" / name for name in ("evaluation.py", "features.py", "models.py", "config.py",
                                                       "external_features.py", "data_preprocessing.py"))]},
    }
    write_json(output / "manifest.json", manifest)
    try:
        from src.analysis_eda import generate_eda, profile_keys
        from src.analysis_interpretation import generate_interpretation
        from src.analysis_residuals import align_holdout_predictions, generate_residuals

        print("Loading the shared frozen modeling data", flush=True)
        train, test = load_modeling_data()
        fit_mask, valid_mask = time_holdout_masks(train)
        if train.ROW_ID.isna().any() or train.ROW_ID.duplicated().any():
            raise ValueError("Training ROW_ID must be unique and nonmissing")
        if not np.all(fit_mask | valid_mask) or np.any(fit_mask & valid_mask):
            raise ValueError("Training rows must partition into the shared fit and validation windows")
        fit, valid = train.loc[fit_mask], train.loc[valid_mask]
        source_table = pd.read_csv(source_path, keep_default_na=False)
        if len(source_table) != 1 or source_table.iloc[0]["run_id"] != REFERENCE_RUN:
            raise ValueError("Expected exactly one matching source-run record")
        source = source_table.iloc[0].to_dict()
        columns = feature_columns("G0+G1")
        bundles, predictions, timing = {}, {}, {}
        for label, factory in (("boosting", make_lightgbm_tuned), ("ridge", make_ridge)):
            print(f"Reproducing {label} on the shared fit rows, G0+G1, index target", flush=True)
            pred, fit_s, predict_s, fitted = fit_predict(factory, fit, valid, columns, "index", return_fitted=True)
            prediction = pd.DataFrame({"ROW_ID": valid.ROW_ID.to_numpy(), "fold": "time",
                                       "y_true": valid[TARGET].to_numpy(), "y_pred": pred})
            aligned = align_holdout_predictions(train, prediction)
            bundles[label], predictions[label] = fitted, prediction
            timing[label] = {"fit_seconds": fit_s, "predict_seconds": predict_s}
            prediction.to_parquet(output / f"{label}_validation.parquet", index=False)
            joblib.dump(fitted, output / f"{label}.joblib", compress=3)
            restored = joblib.load(output / f"{label}.joblib")
            np.testing.assert_allclose(restored.predict(valid), aligned.y_pred, rtol=1e-12, atol=1e-8)
            if label == "boosting":
                manifest["reference_agreement"] = verify_reference(train, prediction, source)
        fit[["ROW_ID"]].to_csv(output / "fit_ids.csv", index=False)
        manifest.update(feature_set="G0+G1", columns=columns, target_mode="index", split="time", seed=SEED,
                        source_run=source, n_fit=len(fit), n_valid=len(valid), timings=timing,
                        comparison_factory="src.models.make_ridge (Ridge alpha=1, all 39 inputs)",
                        configuration_sha256=file_hash(config_path), source_record_sha256=file_hash(source_path),
                        primary_parameters=bundles["boosting"].model.get_params(),
                        data_sha256={p.relative_to(PROJECT_ROOT).as_posix(): file_hash(p) for p in
                            [*data_paths, PROJECT_ROOT / "data/derived/block_extra_features.csv",
                             PROJECT_ROOT / "data/auxiliary/sg-cpi.csv"]})
        write_json(output / "manifest.json", manifest)
        print("Generating EDA, repeated-profile and row-aligned residual diagnostics", flush=True)
        eda = generate_eda((train, test), output / "eda")
        residuals = generate_residuals(train, predictions["boosting"], output / "residuals",
                                      comparison_predictions=predictions["ridge"])
        aligned = align_holdout_predictions(train, predictions["boosting"])
        profile_comparison = []
        for strict in (False, True):
            keys = profile_keys(aligned, strict=strict)
            repeated = keys.notna().all(axis=1) & keys.duplicated(keep=False)
            part = aligned.loc[repeated]
            profile_comparison.append({"profile": "strict" if strict else "plan", "n": len(part),
                "model_rmse_sgd": regression_metrics(part.y_true, part.y_pred)["rmse"] if len(part) else None,
                "population": "Shared holdout rows in within-holdout repeated groups; descriptive comparator only"})
        pd.DataFrame(profile_comparison).to_csv(output / "profile_model_comparison.csv", index=False)
        print("Computing held-out repeated permutation importance and partial dependence", flush=True)
        interpretation = generate_interpretation(bundles["boosting"], valid, output / "interpretation",
            comparison_fitted=bundles["ridge"], sample_size=args.sample_size,
            n_repeats=args.permutation_repeats, seed=SEED)
        write_json(output / "eda_summary.json", eda)
        write_full_report(output, eda, residuals, interpretation, profile_comparison, manifest)
        manifest.update(status="complete", sampling=interpretation)
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    print(f"Saved {manifest['status']}: {output / 'report.md'}", flush=True)
    return output


if __name__ == "__main__":
    main()
