"""Synthetic integration and provenance guards; never project evidence.

Run ``python -m unittest checks.test_analysis_runner``. Temporary repositories
are created below ignored tmp/ and removed by TemporaryDirectory afterwards.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from scripts import run_analysis as runner
from src import analysis_report as report
from src.config import PROJECT_ROOT, SEED
from src.evaluation import regression_metrics
from src.features import CATEGORICAL_COLUMNS, feature_columns


def small_data():
    dates = pd.Series(np.repeat(pd.date_range("2023-01-01", "2025-03-01", freq="MS"), 4))
    n = len(dates)
    month = (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    frame = pd.DataFrame({name: np.full(n, .25) for name in feature_columns("ALL")})
    frame["ROW_ID"] = np.arange(1000, 1000 + n)
    frame["RENT_APPROVAL_DATE"] = dates
    frame["MONTH_INDEX"] = month
    frame["FLOOR_AREA_SQM"] = np.resize([65., 65., 90., 90.], n)
    frame["FLAT_TYPE_ORDINAL"] = np.resize([3, 3, 4, 4], n)
    frame["FLOOR_AREA_PER_ROOM_PROXY"] = frame.FLOOR_AREA_SQM / frame.FLAT_TYPE_ORDINAL
    frame["LOG_FLOOR_AREA_SQM"] = np.log(frame.FLOOR_AREA_SQM)
    frame["LEASE_COMMENCE_DATE"] = np.resize([1990, 1990, 2000, 2000], n)
    frame["FLAT_AGE_YEARS"] = dates.dt.year - frame.LEASE_COMMENCE_DATE
    frame["INVALID_LEASE_DATE_FLAG"] = 0
    frame["MONTH_SIN"] = np.sin(2 * np.pi * (dates.dt.month - 1) / 12)
    frame["MONTH_COS"] = np.cos(2 * np.pi * (dates.dt.month - 1) / 12)
    frame["TOWN"] = np.resize(["alpha", "alpha", "beta", "beta"], n)
    frame["FLAT_TYPE"] = np.resize(["3-room", "3-room", "4-room", "4-room"], n)
    frame["FLAT_MODEL"] = "new generation"
    frame["HDB_SUBZONE"] = frame.TOWN
    frame["BLOCK"] = "1"
    frame["STREET"] = "first street"
    frame["ADDRESS_KEY"] = frame.TOWN + "|1|first street"
    frame["MONTHLY_RENT"] = 1600 + 20 * month + np.resize([0, 200, 400, 600], n)
    for name in CATEGORICAL_COLUMNS:
        frame[name] = frame[name].astype("category")
    test = frame.iloc[:8].drop(columns="MONTHLY_RENT").copy()
    test["RENT_APPROVAL_DATE"] = pd.to_datetime("2025-03-01")
    return frame, test


def prepare_test_repository(root, train, test):
    """Copy the small inputs needed for source hashing into an isolated repo."""
    paths = ["scripts/run_analysis.py", "scripts/prepare_analysis_data.py", "requirements-analysis.txt",
             "configs/lightgbm_tuned.json",
             "artifacts/data_audit/audit_summary.json",
             "data/derived/block_extra_features.csv", "data/auxiliary/sg-cpi.csv",
             "results/runs/" + runner.REFERENCE_RUN + ".csv"]
    paths += [p.relative_to(PROJECT_ROOT).as_posix()
              for p in (PROJECT_ROOT / "src").glob("analysis_*.py")]
    paths += ["src/" + name for name in ("evaluation.py", "features.py", "models.py", "config.py",
                                        "external_features.py", "data_preprocessing.py")]
    for name in paths:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / name, destination)
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    train.to_parquet(processed / "train_features.parquet", index=False)
    test.to_parquet(processed / "test_features.parquet", index=False)
    return processed


class RunnerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)

    def test_reference_checks_protocol_metrics_and_row_level_predictions(self):
        train, _ = small_data()
        valid = train[train.RENT_APPROVAL_DATE >= pd.Timestamp("2024-10-01")]
        predicted = pd.DataFrame({"ROW_ID": valid.ROW_ID.to_numpy(), "fold": "time",
                                  "y_true": valid.MONTHLY_RENT.to_numpy(),
                                  "y_pred": valid.MONTHLY_RENT.to_numpy() + 25})
        reference = {"model": "lightgbm_tuned", "feature_set": "G0+G1", "split": "time",
                     "target_mode": "index", "seed": SEED, "n_features": 39,
                     "n_fit": len(train) - len(valid), "n_valid": len(valid),
                     **regression_metrics(predicted.y_true, predicted.y_pred)}
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as folder:
            with patch.object(runner, "PREDICTIONS_DIR", Path(folder)):
                check = runner.verify_reference(train, predicted, reference)
                self.assertEqual(check["level"], "aggregate metrics only")
                with self.assertRaisesRegex(ValueError, "seed"):
                    runner.verify_reference(train, predicted, {**reference, "seed": 4})
                with self.assertRaisesRegex(ValueError, "rmse"):
                    runner.verify_reference(train, predicted, {**reference, "rmse": 26.})
                saved_path = Path(folder) / (runner.REFERENCE_RUN + ".parquet")
                predicted.sample(frac=1, random_state=12).to_parquet(saved_path, index=False)
                check = runner.verify_reference(train, predicted, reference)
                self.assertEqual(check["level"], "ROW_ID-aligned prediction agreement")
                changed = predicted.copy()
                changed["y_pred"] += np.resize([10., -10.], len(changed))
                changed.to_parquet(saved_path, index=False)
                with self.assertRaises(AssertionError):
                    runner.verify_reference(train, predicted, reference)

    def test_missing_data_rejected_without_creating_partial_output(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as folder:
            root = Path(folder)
            destination = root / "results/analysis/missing"
            with patch.object(runner, "PROJECT_ROOT", root), patch.object(
                    runner, "PROCESSED_DIR", root / "data/processed"), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.main(["--output", str(destination)])
            self.assertFalse(destination.exists())

    def test_loading_failure_marks_manifest_before_fitting(self):
        train, test = small_data()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as folder:
            root = Path(folder)
            processed = prepare_test_repository(root, train, test)
            destination = root / "results/analysis/failed"
            with patch.object(runner, "PROJECT_ROOT", root), \
                    patch.object(runner, "PROCESSED_DIR", processed), \
                    patch.object(runner, "__file__", str(root / "scripts/run_analysis.py")), \
                    patch.object(runner, "load_modeling_data", side_effect=ValueError("bad source")) as load, \
                    patch.object(runner, "fit_predict") as fit, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "bad source"):
                    runner.main(["--output", str(destination)])
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("bad source", manifest["error"])
            fit.assert_not_called()
            load.assert_called_once_with()

    def test_missing_configuration_and_stale_preparation_fail_before_output(self):
        train, test = small_data()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as folder:
            root = Path(folder)
            processed = prepare_test_repository(root, train, test)
            destination = root / "results/analysis/rejected"
            configuration = root / "configs/lightgbm_tuned.json"
            content = configuration.read_bytes()
            configuration.unlink()
            with ExitStack() as stack:
                stack.enter_context(patch.object(runner, "PROJECT_ROOT", root))
                stack.enter_context(patch.object(runner, "PROCESSED_DIR", processed))
                stack.enter_context(patch.object(runner, "RUNS_DIR", root / "results/runs"))
                stack.enter_context(redirect_stderr(io.StringIO()))
                load = stack.enter_context(patch.object(runner, "load_modeling_data"))
                with self.assertRaises(SystemExit):
                    runner.main(["--output", str(destination)])
                self.assertFalse(destination.exists())
                configuration.write_bytes(content)
                preparation = processed / "analysis_preparation/input_preparation.json"
                preparation.parent.mkdir()
                preparation.write_text(json.dumps({"processed_sha256": {
                    path.relative_to(root).as_posix(): report.file_hash(path)
                    for path in processed.glob("*.parquet")}}), encoding="utf-8")
                changed = train.copy()
                changed.loc[0, "MONTHLY_RENT"] += 100
                changed.to_parquet(processed / "train_features.parquet", index=False)
                with self.assertRaises(SystemExit):
                    runner.main(["--output", str(destination)])
                self.assertFalse(destination.exists())
                load.assert_not_called()

    def test_full_synthetic_pipeline_serializes_and_links_every_asset(self):
        train, test = small_data()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as folder:
            root = Path(folder)
            processed = prepare_test_repository(root, train, test)
            destination = root / "results/analysis/synthetic"
            preparation_dir = processed / "analysis_preparation"
            preparation_dir.mkdir()
            shutil.copyfile(root / "artifacts/data_audit/audit_summary.json",
                            preparation_dir / "audit_summary.json")
            report.write_json(preparation_dir / "input_preparation.json", {"processed_sha256": {
                path.relative_to(root).as_posix(): report.file_hash(path)
                for path in processed.glob("*.parquet")}})
            with ExitStack() as stack:
                stack.enter_context(patch.object(runner, "PROJECT_ROOT", root))
                stack.enter_context(patch.object(report, "PROJECT_ROOT", root))
                stack.enter_context(patch.object(runner, "PROCESSED_DIR", processed))
                stack.enter_context(patch.object(runner, "RUNS_DIR", root / "results/runs"))
                stack.enter_context(patch.object(runner, "__file__", str(root / "scripts/run_analysis.py")))
                stack.enter_context(patch.object(runner, "load_modeling_data", return_value=(train, test)))
                stack.enter_context(patch.object(runner, "make_lightgbm_tuned",
                                                 return_value=DummyRegressor(strategy="median")))
                # This checks plumbing using artificial data, never agreement
                # with the numerical results of a competition experiment.
                stack.enter_context(patch.object(runner, "verify_reference", return_value={
                    "level": "synthetic smoke only; archived numerical agreement not asserted"}))
                stack.enter_context(redirect_stdout(io.StringIO()))
                actual = runner.main(["--output", str(destination), "--sample-size", "12",
                                      "--permutation-repeats", "2"])
            self.assertEqual(actual, destination)
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["source_run"]["notes"], "")
            self.assertEqual(manifest["n_valid"], 24)
            self.assertEqual(manifest["sampling"]["sample_n"], 12)
            self.assertIn("data/derived/block_extra_features.csv", manifest["data_sha256"])
            self.assertIn("src/data_preprocessing.py", manifest["code_sha256"])
            self.assertIn("local_preprocessing_audit_sha256", manifest)
            self.assertEqual(report.file_hash(destination / "preparation/input_preparation.json"),
                             manifest["input_preparation_sha256"])
            self.assertEqual(report.file_hash(destination / "preparation/audit_summary.json"),
                             manifest["local_preprocessing_audit_sha256"])
            self.assertEqual(manifest["command_args"][-1], "2")
            for path in destination.rglob("*.json"):
                # Strict parsing forbids NaN and Infinity in serialized results.
                json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda v: self.fail(f"Invalid JSON {v} in {path}"))
            markdown = (destination / "report.md").read_text(encoding="utf-8")
            # Check ordinary local links as well as images; optional manually
            # reviewed notes must not leave broken links in a fresh report.
            links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", markdown)
            self.assertGreaterEqual(len(links), 9)
            for link in links:
                local_path = unquote(link.split("#", 1)[0].strip("<>"))
                if local_path and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", local_path):
                    self.assertTrue((destination / local_path).is_file(), link)
            self.assertTrue((destination / "sections.tex").is_file())
            for filename in ("repeated_profile_summary.csv", "correlation_vs_importance.csv",
                             "residuals/top20_errors.csv", "profile_model_comparison.csv",
                             "residuals/outlier_summary.csv",
                             "residuals/figures/validation_outliers.png",
                             "residuals/figures/validation_outliers.pdf"):
                self.assertGreater((destination / filename).stat().st_size, 0)
            comparison = pd.read_csv(destination / "profile_model_comparison.csv")
            self.assertTrue(comparison.n.eq(24).all())


if __name__ == "__main__":
    unittest.main()
