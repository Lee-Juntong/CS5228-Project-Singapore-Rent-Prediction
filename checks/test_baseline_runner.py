"""Runner control-flow checks using stubbed fits and temporary artifacts."""

from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import run_baselines as runner
from src.baselines import BASELINE_MODELS
from src.models import MODELS


class RunnerChecks(unittest.TestCase):
    def test_registry_preserves_baseline_and_shared_factories(self):
        for name, factory in BASELINE_MODELS.items():
            self.assertIs(type(MODELS[name]()), type(factory()))
        self.assertEqual(MODELS["ridge"]().named_steps["tabular"].named_steps["model"].alpha, 10)
        self.assertEqual(MODELS["ridge_default"]().named_steps["model"].alpha, 1)
        for name in ("lightgbm_default", "catboost_default", "lightgbm_tuned", "catboost_tuned"):
            self.assertIn(name, MODELS)

    def exercise(self, extra):
        train = pd.DataFrame({"ROW_ID": [0, 1], "MONTH_INDEX": [44, 45],
                              "RENT_APPROVAL_DATE": pd.to_datetime(["2024-09-01", "2024-10-01"]),
                              "FLAT_TYPE": ["3", "3"], "MONTHLY_RENT": [100., 200.]})
        fitted = SimpleNamespace(model=object(), columns=["MONTH_INDEX"])

        def fake_experiment(name, factory, **kwargs):
            self.assertIs(factory, MODELS[name])
            if kwargs.get("split") == "random5":
                return {"run_id": "stub", "rmse": 12., "rmse_std": 1., "rmse_window": 11.}
            folder = kwargs["artifact_dir"]
            (folder / "time").mkdir(parents=True)
            pd.DataFrame([{"rmse": 1., "mae": 1., "r2": .9}]).to_csv(folder / "time" / "train_metrics.csv", index=False)
            pd.DataFrame([{"segment_type": "month", "segment": "2024-10", "rmse": 10., "n": 1}]).to_csv(folder / "segment_metrics.csv", index=False)
            return {"model": name, "rmse": 10. if name == "ridge" else 20., "mae": 8.,
                    "mape": 4., "r2": .8, "bias": 1., "n_fit": 1, "n_valid": 1,
                    "train_seconds": 0., "predict_seconds": 0.}

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            output = Path(temp) / "run"
            args = runner.parse_args(["--output", str(output), *extra])
            stack.enter_context(patch.object(runner, "load_modeling_data", return_value=(train, train.drop(columns="MONTHLY_RENT"))))
            stack.enter_context(patch.object(runner.joblib, "load", return_value=fitted))
            experiment = stack.enter_context(patch.object(runner, "run_experiment", side_effect=fake_experiment))
            refit = stack.enter_context(patch.object(runner, "refit_final"))
            curves = stack.enter_context(patch.object(runner, "learning_curve"))
            forest = stack.enter_context(patch.object(runner, "forest_curve"))
            for name in ("plot_comparison", "plot_segments"):
                stack.enter_context(patch.object(runner, name))
            stack.enter_context(patch.object(runner.shutil, "copy2"))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            runner.run(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["completed"])
            report = (output / "evaluation_report.md").read_text()
            return experiment.call_args_list, refit.call_args_list, curves.call_args_list, forest.call_count, manifest, report

    def test_default_runs_seven_holdouts_without_optional_fits(self):
        calls, refits, curves, forest, manifest, report = self.exercise([])
        self.assertEqual(len(calls), 7)
        self.assertTrue(all(c.kwargs.get("split", "time") == "time" for c in calls))
        self.assertFalse(refits or curves or forest)
        self.assertIsNone(manifest["refit_model"])
        self.assertIn("No full-data refit", report)

    def test_optional_diagnostics_and_only_best_refit(self):
        calls, refits, curves, forest, manifest, report = self.exercise(
            ["--models", "ridge", "random_forest", "--learning-curves", "--random-cv", "--refit", "best"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1].kwargs["split"], "random5")
        self.assertEqual(calls[-1].args[0], "ridge")
        self.assertEqual([c.args[0] for c in refits], ["ridge"])
        self.assertEqual(len(curves), 2)
        self.assertTrue(all(c.kwargs["fitted_full"] is not None for c in curves))
        self.assertEqual(forest, 1)
        self.assertEqual(manifest["refit_model"], "ridge")
        self.assertIn("Only `ridge` was refitted", report)

    def test_explicit_refit_and_deduplicated_models(self):
        calls, refits, *_ = self.exercise(["--models", "ridge", "random_forest", "ridge", "--refit", "random_forest"])
        self.assertEqual(len(calls), 2)
        self.assertEqual([c.args[0] for c in refits], ["random_forest"])

    def test_reject_refit_outside_evaluated_models(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            runner.parse_args(["--models", "ridge", "--refit", "knn"])


if __name__ == "__main__":
    unittest.main()
