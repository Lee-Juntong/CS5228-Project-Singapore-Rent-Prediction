"""Fast protocol checks, runnable with python -m unittest discover -s checks."""

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from src.baselines import GroupMedian, make_linear, make_baseline_ridge, make_knn
from src.evaluation import fit_predict, regression_metrics, time_holdout_masks, iter_folds
from src.evaluation_artifacts import evaluate_predictions
from src.models import sklearn_tabular
from sklearn.tree import DecisionTreeRegressor


class ProtocolChecks(unittest.TestCase):
    def test_metric_values_and_shape_validation(self):
        metrics = regression_metrics([100, 200], [110, 180])
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(250))
        self.assertEqual(metrics["mae"], 15)
        self.assertAlmostEqual(metrics["mape"], 10)
        self.assertEqual(metrics["bias"], -5)
        self.assertAlmostEqual(metrics["r2"], 0.9)
        for prediction in ([[110], [180]], [110], [np.nan, 180]):
            with self.assertRaises(ValueError):
                regression_metrics([100, 200], prediction)

    def test_time_boundaries_and_cv_once_per_row(self):
        frame = pd.DataFrame({"RENT_APPROVAL_DATE": pd.to_datetime(
            ["2024-09-01", "2024-10-01", "2025-03-01", "2025-04-01", "2021-01-01"])})
        fit, valid = time_holdout_masks(frame)
        np.testing.assert_array_equal(fit, [1, 0, 0, 0, 1])
        np.testing.assert_array_equal(valid, [0, 1, 1, 0, 0])
        folds = list(iter_folds(frame, "random5"))
        self.assertEqual(sorted(np.concatenate([f.valid_index for f in folds])), list(range(5)))
        for fold in folds:
            self.assertFalse(set(fold.fit_index) & set(fold.valid_index))

    def test_group_fallbacks_and_absolute_quarter(self):
        frame = pd.DataFrame({"TOWN": ["A", "A", "A", "B"],
                              "FLAT_TYPE": ["3", "3", "4", "4"], "MONTH_INDEX": [0, 3, 0, 0]})
        model = GroupMedian().fit(frame, [100, 300, 500, 700])
        query = pd.DataFrame({"TOWN": ["A", "A", "C", "B", "C"],
                              "FLAT_TYPE": ["3", "3", "4", "9", "9"], "MONTH_INDEX": [0, 12, 12, 12, 12]})
        prediction, levels = model.predict_with_levels(query)
        np.testing.assert_allclose(prediction, [100, 200, 600, 700, 400])
        self.assertEqual(levels.tolist(), ["TOWN+FLAT_TYPE+QUARTER", "TOWN+FLAT_TYPE", "FLAT_TYPE", "TOWN", "global"])

    def test_linear_area_exclusion_and_knn_inputs(self):
        frame = pd.DataFrame({"TOWN": pd.Categorical(["A", "B"] * 30), "FLAT_TYPE": pd.Categorical(["3", "4"] * 30),
                              "MONTH_INDEX": np.arange(60), "FLAT_AGE_YEARS": np.arange(60) + 1,
                              "LEASE_COMMENCE_DATE": 1990, "FLAT_TYPE_ORDINAL": 3,
                              "FLOOR_AREA_SQM": 90, "LOG_FLOOR_AREA_SQM": 4.5, "FLOOR_AREA_PER_ROOM_PROXY": 30,
                              "HDB_LATITUDE": 1.3, "HDB_LONGITUDE": 103.8})
        for factory in (make_linear, make_baseline_ridge):
            model = factory().fit(frame, np.arange(60) + 1000)
            self.assertFalse(any("AREA" in c for c in model.named_steps["columns"].columns_))
            self.assertIn("FLAT_TYPE", model.named_steps["columns"].columns_)
        self.assertEqual(make_knn().fit(frame, np.arange(60)).named_steps["model"].n_features_in_, 4)

    def test_preprocessing_is_fit_only_and_serializable(self):
        frame = pd.DataFrame({"num": [1., np.nan, 3.], "cat": pd.Categorical(["a", "a", "b"])})
        model = sklearn_tabular(DecisionTreeRegressor(random_state=5228)).fit(frame, [100, 200, 300])
        unseen = pd.DataFrame({"num": [1000., np.nan], "cat": pd.Categorical(["new", "a"])})
        model.predict(unseen)
        self.assertEqual(model.named_steps["prep"].named_transformers_["num"].named_steps["simpleimputer"].statistics_[0], 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(model.predict(unseen), joblib.load(path).predict(unseen))

    def test_id_alignment_and_bad_ids(self):
        frame = pd.DataFrame({"ROW_ID": [2, 1], "MONTHLY_RENT": [200., 100.],
                              "FLAT_TYPE": ["3", "4"], "RENT_APPROVAL_DATE": pd.to_datetime(["2024-10-01"] * 2)})
        pred = pd.DataFrame({"ROW_ID": [1, 2], "y_pred": [100., 200.]})
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(evaluate_predictions(frame, pred, Path(temp) / "good")["rmse"], 0)
            for wrong in (pred.iloc[:1], pd.concat([pred, pred.iloc[:1]]), pred.assign(ROW_ID=[1, 3])):
                with self.assertRaises(ValueError):
                    evaluate_predictions(frame, wrong, Path(temp) / "bad")

    def test_target_index_saved_with_model(self):
        from src.models import make_global_median
        fit = pd.DataFrame({"MONTH_INDEX": [0, 1, 2], "MONTHLY_RENT": [100., 110., 120.]})
        valid = pd.DataFrame({"MONTH_INDEX": [3, 4]})
        prediction, _, _, fitted = fit_predict(make_global_median, fit, valid, ["MONTH_INDEX"],
                                                target_mode="index", return_fitted=True)
        np.testing.assert_allclose(prediction, [130., 140.])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.joblib"
            joblib.dump(fitted, path)
            np.testing.assert_allclose(joblib.load(path).predict(valid), prediction)


if __name__ == "__main__":
    unittest.main()
