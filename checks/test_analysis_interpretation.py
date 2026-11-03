"""Focused interpretation checks; run ``python -m unittest checks.test_analysis_interpretation``."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import FittedRentModel, fit_predict
from src.analysis_interpretation import (
    generate_interpretation, heldout_sample, partial_dependence_table,
)
from src.models import GlobalMedian


def example_frame(dates):
    dates = pd.Series(pd.to_datetime(dates))
    n = len(dates)
    area = np.linspace(45, 120, n)
    ordinal = np.resize([3, 4, 5], n)
    month = (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    age = np.resize([15, 30, 40], n)
    return pd.DataFrame({
        "ROW_ID": np.arange(n), "RENT_APPROVAL_DATE": dates,
        "MONTHLY_RENT": 2000 + 20 * month,
        "MONTH_INDEX": month,
        "MONTH_SIN": np.sin(2 * np.pi * (dates.dt.month - 1) / 12),
        "MONTH_COS": np.cos(2 * np.pi * (dates.dt.month - 1) / 12),
        "FLOOR_AREA_SQM": area, "LOG_FLOOR_AREA_SQM": np.log(area),
        "FLAT_TYPE_ORDINAL": ordinal, "FLOOR_AREA_PER_ROOM_PROXY": area / ordinal,
        "FLAT_AGE_YEARS": age, "LEASE_COMMENCE_DATE": dates.dt.year - age,
        "INVALID_LEASE_DATE_FLAG": np.zeros(n),
        "NEAREST_OPEN_MRT_KM": np.linspace(.1, 2, n),
    })


def columns(frame):
    return list(frame.drop(columns=["ROW_ID", "RENT_APPROVAL_DATE", "MONTHLY_RENT"]))


class AreaDerivatives:
    def predict(self, X):
        return 1000 + X.LOG_FLOOR_AREA_SQM.to_numpy() * 50 + X.FLOOR_AREA_PER_ROOM_PROXY.to_numpy() * 10


class InterpretationChecks(unittest.TestCase):
    def setUp(self):
        self.fit = example_frame(pd.date_range("2023-10-01", "2024-09-01", freq="MS"))
        self.valid = example_frame(np.repeat(pd.date_range("2024-10-01", "2025-03-01", freq="MS"), 5))

    def test_sample_stable_under_ordering_and_targets(self):
        first = heldout_sample(self.valid, 15, 5228)
        shuffled = self.valid.sample(frac=1, random_state=12)
        shuffled["MONTHLY_RENT"] += 123
        second = heldout_sample(shuffled, 15, 5228)
        self.assertEqual(first.ROW_ID.tolist(), second.ROW_ID.tolist())
        self.assertEqual(heldout_sample(self.valid, 100, 5228).shape, self.valid.shape)

    def test_sample_rejects_identity_and_calendar_problems(self):
        duplicate = self.valid.copy()
        duplicate.loc[1, "ROW_ID"] = duplicate.loc[0, "ROW_ID"]
        with self.assertRaises(ValueError):
            heldout_sample(duplicate, 15, 5228)
        with self.assertRaises(ValueError):
            heldout_sample(self.fit, 15, 5228)
        with self.assertRaises(ValueError):
            heldout_sample(self.valid, 0, 5228)
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            heldout_sample(self.valid.iloc[:1], 15, 5228)

    def test_retained_index_is_training_only_and_restores_sgd(self):
        prediction, _, _, fitted = fit_predict(GlobalMedian, self.fit, self.valid,
                                                columns(self.fit), "index", return_fitted=True)
        changed = self.valid.copy()
        changed["MONTHLY_RENT"] *= 20
        second, _, _, second_fitted = fit_predict(GlobalMedian, self.fit, changed,
                                                  columns(self.fit), "index", return_fitted=True)
        np.testing.assert_allclose(prediction, self.valid.MONTHLY_RENT, atol=1e-9)
        np.testing.assert_array_equal(prediction, second)
        self.assertEqual(fitted.rent_index.last_month_, int(self.fit.MONTH_INDEX.max()))
        pd.testing.assert_series_equal(fitted.rent_index.medians_, second_fitted.rent_index.medians_)

    def test_coherent_area_curve_exposes_engineered_dependence(self):
        fitted = FittedRentModel(AreaDerivatives(), columns(self.valid), "raw")
        table = partial_dependence_table(fitted, heldout_sample(self.valid, 15, 5228))
        area = table[table.feature == "FLOOR_AREA_SQM"]
        standard = area[area.intervention == "standard"].mean_prediction_sgd
        coherent = area[area.intervention == "coherent_derivatives"].mean_prediction_sgd
        self.assertAlmostEqual(float(standard.max() - standard.min()), 0)
        self.assertGreater(float(coherent.max() - coherent.min()), 100)

    def test_repeatability_full_sgd_importance_and_artifacts(self):
        _, _, _, fitted = fit_predict(GlobalMedian, self.fit, self.valid, columns(self.fit),
                                      "index", return_fitted=True)
        with tempfile.TemporaryDirectory() as folder:
            first = generate_interpretation(fitted, self.valid, Path(folder) / "a", sample_size=20, n_repeats=3)
            second = generate_interpretation(fitted, self.valid.sample(frac=1, random_state=4),
                                             Path(folder) / "b", sample_size=20, n_repeats=3)
            self.assertEqual(first, second)
            importance = pd.read_csv(Path(folder) / "a" / "permutation_importance.csv")
            self.assertEqual(importance.iloc[0].feature, "MONTH_INDEX")
            self.assertGreater(importance.iloc[0].importance_mean_sgd, 0)
            self.assertEqual(first["sample_n"], 20)
            for filename in ("sample_rows.csv", "permutation_repeats.csv", "partial_dependence.csv",
                             "partial_dependence.png", "partial_dependence.pdf", "summary.json"):
                self.assertGreater((Path(folder) / "a" / filename).stat().st_size, 0)

    def test_comparison_requires_matching_information(self):
        fitted = FittedRentModel(AreaDerivatives(), columns(self.valid), "raw")
        other = FittedRentModel(AreaDerivatives(), columns(self.valid)[:-1], "raw")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                generate_interpretation(fitted, self.valid, folder, comparison_fitted=other)
            index_without_transform = FittedRentModel(AreaDerivatives(), columns(self.valid), "index")
            with self.assertRaises(ValueError):
                generate_interpretation(index_without_transform, self.valid, folder)


if __name__ == "__main__":
    unittest.main()
