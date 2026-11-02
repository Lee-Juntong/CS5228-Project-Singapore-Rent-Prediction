"""Focused Member D checks with synthetic rents, never project results."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.member_d_residuals import (
    MISSING, align_holdout_predictions, diagnostic_segments, generate_residuals,
    generate_outlier_diagnostics, summarize_segments, _save_figure,
)


def fixture():
    frame = pd.DataFrame({
        "ROW_ID": [900, 101, 102, 103, 104, 901],
        "RENT_APPROVAL_DATE": pd.to_datetime(["2024-09-01", "2024-10-01", "2024-11-01",
                                             "2025-02-01", "2025-03-01", "2025-04-01"]),
        "MONTHLY_RENT": [1200., 1000., 2000., 3000., 4000., 5000.],
        "TOWN": ["A", "A", "B", None, "B", "A"],
        "FLAT_TYPE": ["3", "3", "4", "4", "", "3"],
        "FLAT_AGE_YEARS": [5., 0., 10., np.nan, 50., 15.],
        "FLOOR_AREA_SQM": [60., 70., 90., 110., 120., 80.],
        "NEAREST_OPEN_MRT_KM": [0.2, 0.3, 0.5, np.nan, 2., 1.],
    }, index=[42, 15, 7, 19, 22, 1])
    pred = pd.DataFrame({"ROW_ID": [104, 102, 101, 103], "fold": "time",
                         "y_true": [4000., 2000., 1000., 3000.], "y_pred": [3800., 2200., 1100., 2900.]})
    return frame, pred


class ResidualChecks(unittest.TestCase):
    def test_join_uses_id_not_positions_and_shared_boundaries(self):
        frame, pred = fixture()
        aligned = align_holdout_predictions(frame, pred)
        self.assertEqual(aligned.ROW_ID.tolist(), [101, 102, 103, 104])
        np.testing.assert_array_equal(aligned.residual, [100., 200., -100., -200.])
        self.assertEqual(diagnostic_segments(aligned).horizon.tolist(), ["1", "2", "5", "6"])

    def test_duplicate_missing_extra_ids_and_target_misalignment_fail(self):
        frame, pred = fixture()
        broken = [pd.concat([pred, pred.iloc[:1]]), pred.iloc[1:].copy(),
                  pred.assign(ROW_ID=[104, 102, 101, 900]), pred.assign(y_true=2000),
                  pred.assign(y_pred=np.nan), pred.assign(y_true=-1), pred.assign(fold="random1")]
        broken.append(pred.assign(ROW_ID=[104, 102, 101, None]))
        for bad in broken:
            with self.subTest(bad=bad.to_dict("list")), self.assertRaises(ValueError):
                align_holdout_predictions(frame, bad)
        for bad in (pd.concat([frame, frame.iloc[:1]]),
                    frame.assign(RENT_APPROVAL_DATE=frame.RENT_APPROVAL_DATE.mask(frame.ROW_ID.eq(101), pd.NaT))):
            with self.assertRaises(ValueError):
                align_holdout_predictions(bad, pred)

    def test_metrics_sse_counts_missing_and_bin_edges(self):
        frame, pred = fixture()
        aligned = align_holdout_predictions(frame, pred)
        segments = diagnostic_segments(aligned)
        self.assertEqual(segments.rent_band.tolist(), ["1000-<2000", "2000-<3000", "3000-<4000", "4000-<5000"])
        self.assertEqual(segments.flat_age.tolist(), ["0-<10", "10-<20", MISSING, "50+"])
        self.assertEqual(segments.town.tolist(), ["A", "B", MISSING, "B"])
        self.assertEqual(segments.flat_type.iloc[-1], MISSING)
        table = summarize_segments(aligned, segments)
        for _, part in table.groupby("segment_type"):
            self.assertEqual(part.n.sum(), 4)
            self.assertEqual(part.squared_error_sum.sum(), 100000.)
            self.assertAlmostEqual(part.squared_error_share.sum(), 1.)
            self.assertAlmostEqual(part.contribution_to_overall_mse.sum(), 25000.)
        first = table.loc[(table.segment_type == "month") & (table.segment == "2024-10")].iloc[0]
        self.assertEqual((first.rmse, first.mae, first.mape, first.bias), (100., 100., 10., 100.))

    def test_generation_handles_constant_predictions_and_matched_comparison(self):
        frame, pred = fixture()
        comparator = pred.sample(frac=1, random_state=5228).copy()
        comparator.y_pred = comparator.y_true + 300
        with tempfile.TemporaryDirectory() as directory:
            summary = generate_residuals(frame, pred, directory, comparison_predictions=comparator)
            self.assertAlmostEqual(summary["metrics"]["rmse"], np.sqrt(25000.))
            self.assertEqual(summary["metrics"]["mae"], 150.)
            self.assertEqual(summary["metrics"]["bias"], 0.)
            self.assertEqual(summary["comparison"]["rmse"], 300.)
            self.assertEqual(summary["top20_overpredictions"], 2)
            self.assertEqual(summary["top20_underpredictions"], 2)
            cases = pd.read_csv(Path(directory) / "top20_errors.csv")
            self.assertEqual(cases.ROW_ID.tolist(), [102, 104, 101, 103])
            self.assertEqual(cases.observed_metadata_issue.notna().sum(), 2)
            self.assertTrue(cases.loc[cases.observed_metadata_issue.isna(), "review_category"].str.contains(
                "cause unexplained").all())
            table = pd.read_csv(Path(directory) / "boosting_vs_ridge_segments.csv")
            self.assertIn("town_flat_type", table.segment_type.to_list())
            self.assertIn("mrt_distance", table.segment_type.to_list())
            self.assertTrue((Path(directory) / "figures/error_by_horizon.png").is_file())
            self.assertTrue((Path(directory) / "figures/validation_outliers.png").is_file())
            self.assertTrue((Path(directory) / "outlier_summary.csv").is_file())
            self.assertIn("outliers", summary)
        with tempfile.TemporaryDirectory() as directory:
            summary = generate_residuals(frame, pred.assign(y_pred=2500.), directory)
            calibration = pd.read_csv(Path(directory) / "calibration.csv")
            self.assertEqual(calibration.n.sum(), 4)
            self.assertEqual(len(calibration), 1)

    def test_perfect_predictions_have_zero_sse_shares_without_nan(self):
        frame, pred = fixture()
        aligned = align_holdout_predictions(frame, pred.assign(y_pred=pred.y_true))
        table = summarize_segments(aligned)
        self.assertTrue(table.squared_error_share.eq(0).all())
        self.assertTrue(table.rmse.eq(0).all())

    def test_outlier_strict_threshold_empty_subsets_missing_bands_and_zero_sse(self):
        frame, pred = fixture()
        # Saved predictions are intentionally shuffled: IDs define the four errors.
        changed = pred.assign(y_pred=[2000., 3001., 2000., 1999.])
        aligned = align_holdout_predictions(frame, changed)
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.member_d_residuals._save_figure", wraps=_save_figure) as save:
                summary = generate_outlier_diagnostics(aligned, summarize_segments(aligned), directory)
            self.assertEqual(save.call_args.args[0].axes[0].get_title(), "Validation predictions (n=4)")
            rows = {r["population"]: r for r in summary["populations"]}
            large = rows["Absolute error > SGD 1000"]
            self.assertEqual(large["n"], 3)  # Exactly +1000 is excluded.
            self.assertAlmostEqual(large["squared_error_share_pct"], 100 * 6004002 / 7004002)
            self.assertEqual(rows["True rent < SGD 2000"]["n"], 1)  # Exactly 2000 is excluded.
            self.assertEqual(rows["True rent >= SGD 2000 (retrospective)"]["n"], 3)
            self.assertEqual(rows["Top 4 absolute errors removed (retrospective)"]["n"], 0)
            self.assertIsNone(rows["Top 4 absolute errors removed (retrospective)"]["rmse_sgd"])
            # The sparse synthetic holdout has four observed rent bands; no fabricated bands.
            self.assertEqual(len(save.call_args.args[0].axes[1].get_yticklabels()), 4)
            self.assertTrue((Path(directory) / "figures/validation_outliers.pdf").is_file())
        perfect = align_holdout_predictions(frame, pred.assign(y_pred=pred.y_true))
        with tempfile.TemporaryDirectory() as directory:
            summary = generate_outlier_diagnostics(perfect, summarize_segments(perfect), directory)
            self.assertTrue(all(row["squared_error_share_pct"] == 0 for row in summary["populations"]))
            empty = summary["populations"][1]
            self.assertEqual(empty["n"], 0)
            self.assertIsNone(empty["rmse_sgd"])
            self.assertIsNone(empty["bias_sgd"])


if __name__ == "__main__":
    unittest.main()
