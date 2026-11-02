"""Focused synthetic checks for Member D's descriptive calculations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.member_d_eda import generate_eda, profile_keys, repeated_profile_dispersion


def fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "ROW_ID": [1, 2, 3, 4, 5, 6, 7],
        "TOWN": ["Alpha"] * 5 + ["Beta", "Alpha"],
        "BLOCK": ["1"] * 7,
        "STREET": ["One St"] * 7,
        "ADDRESS_KEY": [" alpha | 1 | one  st "] * 2 + ["alpha|1|two st"] * 3 + ["beta|1|one st", None],
        "FLAT_TYPE": ["4 room", "4-room"] + ["4-room"] * 5,
        "FLOOR_AREA_SQM": [90.0] * 7,
        "RENT_APPROVAL_DATE": pd.to_datetime(["2024-10-01"] * 7),
        "FLAT_MODEL": ["new", "old", "new", "new", "new", "new", "new"],
        "LEASE_COMMENCE_DATE": [1990] * 7,
        "MONTH_INDEX": [45] * 7,
        "MONTHLY_RENT": [1000., 1400., 2000., 2600., 3200., 1800., 1700.],
    })


class RepeatedProfiles(unittest.TestCase):
    def test_spreads_have_distinct_denominators_and_no_self_inclusion_in_loo(self):
        summary, groups, sizes = repeated_profile_dispersion(fixture())
        self.assertEqual(summary["eligible_rows"], 6)
        self.assertEqual(summary["excluded_missing_key_rows"], 1)
        self.assertEqual(summary["all_groups"], 3)
        self.assertEqual(summary["repeated_groups"], 2)
        self.assertEqual(summary["repeated_rows"], 5)
        self.assertAlmostEqual(summary["coverage_fraction_all_rows"], 5 / 7)
        self.assertEqual(sizes["row_count"].sum(), 6)
        self.assertEqual(set(groups["n"]), {2, 3})
        # Independent hand calculation: SSE_A=80,000; SSE_B=720,000.
        expected_sse = 800000.
        self.assertAlmostEqual(groups["within_group_sse"].sum(), expected_sse)
        self.assertAlmostEqual(summary["unweighted_mean_group_sample_sd_sgd"], (np.sqrt(80000.) + 600.) / 2)
        self.assertAlmostEqual(summary["row_weighted_rms_group_sample_sd_sgd"], np.sqrt((2 * 80000 + 3 * 360000) / 5))
        self.assertAlmostEqual(summary["pooled_within_group_sample_sd_sgd"], np.sqrt(expected_sse / 3))
        self.assertAlmostEqual(summary["self_inclusive_group_mean_rmse_sgd"], 400.)
        # Explicit row LOO predictions: [1400,1000,2900,2600,2300].
        y = np.array([1000, 1400, 2000, 2600, 3200])
        loo = np.array([1400, 1000, 2900, 2600, 2300])
        self.assertAlmostEqual(summary["leave_one_out_group_mean_rmse_sgd"], np.sqrt(np.mean((y - loo) ** 2)))

    def test_address_identity_and_strict_sensitivity(self):
        frame = fixture()
        base, _, _ = repeated_profile_dispersion(frame)
        strict, _, _ = repeated_profile_dispersion(frame, strict=True)
        self.assertEqual(base["repeated_rows"], 5)
        self.assertEqual(strict["repeated_rows"], 3)
        # Block number 1 appears in both towns and two streets; it must not merge.
        keys = profile_keys(frame.drop(columns="ADDRESS_KEY"))
        self.assertNotEqual(keys.loc[0, "ADDRESS_KEY"], keys.loc[5, "ADDRESS_KEY"])
        self.assertEqual(keys.loc[0, "FLAT_TYPE"], keys.loc[1, "FLAT_TYPE"])

    def test_missing_malformed_dates_area_and_singletons(self):
        frame = fixture().iloc[:2].copy()
        frame.loc[frame.index[0], "ADDRESS_KEY"] = "alpha||one st"
        frame.loc[frame.index[1], "FLOOR_AREA_SQM"] = np.nan
        summary, groups, sizes = repeated_profile_dispersion(frame)
        self.assertEqual(summary["eligible_rows"], 0)
        self.assertEqual(summary["repeated_groups"], 0)
        self.assertTrue(groups.empty)
        self.assertTrue(sizes.empty)
        self.assertIsNone(summary["leave_one_out_group_mean_rmse_sgd"])
        frame = fixture().iloc[[5]].copy()
        summary, _, _ = repeated_profile_dispersion(frame)
        self.assertEqual(summary["singleton_groups"], 1)
        self.assertIsNone(summary["unweighted_mean_group_sample_sd_sgd"])
        frame.loc[frame.index[0], "RENT_APPROVAL_DATE"] = pd.NaT
        self.assertEqual(repeated_profile_dispersion(frame)[0]["eligible_rows"], 0)

    def test_duplicate_ids_invalid_targets_and_exact_area(self):
        frame = fixture().iloc[:2].copy()
        frame.loc[frame.index[1], "ROW_ID"] = frame.iloc[0]["ROW_ID"]
        with self.assertRaisesRegex(ValueError, "ROW_ID"):
            repeated_profile_dispersion(frame)
        frame = fixture().iloc[:2].copy()
        frame.loc[frame.index[1], "MONTHLY_RENT"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            repeated_profile_dispersion(frame)
        frame = fixture().iloc[:2].copy()
        frame.loc[frame.index[1], "FLOOR_AREA_SQM"] += .0001
        self.assertEqual(repeated_profile_dispersion(frame)[0]["repeated_groups"], 0)

    def test_row_order_does_not_change_metrics(self):
        original = repeated_profile_dispersion(fixture())[0]
        shuffled = repeated_profile_dispersion(fixture().sample(frac=1, random_state=5228))[0]
        self.assertEqual(original, shuffled)

    def test_eda_tables_and_figures_use_complete_population(self):
        frame = fixture().iloc[:6].copy()
        frame["MONTH_INDEX"] = [0, 1, 2, 3, 4, 45]
        frame["RENT_APPROVAL_DATE"] = pd.to_datetime("2021-01-01") + frame["MONTH_INDEX"].map(lambda x: pd.DateOffset(months=x))
        test = frame.iloc[:2].copy()
        test["RENT_APPROVAL_DATE"] = pd.to_datetime(["2024-10-01", "2025-03-01"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = generate_eda((frame, test), root)
            self.assertEqual(summary["training_rows"], 6)
            self.assertEqual(summary["overlapping_train_test_months"], ["2024-10"])
            corr = summary["time_association"][0]
            self.assertAlmostEqual(corr["spearman_rho"], frame["MONTH_INDEX"].corr(frame["MONTHLY_RENT"], method="spearman"))
            for name in ["rent_by_town", "rent_by_flat_type", "rent_by_month", "rent_histogram"]:
                table = pd.read_csv(root / "tables" / f"{name}.csv")
                self.assertEqual(table["n"].sum(), 6)
            for stem in ["eda_distribution_trend", "eda_flat_type", "eda_town", "eda_town_month"]:
                self.assertGreater((root / "figures" / f"{stem}.png").stat().st_size, 1000)
                self.assertGreater((root / "figures" / f"{stem}.pdf").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
