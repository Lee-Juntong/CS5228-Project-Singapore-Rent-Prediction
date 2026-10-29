"""Audit actual saved artifacts without training models again."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import BASELINE_MODELS
from src.evaluation import regression_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("completed"), "Experiment did not finish"
    table = pd.read_csv(root / "results_table.csv")
    assert set(table.model) == set(BASELINE_MODELS)
    assert not table.model.duplicated().any()
    truth = pd.read_csv(root / "holdout_truth.csv")
    metrics = ["rmse", "mae", "mape", "r2", "bias"]
    tested = []
    for row in table.to_dict("records"):
        name = row["model"]
        holdout = root / "holdout" / name
        prediction = pd.read_csv(holdout / "time" / "validation_predictions.csv")
        assert not prediction.ROW_ID.duplicated().any()
        np.testing.assert_array_equal(prediction.ROW_ID, truth.ROW_ID)
        np.testing.assert_array_equal(prediction.y_true, truth.MONTHLY_RENT)
        scores = regression_metrics(prediction.y_true, prediction.y_pred)
        for metric in metrics:
            np.testing.assert_allclose(scores[metric], row[metric], rtol=1e-10, atol=1e-8)
        ids = pd.read_csv(holdout / "time" / "fit_ids.csv").ROW_ID
        assert len(ids) == manifest["n_fit"] and not set(ids) & set(truth.ROW_ID)
        assert (holdout / "time" / "model.joblib").stat().st_size > 0
        curve = pd.read_csv(root / "curves" / f"{name}.csv")
        assert len(curve) == 5 and curve.n_fit.is_monotonic_increasing
        assert int(curve.iloc[-1].n_fit) == manifest["n_fit"]
        np.testing.assert_allclose(curve.iloc[-1].valid_rmse, row["rmse"], rtol=1e-10)
        assert (root / "curves" / f"{name}.png").exists()
        assert (root / "curves" / f"{name}.svg").exists()
        segments = pd.read_csv(holdout / "segment_metrics.csv")
        for segment_type in ("month", "flat_type"):
            part = segments[segments.segment_type == segment_type]
            assert part.n.sum() == len(truth)
            pooled = np.sqrt(np.average(part.rmse ** 2, weights=part.n))
            np.testing.assert_allclose(pooled, row["rmse"], rtol=1e-10)
        final = root / "full_fit" / name
        test_prediction = pd.read_csv(final / "test_predictions.csv")
        assert len(test_prediction) == manifest["n_test"]
        np.testing.assert_array_equal(test_prediction.ROW_ID, np.arange(manifest["n_test"]))
        assert np.isfinite(test_prediction.y_pred).all()
        assert (final / "model.joblib").stat().st_size > 0
        assert pd.read_csv(final / "metadata.csv").iloc[0].serialization_verified
        tested.append({"model": name, "n_holdout": len(prediction), "n_test": len(test_prediction),
                       "metrics_recomputed": True, "curve_endpoint_matches": True,
                       "test_min": float(test_prediction.y_pred.min()), "test_max": float(test_prediction.y_pred.max())})
    forest = pd.read_csv(root / "curves" / "random_forest_trees.csv")
    assert forest.n_estimators.tolist() == [10, 25, 50, 100, 200, 300]
    np.testing.assert_allclose(forest.iloc[-1].valid_rmse, table.set_index("model").loc["random_forest", "rmse"], rtol=1e-10)
    cv = pd.read_parquet(root / "random5_predictions.parquet")
    assert len(cv) == manifest["n_train"] and not cv.ROW_ID.duplicated().any()
    assert cv.fold.nunique() == 5
    matched = truth[["ROW_ID", "MONTHLY_RENT"]].merge(cv, on="ROW_ID", validate="one_to_one")
    assert len(matched) == len(truth)
    np.testing.assert_array_equal(matched.MONTHLY_RENT, matched.y_true)
    comparison = pd.read_csv(root / "split_comparison.csv").iloc[0]
    np.testing.assert_allclose(regression_metrics(matched.y_true, matched.y_pred)["rmse"],
                               comparison.random5_rmse_same_window, rtol=1e-10)
    for metric in metrics:
        assert (root / "figures" / f"{metric}.png").exists()
        assert (root / "figures" / f"{metric}.svg").exists()
    for path in ("gaps.txt", "evaluation_report.md", "results_table.tex", "hyperparameters.json"):
        assert (root / path).stat().st_size > 0
    (root / "verification.json").write_text(json.dumps({"passed": True, "models": tested,
        "random_cv_rows": len(cv), "random_cv_folds": 5, "holdout_models": 7, "full_fit_models": 7}, indent=2), encoding="utf-8")
    print("PASS: 7 result rows, 14 model files, all validation/test IDs, recomputed metrics, learning curves and random CV.")


if __name__ == "__main__":
    main()
