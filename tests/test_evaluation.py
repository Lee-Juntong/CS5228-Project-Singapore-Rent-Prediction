"""Checks for the shared evaluation interface. Uses small synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src import evaluation
from src.config import TIME_FIT_END, TIME_VALID_END, TIME_VALID_START
from src.evaluation import MonthlyRentIndex, fit_predict, iter_folds, regression_metrics
from src.features import FEATURE_GROUPS, feature_columns, parse_feature_set
from src.models import GlobalMedian, sklearn_tabular


def synthetic_train(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    months = pd.date_range("2021-01-01", "2025-03-01", freq="MS")
    date = pd.DatetimeIndex(rng.choice(months, rows))
    month_index = (date.year - 2021) * 12 + date.month - 1
    area = rng.uniform(40, 130, rows)
    return pd.DataFrame({
        "ROW_ID": np.arange(rows),
        "RENT_APPROVAL_DATE": pd.to_datetime(date),
        "MONTH_INDEX": month_index,
        "FLOOR_AREA_SQM": area,
        "FLAT_TYPE": pd.Categorical(rng.choice(["3-room", "4-room"], rows)),
        "MONTHLY_RENT": 1500 + 20 * area + 25 * month_index + rng.normal(0, 50, rows),
    })


def test_metrics_on_known_values():
    m = regression_metrics([100, 200], [110, 180])
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["mae"] == pytest.approx(15)
    assert m["mape"] == pytest.approx((10 + 10) / 2)
    assert m["bias"] == pytest.approx(-5)


def test_time_holdout_has_no_overlap_and_right_dates():
    train = synthetic_train()
    (fold,) = list(iter_folds(train, "time"))
    fit_dates = train.iloc[fold.fit_index]["RENT_APPROVAL_DATE"]
    valid_dates = train.iloc[fold.valid_index]["RENT_APPROVAL_DATE"]
    assert fit_dates.max() <= TIME_FIT_END
    assert valid_dates.min() >= TIME_VALID_START and valid_dates.max() <= TIME_VALID_END
    assert not set(fold.fit_index) & set(fold.valid_index)


def test_random_cv_covers_every_row_once():
    train = synthetic_train()
    valid = np.concatenate([fold.valid_index for fold in iter_folds(train, "random5")])
    assert sorted(valid) == list(range(len(train)))


def test_rent_index_uses_medians_inside_and_line_outside():
    months = pd.Series(np.repeat(np.arange(24), 3))
    rent = pd.Series(1000 + 10 * months.to_numpy(dtype=float))
    index = MonthlyRentIndex(window=12).fit(months, rent)
    assert index.transform(pd.Series([5]))[0] == pytest.approx(1050)
    assert index.transform(pd.Series([30]))[0] == pytest.approx(1300)


def test_index_mode_lets_a_constant_model_follow_the_trend():
    train = synthetic_train()
    fit, valid = train.iloc[:400], train.iloc[400:]
    raw, _, _ = fit_predict(GlobalMedian, fit, valid, ["FLOOR_AREA_SQM"], "raw")
    indexed, _, _ = fit_predict(GlobalMedian, fit, valid, ["FLOOR_AREA_SQM"], "index")
    assert np.unique(raw).size == 1
    assert np.corrcoef(indexed, valid["MONTH_INDEX"])[0, 1] > 0.7


def test_sklearn_wrapper_accepts_categories_and_nan():
    train = synthetic_train()
    train.loc[0, "FLOOR_AREA_SQM"] = np.nan
    model = sklearn_tabular(Ridge())
    model.fit(train[["FLOOR_AREA_SQM", "FLAT_TYPE"]], train["MONTHLY_RENT"])
    assert np.isfinite(model.predict(train[["FLOOR_AREA_SQM", "FLAT_TYPE"]])).all()


def test_feature_sets_are_nested_and_unique():
    assert parse_feature_set("ALL") == ["G0", "G1", "G2", "G3"]
    assert feature_columns("G0+G1")[: len(FEATURE_GROUPS["G0"])] == FEATURE_GROUPS["G0"]
    every = feature_columns("ALL")
    assert len(every) == len(set(every)) == sum(len(v) for v in FEATURE_GROUPS.values())
    with pytest.raises(KeyError):
        parse_feature_set("G9")


def test_run_experiment_writes_one_run_file(tmp_path, monkeypatch):
    for name in ("RUNS_DIR", "BREAKDOWNS_DIR", "PREDICTIONS_DIR"):
        monkeypatch.setattr(evaluation, name, tmp_path / name)
    monkeypatch.setattr(evaluation, "RESULTS_CSV", tmp_path / "results.csv")
    monkeypatch.setattr(evaluation, "feature_columns", lambda _: ["FLOOR_AREA_SQM", "FLAT_TYPE"])
    train = synthetic_train()
    row = evaluation.run_experiment("ridge", lambda: sklearn_tabular(Ridge()), data=(train, None), member="A")
    table = pd.read_csv(tmp_path / "results.csv")
    assert len(table) == 1 and table.loc[0, "rmse"] == pytest.approx(row["rmse"])
    assert len(list((tmp_path / "BREAKDOWNS_DIR").glob("*.csv"))) == 1


def test_compare_splits_scores_random_cv_on_the_holdout_rows(tmp_path, monkeypatch):
    for name in ("RUNS_DIR", "BREAKDOWNS_DIR", "PREDICTIONS_DIR"):
        monkeypatch.setattr(evaluation, name, tmp_path / name)
    monkeypatch.setattr(evaluation, "RESULTS_CSV", tmp_path / "results.csv")
    monkeypatch.setattr(evaluation, "feature_columns", lambda _: ["FLOOR_AREA_SQM", "MONTH_INDEX"])
    train = synthetic_train(1200)
    table = evaluation.compare_splits("ridge", lambda: sklearn_tabular(Ridge()), data=(train, None))
    runs = pd.read_csv(tmp_path / "results.csv")
    time_run = runs[runs["split"] == "time"].iloc[0]
    assert time_run["rmse_window"] == pytest.approx(time_run["rmse"])
    assert np.isfinite(table.loc[0, "rmse_random5_window"])
    paired = evaluation.split_comparison(runs)
    assert paired.loc[0, "optimism"] == pytest.approx(table.loc[0, "optimism"])
