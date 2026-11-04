"""Five model repeat-rent blend. See docs/repeat_rent_blend.md.

    python -m scripts.lag_blend_submission
        Fits on all training rows and writes submissions/lag_blend5_G0G1_index.csv.

    python -m scripts.lag_blend_submission --validate
        Scores the blends on the time holdout and on a 16 month window, measures
        the limit on any model and the correlation of rents by gap. Writes
        results/member_a/repeat_rent/*.csv and report/figures/lag_correlation.pdf.
"""

from __future__ import annotations

import argparse
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from src.config import PROJECT_ROOT, TARGET
from src.evaluation import MonthlyRentIndex, regression_metrics
from src.features import feature_columns, load_modeling_data

warnings.filterwarnings("ignore")

OUT = PROJECT_ROOT / "results" / "member_a" / "repeat_rent"
SUBMISSION = PROJECT_ROOT / "submissions" / "lag_blend5_G0G1_index.csv"
FIGURE = PROJECT_ROOT / "report" / "figures" / "lag_correlation.pdf"

COLUMNS = feature_columns("G0+G1")
CONFIGS = PROJECT_ROOT / "configs"
LGBM = dict(json.load(open(CONFIGS / "lightgbm_tuned.json"))["params"], random_state=5228, verbose=-1, n_jobs=-1)
LGBM_LAG = {**LGBM, "num_leaves": 31, "min_child_samples": 200, "feature_fraction": 0.5, "n_estimators": 800, "random_state": 4}
XGB = {**json.load(open(CONFIGS / "xgboost_tuned.json"))["params"], "tree_method": "hist", "random_state": 6, "n_jobs": -1}
CATBOOST = {**json.load(open(CONFIGS / "catboost_tuned.json"))["params"], "iterations": 800, "random_seed": 7, "verbose": 0}
FOREST = dict(n_estimators=200, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=8)

# Gaps in months. OFF is a comparison set away from the renewal gaps.
GAPS = {
    "L12": (11, 12, 13),
    "L24": (23, 24, 25),
    "L36": (35, 36, 37),
    "OFF": (6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30),
}

# Last fit month index of each window. The data ends at month 50 (2025-03).
WINDOWS = {"holdout": 44, "16_month": 34}


def add_cell(frame: pd.DataFrame, path: str) -> None:
    """Add BT, the block, street and flat type key, from the raw csv."""
    raw = pd.read_csv(path).iloc[frame.ROW_ID.values].reset_index(drop=True)
    for column in ["BLOCK", "STREET", "FLAT_TYPE"]:
        raw[column] = raw[column].astype(str).str.lower().str.strip()
    raw["FLAT_TYPE"] = raw.FLAT_TYPE.str.replace("-", " ")
    frame["BT"] = (raw.BLOCK + "|" + raw.STREET + "|" + raw.FLAT_TYPE).values


def lag_features(history: pd.DataFrame, target: pd.DataFrame, cut: int) -> pd.DataFrame:
    """Rent ratios of earlier rows in the same cell, at each gap.

    `history` holds the known rows with their ratio `a`. Only rows up to month
    `cut` are used, and every gap is positive, so a row never sees itself.
    """
    out = pd.DataFrame(index=target.index)
    base = history[["BT", "MONTH_INDEX", "a"]]
    keys = target[["BT", "MONTH_INDEX"]].reset_index()
    for name, gaps in GAPS.items():
        parts = []
        for gap in gaps:
            shifted = base.assign(MONTH_INDEX=base.MONTH_INDEX + gap)
            parts.append(shifted[shifted.MONTH_INDEX - gap <= cut])
        stats = pd.concat(parts).groupby(["BT", "MONTH_INDEX"]).a.agg(["mean", "count", "min", "max"]).reset_index()
        joined = keys.merge(stats, on=["BT", "MONTH_INDEX"], how="left").set_index("index")
        out[name + "_mean"] = joined["mean"]
        out[name + "_n"] = joined["count"]
        out[name + "_min"] = joined["min"]
        out[name + "_max"] = joined["max"]
    return out.loc[target.index]


def as_numeric(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in categorical:
        frame[column] = frame[column].cat.codes
    return frame


def fit_five(fit: pd.DataFrame, valid: pd.DataFrame, cut: int, target_mode: str = "index", only_lightgbm: bool = False) -> dict:
    """Fit the blend members on `fit` and return their rent predictions for `valid`."""
    index = MonthlyRentIndex().fit(fit.MONTH_INDEX, fit[TARGET])
    if target_mode == "index":
        fit = fit.assign(a=fit[TARGET].values / index.transform(fit.MONTH_INDEX))
        scale = index.transform(valid.MONTH_INDEX)
    else:
        fit = fit.assign(a=fit[TARGET].values.astype(float))
        scale = 1.0
    fit_lags, valid_lags = lag_features(fit, fit, cut), lag_features(fit, valid, cut)
    X, X_valid = pd.concat([fit[COLUMNS], fit_lags], axis=1), pd.concat([valid[COLUMNS], valid_lags], axis=1)
    predictions = {
        "base": lgb.LGBMRegressor(**LGBM).fit(fit[COLUMNS], fit.a).predict(valid[COLUMNS]) * scale,
        "lag_reg": lgb.LGBMRegressor(**LGBM_LAG).fit(X, fit.a).predict(X_valid) * scale,
    }
    if only_lightgbm:
        return predictions
    categorical = [c for c in COLUMNS if str(fit[c].dtype) == "category"]
    X_num, X_valid_num = as_numeric(X, categorical), as_numeric(X_valid, categorical)
    predictions["xgb"] = xgb.XGBRegressor(**XGB).fit(X_num, fit.a).predict(X_valid_num) * scale
    predictions["cat"] = CatBoostRegressor(**CATBOOST).fit(X_num, fit.a).predict(X_valid_num) * scale
    predictions["rf"] = (
        RandomForestRegressor(**FOREST).fit(X_num.fillna(-1), fit.a).predict(X_valid_num.fillna(-1)) * scale
    )
    return predictions


def submission(train: pd.DataFrame, test: pd.DataFrame) -> None:
    add_cell(test, "data/test.csv")
    predictions = fit_five(train, test, int(train.MONTH_INDEX.max()))
    blend = np.mean(list(predictions.values()), axis=0)
    out = pd.DataFrame({"Id": test.ROW_ID.values, "Predicted": blend}).sort_values("Id")
    SUBMISSION.parent.mkdir(exist_ok=True)
    out.to_csv(SUBMISSION, index=False)
    print(out.shape, out.Id.is_unique, out.Predicted.describe().round(1).to_dict())
    print({name: round(float(values.mean()), 1) for name, values in predictions.items()})


def limit_rows(rows: pd.DataFrame, label: str) -> dict:
    """RMSE of the best possible prediction per group of identical G0+G1 inputs."""
    y = rows[TARGET].values
    groups = rows[COLUMNS].astype(str).agg("|".join, axis=1)
    same = rows.groupby(groups.values)[TARGET]
    mean, size = same.transform("mean").values, same.transform("size").values
    shared = size > 1
    cell = rows.groupby(["BT", "MONTH_INDEX"])[TARGET]
    return {
        "rows": label,
        "n": len(rows),
        "limit_identical_inputs": float(np.sqrt(np.mean((y - mean) ** 2))),
        "limit_block_type_month": float(np.sqrt(np.mean((y - cell.transform("mean").values) ** 2))),
        "share_rows_in_shared_group": float(shared.mean()),
        "within_group_sd": float(np.sqrt(((y - mean) ** 2).sum() / (len(y) - same.ngroups))),
        "mean_sd_block_type_month": float(cell.std().mean()),
    }


def gap_correlation(train: pd.DataFrame) -> pd.DataFrame:
    """Correlation of cell month mean ratios at each gap, net of the cell's own level."""
    index = MonthlyRentIndex().fit(train.MONTH_INDEX, train[TARGET])
    rows = train.assign(a=train[TARGET].values / index.transform(train.MONTH_INDEX))
    cells = rows.groupby(["BT", "MONTH_INDEX"]).a.mean().reset_index()
    cells["a"] -= cells.groupby("BT").a.transform("mean")
    results = []
    for gap in range(1, 37):
        shifted = cells.assign(MONTH_INDEX=cells.MONTH_INDEX + gap)
        pairs = cells.merge(shifted, on=["BT", "MONTH_INDEX"], suffixes=("", "_past"))
        results.append({"gap": gap, "corr": float(pairs.a.corr(pairs.a_past)), "pairs": len(pairs)})
    return pd.DataFrame(results)


def plot_gaps(table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    colors = ["tab:red" if gap in (12, 24, 36) else "tab:gray" for gap in table.gap]
    ax.bar(table.gap, table["corr"], color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
    ax.set_xlabel("Gap between two months (months)")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    fig.savefig(FIGURE)
    plt.close(fig)


def validate(train: pd.DataFrame, test: pd.DataFrame) -> None:
    add_cell(test, "data/test.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    limits = [limit_rows(train, "all_train")]
    for label, last_fit in WINDOWS.items():
        fit = train[train.MONTH_INDEX <= last_fit]
        valid = train[train.MONTH_INDEX > last_fit]
        y = valid[TARGET].values
        limits.append(limit_rows(valid, label))
        index_preds = fit_five(fit, valid, last_fit)
        raw_preds = fit_five(fit, valid, last_fit, target_mode="raw", only_lightgbm=True)
        blends = {
            "lightgbm_tuned_index": index_preds["base"],
            "lag_lightgbm_index": index_preds["lag_reg"],
            "lag_blend2": (index_preds["base"] + index_preds["lag_reg"]) / 2,
            "lag_blend5": np.mean(list(index_preds.values()), axis=0),
            "lag_blend_index_raw": np.mean([index_preds["base"], index_preds["lag_reg"], *raw_preds.values()], axis=0),
        }
        for name, prediction in blends.items():
            rows.append({"window": label, "model": name, **regression_metrics(y, prediction)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "validation.csv", index=False)
    pd.DataFrame(limits).to_csv(OUT / "limit.csv", index=False)
    print(pd.DataFrame(limits).round(3).T)

    cut = int(train.MONTH_INDEX.max())
    history = train.assign(a=1.0)
    lags = lag_features(history, test, cut)
    coverage = (lags.filter(like="_n") > 0).groupby(test.MONTH_INDEX.values).mean()
    coverage.rename_axis("month_index").to_csv(OUT / "test_lag_coverage.csv")
    print(coverage.round(2))

    gaps = gap_correlation(train)
    gaps.to_csv(OUT / "gap_correlation.csv", index=False)
    print(gaps.round(3).to_string())
    plot_gaps(gaps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="score the blends instead of writing a submission")
    args = parser.parse_args()
    train, test = load_modeling_data()
    add_cell(train, "data/train.csv")
    if args.validate:
        validate(train, test)
    else:
        submission(train, test)


if __name__ == "__main__":
    main()
