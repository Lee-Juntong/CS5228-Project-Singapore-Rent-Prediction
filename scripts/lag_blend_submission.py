"""Five model repeat-rent blend. See docs/repeat_rent_blend.md.

    python -m scripts.lag_blend_submission
        Fits on all training rows and writes submissions/lag_blend5_G0G1_index.csv.

    python -m scripts.lag_blend_submission --validate
        Scores the blends on the time holdout and on a 16 month window, measures
        the limit on any model and the correlation of rents by gap. Writes
        results/member_a/repeat_rent/*.csv and report/figures/lag_correlation.pdf.

    python -m scripts.lag_blend_submission --analyse
        Repeats the residual and permutation importance analysis on the blend.
        Writes results/member_a/repeat_rent/{error_analysis,permutation_importance}.csv
        and report/figures/blend_importance.pdf.
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

from src.config import PROJECT_ROOT, SEED, TARGET
from src.evaluation import MonthlyRentIndex, regression_metrics
from src.features import feature_columns, load_modeling_data

warnings.filterwarnings("ignore")

OUT = PROJECT_ROOT / "results" / "member_a" / "repeat_rent"
SUBMISSION = PROJECT_ROOT / "submissions" / "lag_blend5_G0G1_index.csv"
FIGURE = PROJECT_ROOT / "report" / "figures" / "lag_correlation.pdf"
IMPORTANCE_FIGURE = PROJECT_ROOT / "report" / "figures" / "blend_importance.pdf"
IMPORTANCE_ROWS = 2000
IMPORTANCE_REPEATS = 5
LABELS = {
    "FLAT_TYPE_ORDINAL": "Flat type ordinal",
    "FLAT_TYPE": "Flat type",
    "HDB_SUBZONE": "Subzone",
    "HDB_LATITUDE": "Latitude",
    "DISTANCE_TO_RAFFLES_PLACE_KM": "Distance to Raffles Place",
    "FLOOR_AREA_SQM": "Floor area",
    "FLAT_MODEL": "Flat model",
}

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


def as_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns[frame.dtypes == "category"]:
        frame[column] = frame[column].cat.codes
    return frame


def fit_members(fit: pd.DataFrame, cut: int, target_mode: str = "index", only_lightgbm: bool = False) -> tuple:
    """Fit the blend members. Returns the rent index, the fit rows with `a`, and the models."""
    index = MonthlyRentIndex().fit(fit.MONTH_INDEX, fit[TARGET])
    if target_mode == "index":
        fit = fit.assign(a=fit[TARGET].values / index.transform(fit.MONTH_INDEX))
    else:
        fit = fit.assign(a=fit[TARGET].values.astype(float))
    fit_lags = lag_features(fit, fit, cut)
    X = pd.concat([fit[COLUMNS], fit_lags], axis=1)
    models = {
        "base": lgb.LGBMRegressor(**LGBM).fit(fit[COLUMNS], fit.a),
        "lag_reg": lgb.LGBMRegressor(**LGBM_LAG).fit(X, fit.a),
    }
    if not only_lightgbm:
        X_num = as_numeric(X)
        models["xgb"] = xgb.XGBRegressor(**XGB).fit(X_num, fit.a)
        models["cat"] = CatBoostRegressor(**CATBOOST).fit(X_num, fit.a)
        models["rf"] = RandomForestRegressor(**FOREST).fit(X_num.fillna(-1), fit.a)
    return index, fit, models


def predict_members(models: dict, X_valid: pd.DataFrame, scale) -> dict:
    predictions = {
        "base": models["base"].predict(X_valid[COLUMNS]) * scale,
        "lag_reg": models["lag_reg"].predict(X_valid) * scale,
    }
    if "xgb" in models:
        X_valid_num = as_numeric(X_valid)
        predictions["xgb"] = models["xgb"].predict(X_valid_num) * scale
        predictions["cat"] = models["cat"].predict(X_valid_num) * scale
        predictions["rf"] = models["rf"].predict(X_valid_num.fillna(-1)) * scale
    return predictions


def fit_five(fit: pd.DataFrame, valid: pd.DataFrame, cut: int, target_mode: str = "index", only_lightgbm: bool = False) -> dict:
    """Fit the blend members on `fit` and return their rent predictions for `valid`."""
    index, fitted, models = fit_members(fit, cut, target_mode, only_lightgbm)
    X_valid = pd.concat([valid[COLUMNS], lag_features(fitted, valid, cut)], axis=1)
    scale = index.transform(valid.MONTH_INDEX) if target_mode == "index" else 1.0
    return predict_members(models, X_valid, scale)


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


def error_summary(valid: pd.DataFrame, prediction: np.ndarray) -> dict:
    """The residual measures of docs/analysis.md, for one set of holdout predictions."""
    y = valid[TARGET].values
    error = prediction - y
    squared = error**2
    total = squared.sum()
    share = lambda mask: float(squared[mask].sum() / total)
    flat, town = valid.FLAT_TYPE.astype(str).values, valid.TOWN.astype(str).values
    low, very_low = y < 2000, y < 1000
    central = valid.DISTANCE_TO_RAFFLES_PLACE_KM.values < 5
    big_flat, mid_flat = np.isin(flat, ["5-room", "executive"]), np.isin(flat, ["3-room", "4-room"])
    largest = np.argsort(-np.abs(error))[:20]
    towns = pd.Series(squared).groupby(town).sum().sort_values(ascending=False) / total
    groups = valid[COLUMNS].astype(str).agg("|".join, axis=1).values
    repeated = pd.Series(groups).duplicated(keep=False).values
    deciles = pd.Series(error).groupby(pd.qcut(prediction, 10, labels=False, duplicates="drop")).mean()
    result = regression_metrics(y, prediction)
    result.update(
        rows_error_over_1000=float((np.abs(error) > 1000).mean()),
        squared_error_over_1000=share(np.abs(error) > 1000),
        rows_rent_below_2000=int(low.sum()),
        share_rent_below_2000=float(low.mean()),
        squared_rent_below_2000=share(low),
        bias_rent_below_2000=float(error[low].mean()),
        squared_rent_below_1000=share(very_low),
        top20_over_predictions=int((error[largest] > 0).sum()),
        top20_rent_below_1000=int((y[largest] < 1000).sum()),
        squared_5room_executive=share(big_flat),
        squared_3room_4room=share(mid_flat),
        share_within_5km=float(central.mean()),
        squared_within_5km=share(central),
        rmse_within_5km=float(np.sqrt(squared[central].mean())),
        top_town=towns.index[0],
        squared_top_town=float(towns.iloc[0]),
        rows_repeated=int(repeated.sum()),
        rmse_repeated=float(np.sqrt(squared[repeated].mean())),
        rmse_unique=float(np.sqrt(squared[~repeated].mean())),
        decile_bias_min=float(deciles.min()),
        decile_bias_max=float(deciles.max()),
    )
    for month, group in pd.DataFrame({"m": valid.MONTH_INDEX.values, "e": error}).groupby("m").e:
        result[f"rmse_month_{month}"] = float(np.sqrt((group**2).mean()))
        result[f"bias_month_{month}"] = float(group.mean())
    return result


def blend_importance(valid: pd.DataFrame, fitted: pd.DataFrame, index, models: dict, cut: int) -> pd.DataFrame:
    """Permutation importance of the blend on the sample used in docs/analysis.md."""
    from src.analysis_interpretation import heldout_sample

    sample = heldout_sample(valid, IMPORTANCE_ROWS, SEED)
    X = pd.concat([sample[COLUMNS], lag_features(fitted, sample, cut)], axis=1)
    y = sample[TARGET].values

    def score(frame: pd.DataFrame) -> float:
        scale = index.transform(frame.MONTH_INDEX)
        return regression_metrics(y, np.mean(list(predict_members(models, frame, scale).values()), axis=0))["rmse"]

    baseline = score(X)
    inputs = {column: [column] for column in COLUMNS}
    inputs.update({f"Rent lags: {name}": [f"{name}_{stat}" for stat in ("mean", "n", "min", "max")] for name in GAPS})
    rng = np.random.default_rng(SEED)
    rows = []
    for name, columns in inputs.items():
        rises = []
        for _ in range(IMPORTANCE_REPEATS):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            for column in columns:
                shuffled[column] = X[column].iloc[order].values
            rises.append(score(shuffled) - baseline)
        rows.append({"feature": name, "importance_mean_sgd": float(np.mean(rises)), "importance_std_sgd": float(np.std(rises, ddof=1))})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows).sort_values("importance_mean_sgd", ascending=False)
    table["sample_rmse"] = baseline
    return table


def plot_importance(table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8})
    top = table.head(10).iloc[::-1]
    labels = [LABELS.get(name, name.replace("Rent lags: L", "Rent lags, ").replace("Rent lags: OFF", "Rent lags, other gaps")) for name in top.feature]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    colors = ["tab:red" if name.startswith("Rent lags") else "tab:blue" for name in top.feature]
    ax.barh(labels, top.importance_mean_sgd, xerr=top.importance_std_sgd, color=colors, error_kw={"linewidth": 0.7})
    ax.set_xlabel("Rise in RMSE (SGD)")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_FIGURE)
    plt.close(fig)


def analyse(train: pd.DataFrame) -> None:
    """Repeat the residual and importance analysis on the blend. Holdout only."""
    OUT.mkdir(parents=True, exist_ok=True)
    cut = WINDOWS["holdout"]
    fit, valid = train[train.MONTH_INDEX <= cut], train[train.MONTH_INDEX > cut]
    index, fitted, models = fit_members(fit, cut)
    X_valid = pd.concat([valid[COLUMNS], lag_features(fitted, valid, cut)], axis=1)
    predictions = predict_members(models, X_valid, index.transform(valid.MONTH_INDEX))
    summary = pd.DataFrame(
        {
            "lightgbm_tuned_index": error_summary(valid, predictions["base"]),
            "lag_blend5": error_summary(valid, np.mean(list(predictions.values()), axis=0)),
        }
    )
    summary.rename_axis("measure").to_csv(OUT / "error_analysis.csv")
    print(summary.to_string())
    table = blend_importance(valid, fitted, index, models, cut)
    table.to_csv(OUT / "permutation_importance.csv", index=False)
    plot_importance(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="score the blends instead of writing a submission")
    parser.add_argument("--analyse", action="store_true", help="repeat the error analysis on the blend")
    args = parser.parse_args()
    train, test = load_modeling_data()
    add_cell(train, "data/train.csv")
    if args.analyse:
        analyse(train)
    elif args.validate:
        validate(train, test)
    else:
        submission(train, test)


if __name__ == "__main__":
    main()
