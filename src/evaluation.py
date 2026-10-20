"""One shared evaluation path: splits, metrics, trend handling and the results log.

Usage from any notebook or script:

    from src.evaluation import run_experiment
    from src.models import make_lightgbm
    run_experiment("lightgbm_default", make_lightgbm, feature_set="G0+G1", member="C")

`make_model` is any zero-argument function that returns an object with
`fit(X, y)` and `predict(X)`. X is a DataFrame of the chosen feature columns.
Categorical columns arrive as pandas `category` dtype and numeric columns may
hold NaN, so each model handles its own encoding and imputation.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import (
    BREAKDOWNS_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    RANDOM_CV_FOLDS,
    RESULTS_CSV,
    RUNS_DIR,
    SEED,
    TARGET,
    TIME_FIT_END,
    TIME_VALID_END,
    TIME_VALID_START,
    set_seed,
)
from src.features import feature_columns, load_modeling_data

SPLITS = ("time", "random5")
TARGET_MODES = ("raw", "index")


# ---------------------------------------------------------------- metrics


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / y_true) * 100),
        "bias": float(np.mean(error)),
    }


def segment_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """RMSE, MAE and MAPE per flat type and per validation month."""

    data = pd.DataFrame(
        {
            "flat_type": frame["FLAT_TYPE"].astype(str).to_numpy(),
            "month": frame["RENT_APPROVAL_DATE"].dt.strftime("%Y-%m").to_numpy(),
            "y": frame[TARGET].to_numpy(),
            "p": prediction,
        }
    )
    rows = []
    for segment_type in ("flat_type", "month"):
        for segment, part in data.groupby(segment_type, sort=True):
            rows.append({"segment_type": segment_type, "segment": segment, "n": len(part),
                         **regression_metrics(part["y"], part["p"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- splits


@dataclass
class Fold:
    name: str
    fit_index: np.ndarray
    valid_index: np.ndarray


def time_holdout_masks(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit on Jan 2021 to Sep 2024, validate on Oct 2024 to Mar 2025."""

    month = train["RENT_APPROVAL_DATE"]
    fit = (month <= TIME_FIT_END).to_numpy()
    valid = ((month >= TIME_VALID_START) & (month <= TIME_VALID_END)).to_numpy()
    return fit, valid


def iter_folds(train: pd.DataFrame, split: str) -> Iterator[Fold]:
    if split == "time":
        fit, valid = time_holdout_masks(train)
        yield Fold("time", np.flatnonzero(fit), np.flatnonzero(valid))
    elif split == "random5":
        splitter = KFold(n_splits=RANDOM_CV_FOLDS, shuffle=True, random_state=SEED)
        for number, (fit_index, valid_index) in enumerate(splitter.split(train)):
            yield Fold(f"random{number}", fit_index, valid_index)
    else:
        raise ValueError(f"split must be one of {SPLITS}")


# ---------------------------------------------------------------- trend handling


class MonthlyRentIndex:
    """Rent level per month, used to take the time trend out of the target.

    Months seen in the fit data use their median rent. Later months use a
    straight line fitted to the last `window` monthly medians, so the index
    keeps rising past the last training month. Tree models cannot do that on
    their own, which is the point of decision 5 in the plan.
    """

    def __init__(self, window: int = 12):
        self.window = window

    def fit(self, month_index: pd.Series, rent: pd.Series) -> "MonthlyRentIndex":
        medians = rent.groupby(month_index.to_numpy()).median().sort_index()
        self.medians_ = medians
        tail = medians.iloc[-self.window:]
        self.slope_, self.intercept_ = np.polyfit(tail.index.to_numpy(float), tail.to_numpy(float), 1)
        self.last_month_ = int(medians.index.max())
        return self

    def transform(self, month_index: pd.Series) -> np.ndarray:
        months = month_index.to_numpy()
        trend = self.intercept_ + self.slope_ * months.astype(float)
        observed = pd.Series(months).map(self.medians_).to_numpy(dtype=float)
        level = np.where(np.isnan(observed), trend, observed)
        if np.any(level <= 0):
            raise ValueError("Rent index fell to zero or below")
        return level


# ---------------------------------------------------------------- one fit


def fit_predict(make_model: Callable, fit_frame: pd.DataFrame, predict_frame: pd.DataFrame,
                columns: list[str], target_mode: str = "raw") -> tuple[np.ndarray, float, float]:
    """Fit on one frame, predict another. Returns (prediction, fit_s, predict_s)."""

    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}")
    y = fit_frame[TARGET].to_numpy(dtype=float)
    if target_mode == "index":
        index = MonthlyRentIndex().fit(fit_frame["MONTH_INDEX"], fit_frame[TARGET])
        y = y / index.transform(fit_frame["MONTH_INDEX"])

    set_seed()
    model = make_model()
    start = time.perf_counter()
    model.fit(fit_frame[columns], y)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = np.asarray(model.predict(predict_frame[columns]), dtype=float)
    predict_seconds = time.perf_counter() - start

    if target_mode == "index":
        prediction = prediction * index.transform(predict_frame["MONTH_INDEX"])
    return prediction, fit_seconds, predict_seconds


# ---------------------------------------------------------------- experiment log

RESULT_COLUMNS = [
    "run_id", "timestamp", "member", "model", "feature_set", "n_features", "split",
    "target_mode", "n_fit", "n_valid", "rmse", "rmse_std", "mae", "mape", "bias",
    "train_seconds", "predict_seconds", "seed", "git_commit", "notes",
]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


def collect_results() -> pd.DataFrame:
    """Rebuild results/results.csv from the one-row files in results/runs/.

    Each run writes its own file, so two members never edit the same line and
    git merges stay clean. results.csv is the combined view.
    """

    files = sorted(RUNS_DIR.glob("*.csv"))
    table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=RESULT_COLUMNS)
    table = table.reindex(columns=RESULT_COLUMNS).sort_values("timestamp")
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False)
    return table


def run_experiment(model_name: str, make_model: Callable, feature_set: str = "G0+G1",
                   split: str = "time", target_mode: str = "raw", member: str = "",
                   notes: str = "", data: tuple[pd.DataFrame, pd.DataFrame] | None = None,
                   save_predictions: bool = True, log: bool = True) -> dict:
    """Evaluate one model on one split and log a row to the shared results.

    Pass `data=load_modeling_data()` when running many experiments in one
    session to avoid reloading. Set `log=False` for quick trials that should
    not appear in the results table.
    """

    train, _ = data if data is not None else load_modeling_data()
    columns = feature_columns(feature_set)
    fold_rows, predictions, breakdowns = [], [], []
    for fold in iter_folds(train, split):
        fit_frame = train.iloc[fold.fit_index]
        valid_frame = train.iloc[fold.valid_index]
        prediction, fit_s, predict_s = fit_predict(make_model, fit_frame, valid_frame, columns, target_mode)
        fold_rows.append({"n_fit": len(fit_frame), "n_valid": len(valid_frame), "train_seconds": fit_s,
                          "predict_seconds": predict_s, **regression_metrics(valid_frame[TARGET], prediction)})
        predictions.append(pd.DataFrame({"ROW_ID": valid_frame["ROW_ID"].to_numpy(), "fold": fold.name,
                                         "y_true": valid_frame[TARGET].to_numpy(), "y_pred": prediction}))
        if split == "time":
            breakdowns.append(segment_metrics(valid_frame, prediction))

    folds = pd.DataFrame(fold_rows)
    stamp = datetime.now()
    run_id = re.sub(r"[^A-Za-z0-9_.+-]", "-", f"{stamp:%Y%m%d-%H%M%S}_{model_name}_{feature_set}_{split}_{target_mode}")
    row = {
        "run_id": run_id, "timestamp": stamp.isoformat(timespec="seconds"), "member": member,
        "model": model_name, "feature_set": feature_set, "n_features": len(columns), "split": split,
        "target_mode": target_mode, "n_fit": int(folds["n_fit"].mean()), "n_valid": int(folds["n_valid"].sum()),
        "rmse": folds["rmse"].mean(), "rmse_std": folds["rmse"].std(ddof=0), "mae": folds["mae"].mean(),
        "mape": folds["mape"].mean(), "bias": folds["bias"].mean(),
        "train_seconds": folds["train_seconds"].sum(), "predict_seconds": folds["predict_seconds"].sum(),
        "seed": SEED, "git_commit": _git_commit(), "notes": notes,
    }
    if log:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(RUNS_DIR / f"{run_id}.csv", index=False)
        if breakdowns:
            BREAKDOWNS_DIR.mkdir(parents=True, exist_ok=True)
            pd.concat(breakdowns).to_csv(BREAKDOWNS_DIR / f"{run_id}.csv", index=False)
        if save_predictions:
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            pd.concat(predictions).to_parquet(PREDICTIONS_DIR / f"{run_id}.parquet", index=False)
        collect_results()
    print(f"{model_name:<24} {feature_set:<10} {split:<8} {target_mode:<6} "
          f"RMSE {row['rmse']:8.2f}  MAE {row['mae']:8.2f}  MAPE {row['mape']:6.2f}%  "
          f"fit {row['train_seconds']:.1f}s")
    return row


def make_submission(make_model: Callable, feature_set: str, target_mode: str = "raw",
                    path=None, data: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Fit on all training rows and write a Kaggle file with columns Id and Predicted."""

    train, test = data if data is not None else load_modeling_data()
    prediction, _, _ = fit_predict(make_model, train, test, feature_columns(feature_set), target_mode)
    submission = pd.DataFrame({"Id": test["ROW_ID"].to_numpy(), "Predicted": prediction})
    if path is not None:
        submission.to_csv(path, index=False)
    return submission
