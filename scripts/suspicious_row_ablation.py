"""A/B test: does dropping EDA-flagged suspicious rows help the model?

EDA (../cs5228/project/preprocessing/correlations.ipynb, findings.md,
decisions.md) flags rows that are individually plausible but inconsistent with
each other, via two methods:

* **M1** (`suspect_m1`) — rent extreme *within* its FLAT_TYPE group, and within
  its FLAT_TYPE×TOWN group (IQR fence AND |z| > 3). "Wrong price for this type."
* **M2** (`suspect_rent_low_for_area`) — "big flat, tiny rent": fit rent ~
  FLOOR_AREA_SQM with a global degree-1 line, z-score the residual, flag
  residual z < -3. "Wrong price for this size (too cheap)."
* **both** (`suspect_both` = M1 ∩ M2) — the notebook's high-confidence error
  candidates and its recommended review subset.

The team decided to keep + flag these and let a with/without RMSE comparison
decide. This script runs that A/B for the final model (lightgbm_tuned, G0+G1).
Rows are dropped from the TRAINING fit window only; the Oct 2024–Mar 2025
validation window is left untouched, so the "with" (baseline, already logged)
and "without" runs score the identical rows and the RMSE delta reflects only the
cleaner training data. No change to the shared interface: the cleaned frame is
passed via `run_experiment(..., data=(train_clean, test))`.

Each variant is logged under its own model name (`lightgbm_tuned_no_<flag>`, e.g.
`lightgbm_tuned_no_m1`) rather than reusing `lightgbm_tuned` with only a `notes`
tag. `split_comparison()` keys on (model, feature_set, target_mode) and ignores
`notes`, so sharing the baseline's name would let an ablation's time run
overwrite the baseline in that table and get paired against the wrong split. A
distinct name keeps each experiment in its own bucket.

    python -m scripts.suspicious_row_ablation --flag m2     # default
    python -m scripts.suspicious_row_ablation --flag both   # ~243 high-confidence rows
    python -m scripts.suspicious_row_ablation --flag m1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import TARGET
from src.evaluation import run_experiment, time_holdout_masks
from src.features import load_modeling_data
from src.models import MODELS

MODEL = "lightgbm_tuned"
FEATURE_SET = "G0+G1"
TARGET_MODES = ("raw", "index")


def _iqr_outlier(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def _zscore_outlier(series: pd.Series, thresh: float = 3.0) -> pd.Series:
    sd = series.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=series.index)
    return (series - series.mean()).abs() / sd > thresh


def suspect_m1(train: pd.DataFrame) -> np.ndarray:
    """Rent extreme within FLAT_TYPE or FLAT_TYPE×TOWN (IQR and |z|>3). Notebook M1."""

    g_type = train.groupby("FLAT_TYPE", observed=True)[TARGET]
    g_type_town = train.groupby(["FLAT_TYPE", "TOWN"], observed=True)[TARGET]
    in_type = g_type.transform(_iqr_outlier) & g_type.transform(_zscore_outlier)
    in_type_town = g_type_town.transform(_iqr_outlier) & g_type_town.transform(_zscore_outlier)
    return (in_type | in_type_town).to_numpy()


def suspect_rent_low_for_area(train: pd.DataFrame) -> np.ndarray:
    """Big flat, tiny rent: rent~area degree-1 residual z < -3 (numpy only). Notebook M2 low tail."""

    area = train["FLOOR_AREA_SQM"].to_numpy(dtype=float)
    rent = train[TARGET].to_numpy(dtype=float)
    coef = np.polyfit(area, rent, deg=1)
    resid = rent - np.polyval(coef, area)
    z = (resid - resid.mean()) / resid.std()
    return z < -3


FLAGS = {
    "m1": ("suspect_m1", suspect_m1),
    "m2": ("suspect_rent_low_for_area", suspect_rent_low_for_area),
    "both": ("suspect_both", lambda t: suspect_m1(t) & suspect_rent_low_for_area(t)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flag", default="m2", choices=list(FLAGS))
    args = parser.parse_args()
    name, detector = FLAGS[args.flag]

    train, test = load_modeling_data()
    flag = detector(train)
    fit_mask, valid_mask = time_holdout_masks(train)
    print(f"{name}: {int(flag.sum())} rows "
          f"({int((flag & fit_mask).sum())} in fit window, "
          f"{int((flag & valid_mask).sum())} in valid window)")

    # Drop only the flagged rows in the fit window, so the Oct 2024–Mar 2025
    # validation set is identical to the baseline runs.
    drop_mask = flag & fit_mask
    train_clean = train.loc[~drop_mask].reset_index(drop=True)
    print(f"training rows: {len(train)} -> {len(train_clean)} "
          f"(dropped {len(train) - len(train_clean)})")

    data_clean = (train_clean, test)
    # Log under a variant-specific model name so split_comparison() keeps this
    # experiment separate from the lightgbm_tuned baseline (it keys on model /
    # feature_set / target_mode and ignores notes).
    model_name = f"{MODEL}_no_{args.flag}"
    notes = f"drop {name}"
    for target_mode in TARGET_MODES:
        row = run_experiment(model_name, MODELS[MODEL], feature_set=FEATURE_SET,
                             target_mode=target_mode, member="C", notes=notes,
                             data=data_clean)
        print(f"  {model_name} | {target_mode:5s} | "
              f"RMSE {row['rmse']:.2f} (n_valid {row['n_valid']})")


if __name__ == "__main__":
    main()
