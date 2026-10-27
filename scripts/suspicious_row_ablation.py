"""A/B test: does dropping the ~50 "big flat, tiny rent" rows help the model?

EDA (../cs5228/project/preprocessing/findings.md, decisions.md) flagged a
`suspect_rent_low_for_area` subset via Method 2: fit rent ~ FLOOR_AREA_SQM with a
global degree-1 line, z-score the residual, flag rows with residual z < -3 ("big
flat, tiny rent"). The team decided to keep + flag these and let a with/without
RMSE comparison decide. This script runs that comparison for the final boosting
model.

The suspicious rows are dropped from the TRAINING rows only; the Oct 2024–Mar
2025 validation window is left untouched, so the "with" (baseline, already
logged) and "without" runs score the identical rows and the RMSE delta reflects
only the cleaner training data. No change to the shared interface: the cleaned
frame is passed via `run_experiment(..., data=(train_clean, test))`.

    python -m scripts.suspicious_row_ablation
"""

from __future__ import annotations

import numpy as np

from src.config import TARGET
from src.evaluation import run_experiment, time_holdout_masks
from src.features import load_modeling_data
from src.models import MODELS

# The final-model config. "with" numbers are the existing baseline runs.
MODEL = "lightgbm_tuned"
FEATURE_SET = "G0+G1"
TARGET_MODES = ("raw", "index")
NOTES = "drop suspect_rent_low_for_area"


def suspect_rent_low_for_area(train) -> np.ndarray:
    """Reproduce the EDA Method-2 flag on our training rows (numpy only).

    coef = polyfit(area, rent, 1); resid = rent - fit; flag resid z-score < -3.
    Recomputed on our 149,575-row frame (the EDA ran on the raw 150k), so the
    count may differ slightly from the notebook's ~50 — we report what we get.
    """

    area = train["FLOOR_AREA_SQM"].to_numpy(dtype=float)
    rent = train[TARGET].to_numpy(dtype=float)
    coef = np.polyfit(area, rent, deg=1)
    resid = rent - np.polyval(coef, area)
    z = (resid - resid.mean()) / resid.std()
    return z < -3


def main() -> None:
    train, test = load_modeling_data()
    flag = suspect_rent_low_for_area(train)
    fit_mask, valid_mask = time_holdout_masks(train)
    n_total = int(flag.sum())
    n_in_fit = int((flag & fit_mask).sum())
    n_in_valid = int((flag & valid_mask).sum())
    print(f"suspect_rent_low_for_area: {n_total} rows "
          f"({n_in_fit} in fit window, {n_in_valid} in valid window)")

    # Drop only the flagged rows that sit in the fit window, so the Oct 2024–Mar
    # 2025 validation set is identical to the baseline runs.
    drop_mask = flag & fit_mask
    train_clean = train.loc[~drop_mask].reset_index(drop=True)
    print(f"training rows: {len(train)} -> {len(train_clean)} "
          f"(dropped {len(train) - len(train_clean)})")

    data_clean = (train_clean, test)
    for target_mode in TARGET_MODES:
        row = run_experiment(MODEL, MODELS[MODEL], feature_set=FEATURE_SET,
                             target_mode=target_mode, member="C", notes=NOTES,
                             data=data_clean)
        print(f"  without suspect rows | {target_mode:5s} | "
              f"RMSE {row['rmse']:.2f} (n_valid {row['n_valid']})")


if __name__ == "__main__":
    main()
