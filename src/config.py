"""Shared paths, seeds and split dates used by every experiment."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
AUXILIARY_DIR = DATA_DIR / "auxiliary"
PROCESSED_DIR = DATA_DIR / "processed"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
BREAKDOWNS_DIR = RESULTS_DIR / "breakdowns"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RESULTS_CSV = RESULTS_DIR / "results.csv"
CONFIG_DIR = PROJECT_ROOT / "configs"

SEED = 5228
TARGET = "MONTHLY_RENT"

# Time holdout from decision 1 of the final report plan. The validation window
# is six months long and starts right after the fit window ends.
TIME_FIT_END = pd.Timestamp("2024-09-01")
TIME_VALID_START = pd.Timestamp("2024-10-01")
TIME_VALID_END = pd.Timestamp("2025-03-01")
RANDOM_CV_FOLDS = 5


def set_seed(seed: int = SEED) -> None:
    """Fix the Python, NumPy and hash seeds for one run."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
