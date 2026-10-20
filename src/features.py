"""Load the frozen modeling data and define the four fixed feature groups.

Every experiment picks its columns through `feature_columns`, so all members
compare models on exactly the same inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TARGET
from src.external_features import BLOCK_EXTRA_COLUMNS, CPI_COLUMNS, add_external_features, load_block_extras

CATEGORICAL_COLUMNS = ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "HDB_SUBZONE"]

FEATURE_GROUPS: dict[str, list[str]] = {
    # G0 core: the flat itself, its town and the approval month.
    "G0": [
        "TOWN",
        "FLAT_TYPE",
        "FLAT_TYPE_ORDINAL",
        "FLAT_MODEL",
        "FLOOR_AREA_SQM",
        "LOG_FLOOR_AREA_SQM",
        "FLOOR_AREA_PER_ROOM_PROXY",
        "LEASE_COMMENCE_DATE",
        "FLAT_AGE_YEARS",
        "INVALID_LEASE_DATE_FLAG",
        "MONTH_INDEX",
        "MONTH_SIN",
        "MONTH_COS",
    ],
    # G1 location: coordinates, subzone and distances or counts to amenities.
    "G1": [
        "HDB_LATITUDE",
        "HDB_LONGITUDE",
        "HDB_SUBZONE",
        "DISTANCE_TO_RAFFLES_PLACE_KM",
        "NEAREST_OPEN_MRT_KM",
        "OPEN_MRT_COUNT_500M",
        "OPEN_MRT_COUNT_1KM",
        "OPEN_MRT_COUNT_2KM",
        "NEAREST_OPEN_INTERCHANGE_KM",
        "NEAREST_PLANNED_MRT_KM",
        "PLANNED_MRT_COUNT_1KM",
        "PLANNED_MRT_COUNT_2KM",
        "NEAREST_SCHOOL_KM",
        "SCHOOL_COUNT_1KM",
        "SCHOOL_COUNT_2KM",
        "NEAREST_PRIMARY_SCHOOL_KM",
        "PRIMARY_SCHOOL_COUNT_1KM",
        "PRIMARY_SCHOOL_COUNT_2KM",
        "NEAREST_MALL_KM",
        "MALL_COUNT_1KM",
        "MALL_COUNT_2KM",
        *BLOCK_EXTRA_COLUMNS,
    ],
    # G2 block: attributes of the HDB block from the block table.
    "G2": [
        "HDB_MAX_FLOOR",
        "HDB_YEAR_COMPLETED",
        "HDB_BLOCK_AGE_YEARS",
        "HDB_MAX_FLOOR_MISSING_FLAG",
        "HDB_YEAR_COMPLETED_MISSING_FLAG",
    ],
    # G3 market: values for the approval month itself. These would not be
    # known at prediction time (decision 6), which the ablation checks.
    "G3": [
        "COE_A_PREMIUM_CURRENT",
        "COE_A_DEMAND_RATIO_CURRENT",
        "COE_B_PREMIUM_CURRENT",
        "COE_B_DEMAND_RATIO_CURRENT",
        *CPI_COLUMNS,
        "STOCK_SG_MARKET_RETURN_CURRENT",
        "STOCK_SG_POSITIVE_BREADTH_CURRENT",
        "STOCK_BANK_RETURN_CURRENT",
    ],
}

# Named feature sets for the ablation in Phase 2. Phase 1 uses "G0+G1".
FEATURE_SETS = {
    "G0": ["G0"],
    "G0+G1": ["G0", "G1"],
    "G0+G1+G2": ["G0", "G1", "G2"],
    "ALL": ["G0", "G1", "G2", "G3"],
}


def parse_feature_set(name: str) -> list[str]:
    """Accept a named set such as "G0+G1" or "ALL" and return its group names."""

    if name in FEATURE_SETS:
        return FEATURE_SETS[name]
    groups = name.split("+")
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise KeyError(f"Unknown feature groups {unknown}; choose from {list(FEATURE_GROUPS)}")
    return groups


def feature_columns(feature_set: str) -> list[str]:
    """Columns for a feature set, in a fixed order with no duplicates."""

    columns: list[str] = []
    for group in parse_feature_set(feature_set):
        columns += [column for column in FEATURE_GROUPS[group] if column not in columns]
    return columns


def load_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) with every grouped column ready for modeling.

    Categorical columns share one category list across train and test, so
    category codes mean the same thing in both frames.
    """

    train = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")
    extras = load_block_extras([train, test])
    train = add_external_features(train, extras)
    test = add_external_features(test, extras)

    for frame in (train, test):
        frame["LOG_FLOOR_AREA_SQM"] = np.log(frame["FLOOR_AREA_SQM"]).astype("float32")
    for column in CATEGORICAL_COLUMNS:
        categories = sorted(set(train[column].dropna()) | set(test[column].dropna()))
        for frame in (train, test):
            frame[column] = pd.Categorical(frame[column].astype(object), categories=categories)

    every_column = feature_columns("ALL")
    missing = [column for column in every_column if column not in train or column not in test]
    if missing:
        raise KeyError(f"Feature columns missing from the data: {missing}")
    if TARGET not in train:
        raise KeyError(f"{TARGET} missing from training data")
    return train, test
