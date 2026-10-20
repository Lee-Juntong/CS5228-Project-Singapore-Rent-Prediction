"""Reproducible preprocessing and feature engineering for the CS5228 project.

The module keeps target-free transformations separate from target-derived
features.  ``run_pipeline`` creates the processed train/test datasets and the
audit artifacts used by the companion notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]


EARTH_RADIUS_KM = 6_371.0088
BASE_MONTH = pd.Timestamp("2021-01-01")
ADDRESS_COLUMNS = ["TOWN", "BLOCK", "STREET"]
TEXT_COLUMNS = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL"]

CORE_PREDICTOR_COLUMNS = [
    "RENT_APPROVAL_DATE",
    "TOWN",
    "BLOCK",
    "STREET",
    "FLAT_TYPE",
    "FLAT_MODEL",
    "FLOOR_AREA_SQM",
    "FURNISHED",
    "LEASE_COMMENCE_DATE",
    "FEE",
]

DATASET_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "example_submission": "example-submission.csv",
    "hdb": "auxiliary/sg-hdb-block.csv",
    "mrt": "auxiliary/sg-mrt-stations.csv",
    "schools": "auxiliary/sg-schools.csv",
    "malls": "auxiliary/sg-shopping-malls.csv",
    "coe": "auxiliary/sg-coe-prices.csv",
    "stocks": "auxiliary/sg-stock-prices.csv",
}


def normalize_text(values: pd.Series) -> pd.Series:
    """Normalize text without changing missing values into the string 'nan'."""

    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def canonicalize_flat_type(values: pd.Series) -> pd.Series:
    """Map spaced and hyphenated HDB flat labels to one representation."""

    normalized = normalize_text(values)
    room_mask = normalized.str.fullmatch(r"[1-5][ -]?room", na=False)
    normalized.loc[room_mask] = normalized.loc[room_mask].str.extract(
        r"([1-5])", expand=False
    ) + "-room"
    allowed = {"1-room", "2-room", "3-room", "4-room", "5-room", "executive"}
    unexpected = sorted(set(normalized.dropna().unique()) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected FLAT_TYPE values: {unexpected}")
    return normalized


def parse_numeric_text(values: pd.Series) -> pd.Series:
    """Parse currency and comma-separated count fields as numeric values."""

    cleaned = (
        values.astype("string")
        .str.replace(r"[$,\s]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="raise")


def load_raw_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every supplied CSV while preserving identifier columns as text."""

    data_dir = Path(data_dir)
    datasets: dict[str, pd.DataFrame] = {}
    for name, relative_path in DATASET_FILES.items():
        path = data_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset: {path}")
        dtype: dict[str, str] | None = None
        if name in {"train", "test"}:
            dtype = {
                "TOWN": "string",
                "BLOCK": "string",
                "STREET": "string",
                "FLAT_TYPE": "string",
                "FLAT_MODEL": "string",
                "FURNISHED": "string",
            }
        elif name == "hdb":
            dtype = {
                "TOWN": "string",
                "BLOCK": "string",
                "STREET": "string",
                "POSTAL_CODE": "string",
            }
        elif name == "schools":
            dtype = {"POSTAL_CODE": "string"}
        datasets[name] = pd.read_csv(path, dtype=dtype)
    return datasets


def audit_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-friendly structural diagnostics for one table."""

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if int(count) > 0
        },
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    }


def validate_core_schema(frame: pd.DataFrame, *, is_train: bool) -> None:
    """Validate the raw transaction schema and constant fields."""

    required = CORE_PREDICTOR_COLUMNS + (["MONTHLY_RENT"] if is_train else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[required].isna().any().any():
        columns = frame[required].columns[frame[required].isna().any()].tolist()
        raise ValueError(f"Unexpected missing values in core columns: {columns}")

    furnished = set(normalize_text(frame["FURNISHED"]).dropna().unique())
    if furnished != {"yes"}:
        raise ValueError(f"FURNISHED is no longer constant 'yes': {sorted(furnished)}")
    if not frame["FEE"].eq(0).all():
        raise ValueError("FEE is no longer constant 0.0")

    parsed_dates = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("RENT_APPROVAL_DATE contains invalid YYYY-MM values")
    if (frame["FLOOR_AREA_SQM"] <= 0).any():
        raise ValueError("FLOOR_AREA_SQM must be positive")
    if is_train and (frame["MONTHLY_RENT"] <= 0).any():
        raise ValueError("MONTHLY_RENT must be positive")


def _join_coverage(
    transactions: pd.DataFrame,
    hdb: pd.DataFrame,
    *,
    normalize: bool,
) -> dict[str, int | float]:
    left = transactions[ADDRESS_COLUMNS].copy()
    right = hdb[ADDRESS_COLUMNS].copy()
    if normalize:
        for column in ADDRESS_COLUMNS:
            left[column] = normalize_text(left[column])
            right[column] = normalize_text(right[column])
    right = right.drop_duplicates()
    matched = left.merge(right, on=ADDRESS_COLUMNS, how="left", indicator=True)
    matched_rows = int(matched["_merge"].eq("both").sum())
    return {
        "rows": int(len(left)),
        "matched_rows": matched_rows,
        "unmatched_rows": int(len(left) - matched_rows),
        "coverage_rate": float(matched_rows / len(left)),
    }


def validate_auxiliary_data(datasets: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Validate auxiliary schemas, ranges, sentinels, and key uniqueness."""

    hdb = datasets["hdb"]
    normalized_hdb = hdb.copy()
    for column in ADDRESS_COLUMNS:
        normalized_hdb[column] = normalize_text(normalized_hdb[column])
    if normalized_hdb.duplicated(ADDRESS_COLUMNS).any():
        raise ValueError("HDB address keys are not unique after normalization")
    if not hdb["LATITUDE"].between(1.0, 1.6).all() or not hdb[
        "LONGITUDE"
    ].between(103.5, 104.2).all():
        raise ValueError("HDB coordinates fall outside the expected Singapore range")

    max_floor = pd.to_numeric(hdb["MAX_FLOOR"], errors="raise")
    completion_year = pd.to_numeric(hdb["YEAR_COMPLETED"], errors="raise")
    unexpected_floor = max_floor.le(0) & max_floor.ne(-1)
    unexpected_year = completion_year.ne(-1) & ~completion_year.between(1900, 2100)
    if unexpected_floor.any():
        raise ValueError("HDB MAX_FLOOR contains a nonpositive value other than -1")
    if unexpected_year.any():
        raise ValueError("HDB YEAR_COMPLETED contains an invalid value other than -1")

    poi_checks: dict[str, dict[str, Any]] = {}
    for name in ["mrt", "schools", "malls"]:
        points = datasets[name]
        coordinates_valid = bool(
            points["LATITUDE"].between(1.0, 1.6).all()
            and points["LONGITUDE"].between(103.5, 104.2).all()
        )
        if not coordinates_valid:
            raise ValueError(f"{name} coordinates fall outside the expected Singapore range")
        poi_checks[name] = {
            "rows": int(len(points)),
            "coordinates_valid": coordinates_valid,
            "duplicate_coordinate_pairs": int(
                points.duplicated(["LATITUDE", "LONGITUDE"]).sum()
            ),
        }

    coe = datasets["coe"]
    coe_key = ["YEAR", "CATEGORY", "MONTH", "ROUND"]
    if coe.duplicated(coe_key).any():
        raise ValueError("COE year/category/month/round keys must be unique")
    quota = parse_numeric_text(coe["QUOTA"])
    bids = parse_numeric_text(coe["BIDS_RECEIVED"])
    if quota.le(0).any() or bids.lt(0).any():
        raise ValueError("COE quota must be positive and bids must be nonnegative")

    stocks = datasets["stocks"].copy()
    stocks["DATE"] = pd.to_datetime(stocks["DATE"], errors="coerce")
    if stocks["DATE"].isna().any():
        raise ValueError("Stock DATE contains invalid values")
    if stocks.duplicated(["SYMBOL", "DATE"]).any():
        raise ValueError("Stock SYMBOL/DATE keys must be unique")
    adjusted_close = pd.to_numeric(stocks["ADJUSTED_CLOSE"], errors="coerce")
    if adjusted_close.dropna().le(0).any():
        raise ValueError("Stock adjusted closes must be positive when present")

    postal = normalize_text(hdb["POSTAL_CODE"])
    return {
        "hdb": {
            "normalized_address_keys_unique": True,
            "coordinates_valid": True,
            "max_floor_sentinel_rows": int(max_floor.eq(-1).sum()),
            "completion_year_sentinel_rows": int(completion_year.eq(-1).sum()),
            "postal_code_nil_rows": int(postal.eq("nil").sum()),
            "unexpected_max_floor_values": int(unexpected_floor.sum()),
            "unexpected_completion_year_values": int(unexpected_year.sum()),
        },
        "points_of_interest": poi_checks,
        "coe": {
            "composite_keys_unique": True,
            "positive_quota": True,
            "nonnegative_bids": True,
        },
        "stocks": {
            "symbol_date_keys_unique": True,
            "invalid_dates": 0,
            "missing_adjusted_close_rows": int(adjusted_close.isna().sum()),
            "positive_observed_adjusted_closes": True,
        },
    }


def audit_raw_data(datasets: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Audit supplied files before applying any transformations."""

    train = datasets["train"]
    test = datasets["test"]
    hdb = datasets["hdb"]
    validate_core_schema(train, is_train=True)
    validate_core_schema(test, is_train=False)

    predictor_group = train.groupby(CORE_PREDICTOR_COLUMNS, dropna=False)[
        "MONTHLY_RENT"
    ].agg(["size", "nunique"])
    train_dates = pd.to_datetime(train["RENT_APPROVAL_DATE"], format="%Y-%m")
    test_dates = pd.to_datetime(test["RENT_APPROVAL_DATE"], format="%Y-%m")
    approval_year = train_dates.dt.year
    invalid_lease = train["LEASE_COMMENCE_DATE"].gt(approval_year)

    auxiliary_checks = validate_auxiliary_data(datasets)

    return {
        "datasets": {name: audit_frame(frame) for name, frame in datasets.items()},
        "core": {
            "train_date_min": train_dates.min().strftime("%Y-%m"),
            "train_date_max": train_dates.max().strftime("%Y-%m"),
            "test_date_min": test_dates.min().strftime("%Y-%m"),
            "test_date_max": test_dates.max().strftime("%Y-%m"),
            "overlap_months": sorted(
                set(train["RENT_APPROVAL_DATE"]) & set(test["RENT_APPROVAL_DATE"])
            ),
            "march_2025_train_rows": int(train["RENT_APPROVAL_DATE"].eq("2025-03").sum()),
            "march_2025_test_rows": int(test["RENT_APPROVAL_DATE"].eq("2025-03").sum()),
            "train_flat_type_labels": sorted(train["FLAT_TYPE"].astype(str).unique()),
            "test_flat_type_labels": sorted(test["FLAT_TYPE"].astype(str).unique()),
            "constant_columns": {
                "FURNISHED": sorted(train["FURNISHED"].astype(str).unique()),
                "FEE": sorted(float(value) for value in train["FEE"].unique()),
            },
            "train_exact_duplicates": int(train.duplicated().sum()),
            "test_exact_duplicates": int(test.duplicated().sum()),
            "repeated_predictor_groups": int((predictor_group["size"] > 1).sum()),
            "repeated_groups_with_different_rents": int(
                (predictor_group["nunique"] > 1).sum()
            ),
            "rows_in_conflicting_repeated_groups": int(
                predictor_group.loc[predictor_group["nunique"] > 1, "size"].sum()
            ),
            "invalid_lease_date_rows": int(invalid_lease.sum()),
            "invalid_lease_row_ids": train.index[invalid_lease].astype(int).tolist(),
        },
        "target": {
            "minimum": int(train["MONTHLY_RENT"].min()),
            "maximum": int(train["MONTHLY_RENT"].max()),
            "mean": float(train["MONTHLY_RENT"].mean()),
            "median": float(train["MONTHLY_RENT"].median()),
            "standard_deviation": float(train["MONTHLY_RENT"].std()),
            "low_rent_rows_below_1000": int(train["MONTHLY_RENT"].lt(1_000).sum()),
        },
        "hdb_join_coverage": {
            "train_raw": _join_coverage(train, hdb, normalize=False),
            "train_normalized": _join_coverage(train, hdb, normalize=True),
            "test_raw": _join_coverage(test, hdb, normalize=False),
            "test_normalized": _join_coverage(test, hdb, normalize=True),
        },
        "auxiliary_validation": auxiliary_checks,
    }


def prepare_core(frame: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
    """Clean a transaction table and add structural and calendar features."""

    validate_core_schema(frame, is_train=is_train)
    result = frame.copy()
    result.insert(0, "ROW_ID", np.arange(len(result), dtype=np.int64))

    for column in TEXT_COLUMNS:
        result[column] = normalize_text(result[column])
    result["FLAT_TYPE"] = canonicalize_flat_type(result["FLAT_TYPE"])
    result["RENT_APPROVAL_DATE"] = pd.to_datetime(
        result["RENT_APPROVAL_DATE"], format="%Y-%m"
    )

    result["APPROVAL_YEAR"] = result["RENT_APPROVAL_DATE"].dt.year.astype("int16")
    result["APPROVAL_MONTH"] = result["RENT_APPROVAL_DATE"].dt.month.astype("int8")
    result["APPROVAL_QUARTER"] = result["RENT_APPROVAL_DATE"].dt.quarter.astype("int8")
    result["MONTH_INDEX"] = (
        (result["APPROVAL_YEAR"] - BASE_MONTH.year) * 12
        + result["APPROVAL_MONTH"]
        - BASE_MONTH.month
    ).astype("int16")
    angle = 2 * np.pi * (result["APPROVAL_MONTH"] - 1) / 12
    result["MONTH_SIN"] = np.sin(angle).astype("float32")
    result["MONTH_COS"] = np.cos(angle).astype("float32")

    room_map = {
        "1-room": 1,
        "2-room": 2,
        "3-room": 3,
        "4-room": 4,
        "5-room": 5,
        "executive": 6,
    }
    result["FLAT_TYPE_ORDINAL"] = result["FLAT_TYPE"].map(room_map).astype("int8")
    result["FLOOR_AREA_PER_ROOM_PROXY"] = (
        result["FLOOR_AREA_SQM"] / result["FLAT_TYPE_ORDINAL"]
    ).astype("float32")

    raw_age = result["APPROVAL_YEAR"] - result["LEASE_COMMENCE_DATE"]
    invalid_age = raw_age.lt(0)
    result["INVALID_LEASE_DATE_FLAG"] = invalid_age.astype("int8")
    result["FLAT_AGE_YEARS"] = raw_age.astype("float32").mask(invalid_age)
    result["REMAINING_LEASE_PROXY_YEARS"] = (
        99 - result["FLAT_AGE_YEARS"]
    ).clip(lower=0).astype("float32")
    result["ADDRESS_KEY"] = result[ADDRESS_COLUMNS].astype("string").agg("|".join, axis=1)

    result = result.drop(columns=["FURNISHED", "FEE"])
    return result


def prepare_hdb_blocks(hdb: pd.DataFrame) -> pd.DataFrame:
    """Normalize the HDB block table and enforce a unique address key."""

    required = ADDRESS_COLUMNS + [
        "POSTAL_CODE",
        "LATITUDE",
        "LONGITUDE",
        "MAX_FLOOR",
        "YEAR_COMPLETED",
        "SUBZONE",
        "PLANNING_AREA",
        "REGION",
    ]
    missing = [column for column in required if column not in hdb.columns]
    if missing:
        raise ValueError(f"HDB block file is missing columns: {missing}")

    result = hdb[required].copy()
    for column in ADDRESS_COLUMNS + ["SUBZONE", "PLANNING_AREA", "REGION"]:
        result[column] = normalize_text(result[column])
    result["POSTAL_CODE"] = normalize_text(result["POSTAL_CODE"])
    postal_missing = result["POSTAL_CODE"].isin(["nil", "none", ""])
    result["HDB_POSTAL_CODE_MISSING_FLAG"] = postal_missing.astype("int8")
    result["POSTAL_CODE"] = result["POSTAL_CODE"].mask(postal_missing).str.zfill(6)

    max_floor = pd.to_numeric(result["MAX_FLOOR"], errors="raise")
    completion_year = pd.to_numeric(result["YEAR_COMPLETED"], errors="raise")
    max_floor_missing = max_floor.eq(-1)
    completion_year_missing = completion_year.eq(-1)
    if (max_floor.le(0) & ~max_floor_missing).any():
        raise ValueError("HDB MAX_FLOOR contains an unsupported nonpositive value")
    if (completion_year.ne(-1) & ~completion_year.between(1900, 2100)).any():
        raise ValueError("HDB YEAR_COMPLETED contains an unsupported value")
    result["HDB_MAX_FLOOR_MISSING_FLAG"] = max_floor_missing.astype("int8")
    result["HDB_YEAR_COMPLETED_MISSING_FLAG"] = completion_year_missing.astype("int8")
    result["MAX_FLOOR"] = max_floor.astype("float32").mask(max_floor_missing)
    result["YEAR_COMPLETED"] = completion_year.astype("float32").mask(
        completion_year_missing
    )
    if result.duplicated(ADDRESS_COLUMNS).any():
        raise ValueError("HDB address keys must be unique after normalization")
    if not result["LATITUDE"].between(1.0, 1.6).all():
        raise ValueError("HDB latitude is outside the expected Singapore range")
    if not result["LONGITUDE"].between(103.5, 104.2).all():
        raise ValueError("HDB longitude is outside the expected Singapore range")

    return result.rename(
        columns={
            "POSTAL_CODE": "HDB_POSTAL_CODE",
            "LATITUDE": "HDB_LATITUDE",
            "LONGITUDE": "HDB_LONGITUDE",
            "MAX_FLOOR": "HDB_MAX_FLOOR",
            "YEAR_COMPLETED": "HDB_YEAR_COMPLETED",
            "SUBZONE": "HDB_SUBZONE",
            "PLANNING_AREA": "HDB_PLANNING_AREA",
            "REGION": "HDB_REGION",
        }
    )


def _valid_coordinates(frame: pd.DataFrame) -> np.ndarray:
    coords = frame[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Point-of-interest coordinates contain missing or infinite values")
    return coords


def nearest_distance_and_counts(
    origins: np.ndarray,
    destinations: np.ndarray,
    radii_km: Iterable[float],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Calculate Haversine nearest distances and radius counts."""

    origins = np.asarray(origins, dtype=float)
    destinations = np.asarray(destinations, dtype=float)
    if origins.ndim != 2 or origins.shape[1] != 2:
        raise ValueError("origins must have shape (n, 2)")
    if destinations.ndim != 2 or destinations.shape[1] != 2:
        raise ValueError("destinations must have shape (m, 2)")
    if len(destinations) == 0:
        return (
            np.full(len(origins), np.nan),
            {float(radius): np.zeros(len(origins), dtype=np.int32) for radius in radii_km},
        )

    tree = BallTree(np.radians(destinations), metric="haversine")
    nearest_radians, _ = tree.query(np.radians(origins), k=1)
    nearest_km = nearest_radians[:, 0] * EARTH_RADIUS_KM
    counts = {
        float(radius): tree.query_radius(
            np.radians(origins),
            r=float(radius) / EARTH_RADIUS_KM,
            count_only=True,
        ).astype(np.int32)
        for radius in radii_km
    }
    return nearest_km, counts


def haversine_to_point(origins: np.ndarray, destination: Sequence[float]) -> np.ndarray:
    """Calculate Haversine distance from each origin to one destination."""

    origins_rad = np.radians(np.asarray(origins, dtype=float))
    destination_rad = np.radians(np.asarray(destination, dtype=float))
    dlat = destination_rad[0] - origins_rad[:, 0]
    dlon = destination_rad[1] - origins_rad[:, 1]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(origins_rad[:, 0])
        * np.cos(destination_rad[0])
        * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def prepare_mrt_stations(mrt: pd.DataFrame) -> pd.DataFrame:
    """Collapse line-specific MRT rows into physical stations."""

    required = ["CODE", "NAME", "STATUS", "LATITUDE", "LONGITUDE"]
    missing = [column for column in required if column not in mrt.columns]
    if missing:
        raise ValueError(f"MRT file is missing columns: {missing}")
    result = mrt[required].copy()
    result["NAME"] = normalize_text(result["NAME"])
    result["STATUS"] = normalize_text(result["STATUS"])
    result["CODE"] = normalize_text(result["CODE"])
    if not set(result["STATUS"].unique()).issubset({"open", "planned"}):
        raise ValueError("MRT STATUS contains values other than open/planned")

    physical = (
        result.groupby(["NAME", "STATUS"], as_index=False)
        .agg(
            LATITUDE=("LATITUDE", "mean"),
            LONGITUDE=("LONGITUDE", "mean"),
            CODE_COUNT=("CODE", "nunique"),
        )
    )
    return physical


def build_block_geospatial_features(
    hdb: pd.DataFrame,
    mrt: pd.DataFrame,
    schools: pd.DataFrame,
    malls: pd.DataFrame,
) -> pd.DataFrame:
    """Create static location features once for each HDB block."""

    blocks = prepare_hdb_blocks(hdb)
    origins = blocks[["HDB_LATITUDE", "HDB_LONGITUDE"]].to_numpy(dtype=float)
    stations = prepare_mrt_stations(mrt)
    open_stations = stations.loc[stations["STATUS"].eq("open")]
    planned_stations = stations.loc[stations["STATUS"].eq("planned")]
    interchanges = open_stations.loc[open_stations["CODE_COUNT"].ge(2)]

    raffles = stations.loc[stations["NAME"].eq("raffles place mrt station")]
    if len(raffles) != 1:
        raise ValueError("Expected one physical Raffles Place MRT station")
    blocks["DISTANCE_TO_RAFFLES_PLACE_KM"] = haversine_to_point(
        origins, raffles[["LATITUDE", "LONGITUDE"]].iloc[0].to_numpy()
    ).astype("float32")

    nearest, counts = nearest_distance_and_counts(
        origins, _valid_coordinates(open_stations), [0.5, 1.0, 2.0]
    )
    blocks["NEAREST_OPEN_MRT_KM"] = nearest.astype("float32")
    blocks["OPEN_MRT_COUNT_500M"] = counts[0.5].astype("int16")
    blocks["OPEN_MRT_COUNT_1KM"] = counts[1.0].astype("int16")
    blocks["OPEN_MRT_COUNT_2KM"] = counts[2.0].astype("int16")

    nearest, _ = nearest_distance_and_counts(
        origins, _valid_coordinates(interchanges), []
    )
    blocks["NEAREST_OPEN_INTERCHANGE_KM"] = nearest.astype("float32")

    nearest, counts = nearest_distance_and_counts(
        origins, _valid_coordinates(planned_stations), [1.0, 2.0]
    )
    blocks["NEAREST_PLANNED_MRT_KM"] = nearest.astype("float32")
    blocks["PLANNED_MRT_COUNT_1KM"] = counts[1.0].astype("int16")
    blocks["PLANNED_MRT_COUNT_2KM"] = counts[2.0].astype("int16")

    school_points = schools[["NAME", "MAINLEVEL_CODE", "LATITUDE", "LONGITUDE"]].copy()
    school_points["NAME"] = normalize_text(school_points["NAME"])
    school_points["MAINLEVEL_CODE"] = normalize_text(school_points["MAINLEVEL_CODE"])
    school_points = school_points.drop_duplicates("NAME")
    primary_mask = school_points["MAINLEVEL_CODE"].str.startswith(
        "primary", na=False
    ) | school_points["MAINLEVEL_CODE"].str.contains(r"\(p1", regex=True, na=False)
    primary_points = school_points.loc[primary_mask]

    nearest, counts = nearest_distance_and_counts(
        origins, _valid_coordinates(school_points), [1.0, 2.0]
    )
    blocks["NEAREST_SCHOOL_KM"] = nearest.astype("float32")
    blocks["SCHOOL_COUNT_1KM"] = counts[1.0].astype("int16")
    blocks["SCHOOL_COUNT_2KM"] = counts[2.0].astype("int16")

    nearest, counts = nearest_distance_and_counts(
        origins, _valid_coordinates(primary_points), [1.0, 2.0]
    )
    blocks["NEAREST_PRIMARY_SCHOOL_KM"] = nearest.astype("float32")
    blocks["PRIMARY_SCHOOL_COUNT_1KM"] = counts[1.0].astype("int16")
    blocks["PRIMARY_SCHOOL_COUNT_2KM"] = counts[2.0].astype("int16")

    mall_points = malls[["NAME", "LATITUDE", "LONGITUDE"]].drop_duplicates("NAME")
    nearest, counts = nearest_distance_and_counts(
        origins, _valid_coordinates(mall_points), [1.0, 2.0]
    )
    blocks["NEAREST_MALL_KM"] = nearest.astype("float32")
    blocks["MALL_COUNT_1KM"] = counts[1.0].astype("int16")
    blocks["MALL_COUNT_2KM"] = counts[2.0].astype("int16")
    return blocks


def enrich_with_block_features(
    transactions: pd.DataFrame, block_features: pd.DataFrame
) -> pd.DataFrame:
    """Attach HDB and geospatial features with a validated many-to-one join."""

    before = len(transactions)
    result = transactions.merge(
        block_features,
        on=ADDRESS_COLUMNS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if len(result) != before:
        raise AssertionError("HDB join changed the transaction row count")
    if not result["_merge"].eq("both").all():
        unmatched = result.loc[result["_merge"].ne("both"), ADDRESS_COLUMNS]
        raise ValueError(f"Unmatched HDB addresses remain: {len(unmatched)} rows")
    result = result.drop(columns="_merge")
    result["HDB_BLOCK_AGE_YEARS"] = (
        result["APPROVAL_YEAR"] - result["HDB_YEAR_COMPLETED"]
    ).astype("float32")
    return result


def prepare_coe_monthly(coe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate COE rounds to monthly Category A/B features."""

    required = [
        "YEAR",
        "CATEGORY",
        "MONTH",
        "ROUND",
        "QUOTA_PREMIUM",
        "QUOTA",
        "BIDS_RECEIVED",
    ]
    missing = [column for column in required if column not in coe.columns]
    if missing:
        raise ValueError(f"COE file is missing columns: {missing}")
    result = coe[required].copy()
    result["CATEGORY"] = normalize_text(result["CATEGORY"])
    month_number = pd.to_datetime(
        normalize_text(result["MONTH"]), format="%b", errors="coerce"
    ).dt.month
    if month_number.isna().any():
        raise ValueError("COE MONTH contains an invalid month abbreviation")
    result["RENT_APPROVAL_DATE"] = pd.to_datetime(
        {"year": result["YEAR"], "month": month_number, "day": 1}
    )
    for column in ["QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]:
        result[column] = parse_numeric_text(result[column]).astype(float)
    result = result.loc[result["CATEGORY"].isin(["a", "b"])]

    grouped = result.groupby(["RENT_APPROVAL_DATE", "CATEGORY"]).agg(
        PREMIUM=("QUOTA_PREMIUM", "mean"),
        QUOTA=("QUOTA", "sum"),
        BIDS=("BIDS_RECEIVED", "sum"),
    )
    grouped["DEMAND_RATIO"] = grouped["BIDS"] / grouped["QUOTA"]

    pieces = []
    for category in ["a", "b"]:
        category_frame = grouped.xs(category, level="CATEGORY")[[
            "PREMIUM",
            "DEMAND_RATIO",
        ]].copy()
        category_frame.columns = [
            f"COE_{category.upper()}_{column}_CURRENT" for column in category_frame.columns
        ]
        pieces.append(category_frame)
    monthly = pd.concat(pieces, axis=1).sort_index()
    for column in list(monthly.columns):
        stem = column.removesuffix("_CURRENT")
        monthly[f"{stem}_LAG1"] = monthly[column].shift(1)
        monthly[f"{stem}_LAGGED_3M_MEAN"] = monthly[column].shift(1).rolling(
            3, min_periods=3
        ).mean()
    monthly["COE_CURRENT_AVAILABLE"] = monthly.filter(like="_CURRENT").notna().all(axis=1).astype("int8")
    monthly["COE_LAG1_AVAILABLE"] = monthly.filter(like="_LAG1").notna().all(axis=1).astype("int8")
    monthly["COE_LAGGED_3M_AVAILABLE"] = monthly.filter(like="_LAGGED_3M_MEAN").notna().all(axis=1).astype("int8")
    validate_monthly_timing(monthly, prefix="COE", first_current_month_missing=False)
    return monthly.reset_index()


def prepare_stock_monthly(stocks: pd.DataFrame) -> pd.DataFrame:
    """Build stable Singapore market and bank composites from adjusted closes."""

    required = ["NAME", "SYMBOL", "DATE", "ADJUSTED_CLOSE"]
    missing = [column for column in required if column not in stocks.columns]
    if missing:
        raise ValueError(f"Stock file is missing columns: {missing}")
    result = stocks[required].copy()
    result["SYMBOL"] = result["SYMBOL"].astype("string").str.strip().str.upper()
    result["DATE"] = pd.to_datetime(result["DATE"], errors="raise")
    result["ADJUSTED_CLOSE"] = pd.to_numeric(result["ADJUSTED_CLOSE"], errors="coerce")
    result["RENT_APPROVAL_DATE"] = result["DATE"].dt.to_period("M").dt.to_timestamp()

    first_month = result["RENT_APPROVAL_DATE"].min()
    symbol_first_month = result.groupby("SYMBOL")["RENT_APPROVAL_DATE"].min()
    universe = symbol_first_month.index[
        symbol_first_month.eq(first_month) & symbol_first_month.index.str.endswith(".SI")
    ]
    if len(universe) < 3:
        raise ValueError("Could not establish a stable Singapore-listed stock universe")
    result = result.loc[result["SYMBOL"].isin(universe)]

    monthly_close = (
        result.sort_values("DATE")
        .groupby(["RENT_APPROVAL_DATE", "SYMBOL"])["ADJUSTED_CLOSE"]
        .last()
        .unstack("SYMBOL")
        .sort_index()
    )
    expected_months = pd.date_range(
        monthly_close.index.min(), monthly_close.index.max(), freq="MS"
    )
    if not monthly_close.index.equals(expected_months):
        raise ValueError("The fixed stock universe does not cover every calendar month")
    if monthly_close.isna().any().any():
        missing_symbols = monthly_close.columns[monthly_close.isna().any()].tolist()
        raise ValueError(
            f"The fixed stock universe has missing monthly closes: {missing_symbols}"
        )
    returns = monthly_close.pct_change(fill_method=None)
    bank_symbols = ["D05.SI", "O39.SI", "U11.SI"]
    missing_banks = sorted(set(bank_symbols) - set(returns.columns))
    if missing_banks:
        raise ValueError(f"Missing bank symbols for the stock composite: {missing_banks}")

    monthly = pd.DataFrame(index=returns.index)
    monthly["STOCK_SG_MARKET_RETURN_CURRENT"] = returns.median(axis=1, skipna=True)
    monthly["STOCK_SG_POSITIVE_BREADTH_CURRENT"] = returns.gt(0).sum(axis=1) / returns.notna().sum(axis=1)
    monthly.loc[returns.notna().sum(axis=1).eq(0), "STOCK_SG_POSITIVE_BREADTH_CURRENT"] = np.nan
    monthly["STOCK_BANK_RETURN_CURRENT"] = returns[bank_symbols].median(axis=1, skipna=True)
    for column in list(monthly.columns):
        stem = column.removesuffix("_CURRENT")
        monthly[f"{stem}_LAG1"] = monthly[column].shift(1)
        monthly[f"{stem}_LAGGED_3M_MEAN"] = monthly[column].shift(1).rolling(
            3, min_periods=3
        ).mean()
    monthly["STOCK_CURRENT_AVAILABLE"] = monthly.filter(like="_CURRENT").notna().all(axis=1).astype("int8")
    monthly["STOCK_LAG1_AVAILABLE"] = monthly.filter(like="_LAG1").notna().all(axis=1).astype("int8")
    monthly["STOCK_LAGGED_3M_AVAILABLE"] = monthly.filter(like="_LAGGED_3M_MEAN").notna().all(axis=1).astype("int8")
    validate_monthly_timing(monthly, prefix="STOCK", first_current_month_missing=True)
    monthly.attrs["universe_symbols"] = list(universe)
    return monthly.reset_index()


def validate_monthly_timing(
    monthly: pd.DataFrame,
    *,
    prefix: str,
    first_current_month_missing: bool,
) -> None:
    """Enforce complete monthly keys, exact lags, rolling windows, and flags."""

    if not isinstance(monthly.index, pd.DatetimeIndex) or monthly.index.has_duplicates:
        raise ValueError(f"{prefix} monthly features require unique DatetimeIndex keys")
    expected_months = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    if not monthly.index.equals(expected_months):
        raise ValueError(f"{prefix} monthly features contain an internal month gap")

    current_columns = [
        column
        for column in monthly.columns
        if column.startswith(f"{prefix}_") and column.endswith("_CURRENT")
    ]
    if not current_columns:
        raise ValueError(f"{prefix} monthly features have no current columns")
    current_missing = monthly[current_columns].isna()
    if first_current_month_missing:
        expected_current_missing = pd.Series(False, index=monthly.index)
        expected_current_missing.iloc[0] = True
        if not current_missing.all(axis=1).equals(expected_current_missing):
            raise ValueError(
                f"{prefix} current features must be missing only in the first month"
            )
    elif current_missing.any().any():
        raise ValueError(f"{prefix} current features must cover every month")

    for current_column in current_columns:
        stem = current_column.removesuffix("_CURRENT")
        lag_column = f"{stem}_LAG1"
        rolling_column = f"{stem}_LAGGED_3M_MEAN"
        pd.testing.assert_series_equal(
            monthly[lag_column],
            monthly[current_column].shift(1),
            check_names=False,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            monthly[rolling_column],
            monthly[current_column].shift(1).rolling(3, min_periods=3).mean(),
            check_names=False,
            check_dtype=False,
        )

    availability_groups = {
        f"{prefix}_CURRENT_AVAILABLE": current_columns,
        f"{prefix}_LAG1_AVAILABLE": [
            column
            for column in monthly.columns
            if column.startswith(f"{prefix}_") and column.endswith("_LAG1")
        ],
        f"{prefix}_LAGGED_3M_AVAILABLE": [
            column
            for column in monthly.columns
            if column.startswith(f"{prefix}_")
            and column.endswith("_LAGGED_3M_MEAN")
        ],
    }
    for flag, feature_columns in availability_groups.items():
        if not feature_columns:
            raise ValueError(f"No feature columns found for availability flag {flag}")
        complete = monthly[feature_columns].notna().all(axis=1)
        any_present = monthly[feature_columns].notna().any(axis=1)
        if not complete.equals(any_present):
            raise ValueError(f"{prefix} contains partial missingness within {flag}")
        expected_flag = complete.astype("int8")
        if not monthly[flag].equals(expected_flag):
            raise ValueError(f"{flag} disagrees with its feature columns")


def validate_macro_availability_flags(frame: pd.DataFrame) -> None:
    """Check that row-level macro missingness exactly matches availability flags."""

    for prefix in ["COE", "STOCK"]:
        groups = {
            f"{prefix}_CURRENT_AVAILABLE": [
                column
                for column in frame.columns
                if column.startswith(f"{prefix}_") and column.endswith("_CURRENT")
            ],
            f"{prefix}_LAG1_AVAILABLE": [
                column
                for column in frame.columns
                if column.startswith(f"{prefix}_") and column.endswith("_LAG1")
            ],
            f"{prefix}_LAGGED_3M_AVAILABLE": [
                column
                for column in frame.columns
                if column.startswith(f"{prefix}_")
                and column.endswith("_LAGGED_3M_MEAN")
            ],
        }
        for flag, feature_columns in groups.items():
            if frame[flag].isna().any():
                raise ValueError(f"{flag} cannot be missing")
            complete = frame[feature_columns].notna().all(axis=1)
            any_present = frame[feature_columns].notna().any(axis=1)
            if not complete.equals(any_present):
                raise ValueError(f"Partial macro missingness found for {flag}")
            if not frame[flag].astype(bool).equals(complete):
                raise ValueError(f"{flag} disagrees with row-level feature availability")


def add_macro_features(
    transactions: pd.DataFrame,
    coe_monthly: pd.DataFrame,
    stock_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Join monthly macro features without using the rental target."""

    before = len(transactions)
    result = transactions.merge(
        coe_monthly,
        on="RENT_APPROVAL_DATE",
        how="left",
        validate="many_to_one",
    ).merge(
        stock_monthly,
        on="RENT_APPROVAL_DATE",
        how="left",
        validate="many_to_one",
    )
    if len(result) != before:
        raise AssertionError("Macro joins changed the transaction row count")
    current_coe = [column for column in result if column.startswith("COE_") and column.endswith("_CURRENT")]
    if result[current_coe].isna().any().any():
        raise ValueError("COE contemporaneous coverage is incomplete")
    validate_macro_availability_flags(result)
    return result


@dataclass
class SmoothedGroupTargetEncoder:
    """Smoothed group mean encoder with leave-one-out training values.

    ``fit_transform`` excludes each training row's own target.  ``transform``
    uses mappings fitted only from the caller's training partition and falls
    back to the fitted global mean for unseen groups.
    """

    group_cols: Sequence[str]
    alpha: float = 20.0
    output_name: str = "SMOOTHED_TARGET_MEAN"

    def __post_init__(self) -> None:
        self.group_cols = list(self.group_cols)
        if not self.group_cols:
            raise ValueError("group_cols must contain at least one column")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        self.global_mean_: float | None = None
        self.mapping_: pd.DataFrame | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "SmoothedGroupTargetEncoder":
        self._validate_inputs(frame, target)
        work = frame[self.group_cols].copy()
        work["__target"] = np.asarray(target, dtype=float)
        self.global_mean_ = float(work["__target"].mean())
        stats = work.groupby(self.group_cols, dropna=False)["__target"].agg(["sum", "count"])
        stats[self.output_name] = (
            stats["sum"] + self.alpha * self.global_mean_
        ) / (stats["count"] + self.alpha)
        self.mapping_ = stats[[self.output_name]].reset_index()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        if self.mapping_ is None or self.global_mean_ is None:
            raise RuntimeError("The target encoder must be fitted before transform")
        missing = [column for column in self.group_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing grouping columns: {missing}")
        keys = frame[self.group_cols].copy()
        keys["__row_order"] = np.arange(len(keys))
        encoded = keys.merge(
            self.mapping_, on=self.group_cols, how="left", validate="many_to_one"
        ).sort_values("__row_order")[self.output_name]
        encoded = encoded.fillna(self.global_mean_)
        return pd.Series(encoded.to_numpy(dtype=float), index=frame.index, name=self.output_name)

    def fit_transform(self, frame: pd.DataFrame, target: pd.Series) -> pd.Series:
        """Return leave-one-out training values and store the full fold mapping."""

        self._validate_inputs(frame, target)
        y = pd.Series(np.asarray(target, dtype=float), index=frame.index)
        work = frame[self.group_cols].copy()
        work["__target"] = y.to_numpy()
        grouped_target = work.groupby(self.group_cols, dropna=False)["__target"]
        group_sum = grouped_target.transform("sum")
        group_count = grouped_target.transform("count")
        if len(work) <= 1:
            raise ValueError("At least two rows are required for leave-one-out encoding")
        global_loo = (y.sum() - y) / (len(y) - 1)
        encoded = (
            group_sum - y + self.alpha * global_loo
        ) / (group_count - 1 + self.alpha)
        self.fit(frame, y)
        encoded.name = self.output_name
        return encoded.astype(float)

    def _validate_inputs(self, frame: pd.DataFrame, target: pd.Series) -> None:
        missing = [column for column in self.group_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing grouping columns: {missing}")
        if len(frame) != len(target):
            raise ValueError("frame and target must have the same number of rows")
        if pd.isna(target).any():
            raise ValueError("target cannot contain missing values")


def build_feature_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe exact derivation, source, timing, and expected missingness."""

    definitions = {
        "ROW_ID": ("identifier", "key", "observed", "generated", "Zero-based source row position; unchanged through every join.", "none"),
        "MONTHLY_RENT": ("target", "target", "training only", "train.csv", "Observed monthly rent in SGD; copied without clipping or transformation.", "none"),
        "RENT_APPROVAL_DATE": ("feature", "core temporal", "observed", "train/test", "Source YYYY-MM approval value converted to the first calendar day of that month.", "none"),
        "TOWN": ("feature", "core", "observed", "train/test", "Source town after lowercase conversion, trimming, and whitespace collapse.", "none"),
        "BLOCK": ("feature", "core", "observed", "train/test", "Source HDB block identifier after lowercase conversion, trimming, and whitespace collapse.", "none"),
        "STREET": ("feature", "core", "observed", "train/test", "Source street after lowercase conversion, trimming, and whitespace collapse.", "none"),
        "FLAT_TYPE": ("feature", "core", "observed", "train/test", "Canonical flat label: 1-room through 5-room or executive.", "none"),
        "FLAT_MODEL": ("feature", "core", "observed", "train/test", "Source flat model after lowercase conversion, trimming, and whitespace collapse.", "none"),
        "FLOOR_AREA_SQM": ("feature", "core", "observed", "train/test", "Source floor area in square metres; must be positive.", "none"),
        "LEASE_COMMENCE_DATE": ("feature", "core", "observed", "train/test", "Source lease commencement year retained unchanged, including the flagged inconsistent row.", "none"),
        "APPROVAL_YEAR": ("feature", "core temporal", "derived", "train/test", "Calendar year extracted from RENT_APPROVAL_DATE.", "none"),
        "APPROVAL_MONTH": ("feature", "core temporal", "derived", "train/test", "Calendar month number from RENT_APPROVAL_DATE, from 1 to 12.", "none"),
        "APPROVAL_QUARTER": ("feature", "core temporal", "derived", "train/test", "Calendar quarter from RENT_APPROVAL_DATE, from 1 to 4.", "none"),
        "MONTH_INDEX": ("feature", "core temporal", "derived", "train/test", "Whole months since January 2021: 12*(year-2021)+(month-1).", "none"),
        "MONTH_SIN": ("feature", "core temporal", "derived", "train/test", "Annual cyclic encoding sin(2*pi*(approval_month-1)/12).", "none"),
        "MONTH_COS": ("feature", "core temporal", "derived", "train/test", "Annual cyclic encoding cos(2*pi*(approval_month-1)/12).", "none"),
        "FLAT_TYPE_ORDINAL": ("feature", "core structural", "derived", "train/test", "Flat-type proxy mapping 1-room..5-room to 1..5 and executive to 6.", "none"),
        "FLOOR_AREA_PER_ROOM_PROXY": ("feature", "core structural", "derived", "train/test", "FLOOR_AREA_SQM divided by FLAT_TYPE_ORDINAL; executive uses divisor 6.", "none"),
        "INVALID_LEASE_DATE_FLAG": ("feature", "core structural", "derived", "train/test", "1 when LEASE_COMMENCE_DATE exceeds APPROVAL_YEAR; otherwise 0.", "none"),
        "FLAT_AGE_YEARS": ("feature", "core structural", "derived", "train/test", "APPROVAL_YEAR minus LEASE_COMMENCE_DATE; missing when INVALID_LEASE_DATE_FLAG=1.", "Only rows where INVALID_LEASE_DATE_FLAG=1."),
        "REMAINING_LEASE_PROXY_YEARS": ("feature", "core structural", "derived", "train/test", "max(0, 99-FLAT_AGE_YEARS); a proxy rather than an observed lease balance.", "Only rows where INVALID_LEASE_DATE_FLAG=1."),
        "ADDRESS_KEY": ("feature", "core structural", "derived", "train/test", "Normalized TOWN, BLOCK, and STREET joined with vertical bars.", "none"),
        "HDB_POSTAL_CODE": ("feature", "HDB block", "static snapshot", "sg-hdb-block.csv", "Six-character postal code joined on normalized town/block/street; source NIL becomes missing.", "Rows where HDB_POSTAL_CODE_MISSING_FLAG=1."),
        "HDB_LATITUDE": ("feature", "HDB block", "static snapshot", "sg-hdb-block.csv", "Supplied HDB-block latitude in decimal degrees.", "none"),
        "HDB_LONGITUDE": ("feature", "HDB block", "static snapshot", "sg-hdb-block.csv", "Supplied HDB-block longitude in decimal degrees.", "none"),
        "HDB_MAX_FLOOR": ("feature", "HDB block", "static snapshot", "sg-hdb-block.csv", "Supplied maximum block floor; source sentinel -1 becomes missing.", "Rows where HDB_MAX_FLOOR_MISSING_FLAG=1."),
        "HDB_YEAR_COMPLETED": ("feature", "HDB block", "static snapshot", "sg-hdb-block.csv", "Supplied block completion year; source sentinel -1 becomes missing.", "Rows where HDB_YEAR_COMPLETED_MISSING_FLAG=1."),
        "HDB_SUBZONE": ("feature", "HDB block", "static snapshot", "sg-hdb-block.csv", "Supplied subzone after lowercase conversion and whitespace normalization.", "none"),
        "HDB_PLANNING_AREA": ("feature", "HDB block", "static snapshot", "sg-hdb-block.csv", "Supplied planning area after lowercase conversion and whitespace normalization.", "none"),
        "HDB_REGION": ("feature", "HDB block", "static snapshot", "sg-hdb-block.csv", "Supplied region after lowercase conversion and whitespace normalization.", "none"),
        "HDB_POSTAL_CODE_MISSING_FLAG": ("feature", "HDB block", "static diagnostic", "sg-hdb-block.csv", "1 when the source postal code is NIL, none, or blank; otherwise 0.", "none"),
        "HDB_MAX_FLOOR_MISSING_FLAG": ("feature", "HDB block", "static diagnostic", "sg-hdb-block.csv", "1 when source MAX_FLOOR uses sentinel -1; otherwise 0.", "none"),
        "HDB_YEAR_COMPLETED_MISSING_FLAG": ("feature", "HDB block", "static diagnostic", "sg-hdb-block.csv", "1 when source YEAR_COMPLETED uses sentinel -1; otherwise 0.", "none"),
        "HDB_BLOCK_AGE_YEARS": ("feature", "HDB block", "derived", "sg-hdb-block.csv + train/test", "APPROVAL_YEAR minus HDB_YEAR_COMPLETED.", "Rows where HDB_YEAR_COMPLETED_MISSING_FLAG=1."),
        "DISTANCE_TO_RAFFLES_PLACE_KM": ("feature", "geospatial", "static snapshot", "sg-hdb-block.csv + sg-mrt-stations.csv", "Haversine distance in km from the block to the supplied Raffles Place MRT coordinates.", "none"),
        "NEAREST_OPEN_MRT_KM": ("feature", "geospatial", "static snapshot", "sg-mrt-stations.csv", "Haversine distance in km to the nearest physical station with STATUS=open.", "none"),
        "OPEN_MRT_COUNT_500M": ("feature", "geospatial", "static snapshot", "sg-mrt-stations.csv", "Number of distinct physical open MRT stations within 0.5 km Haversine distance.", "none"),
        "OPEN_MRT_COUNT_1KM": ("feature", "geospatial", "static snapshot", "sg-mrt-stations.csv", "Number of distinct physical open MRT stations within 1 km Haversine distance.", "none"),
        "OPEN_MRT_COUNT_2KM": ("feature", "geospatial", "static snapshot", "sg-mrt-stations.csv", "Number of distinct physical open MRT stations within 2 km Haversine distance.", "none"),
        "NEAREST_OPEN_INTERCHANGE_KM": ("feature", "geospatial", "static snapshot", "sg-mrt-stations.csv", "Haversine distance in km to the nearest open physical station represented by at least two open station codes.", "none"),
        "NEAREST_PLANNED_MRT_KM": ("feature", "geospatial", "static snapshot", "sg-mrt-stations.csv", "Haversine distance in km to the nearest physical station entry with STATUS=planned; planned connectivity at open sites remains separate from open features.", "none"),
        "PLANNED_MRT_COUNT_1KM": ("feature", "geospatial", "static snapshot", "sg-mrt-stations.csv", "Number of distinct physical planned-status MRT station entries within 1 km.", "none"),
        "PLANNED_MRT_COUNT_2KM": ("feature", "geospatial", "static snapshot", "sg-mrt-stations.csv", "Number of distinct physical planned-status MRT station entries within 2 km.", "none"),
        "NEAREST_SCHOOL_KM": ("feature", "geospatial", "static snapshot", "sg-schools.csv", "Haversine distance in km to the nearest distinct supplied school of any level.", "none"),
        "SCHOOL_COUNT_1KM": ("feature", "geospatial", "static snapshot", "sg-schools.csv", "Number of distinct supplied schools of any level within 1 km.", "none"),
        "SCHOOL_COUNT_2KM": ("feature", "geospatial", "static snapshot", "sg-schools.csv", "Number of distinct supplied schools of any level within 2 km.", "none"),
        "NEAREST_PRIMARY_SCHOOL_KM": ("feature", "geospatial", "static snapshot", "sg-schools.csv", "Haversine distance in km to the nearest school whose MAINLEVEL_CODE starts with primary or includes P1.", "none"),
        "PRIMARY_SCHOOL_COUNT_1KM": ("feature", "geospatial", "static snapshot", "sg-schools.csv", "Number of primary-inclusive supplied schools within 1 km.", "none"),
        "PRIMARY_SCHOOL_COUNT_2KM": ("feature", "geospatial", "static snapshot", "sg-schools.csv", "Number of primary-inclusive supplied schools within 2 km.", "none"),
        "NEAREST_MALL_KM": ("feature", "geospatial", "static snapshot", "sg-shopping-malls.csv", "Haversine distance in km to the nearest distinct supplied shopping mall.", "none"),
        "MALL_COUNT_1KM": ("feature", "geospatial", "static snapshot", "sg-shopping-malls.csv", "Number of distinct supplied shopping malls within 1 km.", "none"),
        "MALL_COUNT_2KM": ("feature", "geospatial", "static snapshot", "sg-shopping-malls.csv", "Number of distinct supplied shopping malls within 2 km.", "none"),
    }
    records: list[dict[str, str]] = []
    for column in frame.columns:
        metadata = definitions.get(column)
        if metadata is None and column.startswith("COE_"):
            role, group, source = "feature", "macro COE", "sg-coe-prices.csv"
            if column.endswith("_AVAILABLE"):
                availability = "diagnostic flag"
                if "CURRENT" in column:
                    window = "all four current Category A/B premium and demand-ratio features"
                elif "LAG1" in column:
                    window = "all four one-month-lagged COE features"
                else:
                    window = "all four prior-three-month COE means"
                definition = f"1 when {window} are present for the row; otherwise 0."
                expected_missingness = "none"
            else:
                category = "A" if "COE_A_" in column else "B"
                measure = (
                    "mean quota premium across rounds"
                    if "PREMIUM" in column
                    else "total bids divided by total quota across rounds"
                )
                if column.endswith("_CURRENT"):
                    availability = "competition-only current month"
                    timing = "for the approval month"
                    expected_missingness = "none"
                elif column.endswith("_LAG1"):
                    availability = "lagged/deployment"
                    timing = "from the previous calendar month"
                    expected_missingness = "Only months where COE_LAG1_AVAILABLE=0."
                else:
                    availability = "lagged/deployment"
                    timing = "mean of the three complete calendar months before approval"
                    expected_missingness = "Only months where COE_LAGGED_3M_AVAILABLE=0."
                definition = f"COE Category {category} {measure} {timing}."
            metadata = (role, group, availability, source, definition, expected_missingness)
        elif metadata is None and column.startswith("STOCK_"):
            role, group, source = "feature", "macro stock", "sg-stock-prices.csv"
            if column.endswith("_AVAILABLE"):
                availability = "diagnostic flag"
                if "CURRENT" in column:
                    window = "all three current-month stock composites"
                elif "LAG1" in column:
                    window = "all three one-month-lagged stock composites"
                else:
                    window = "all three prior-three-month stock means"
                definition = f"1 when {window} are present for the row; otherwise 0."
                expected_missingness = "none"
            else:
                if "SG_MARKET_RETURN" in column:
                    measure = "cross-sectional median adjusted-close return across the fixed .SI universe present in January 2021"
                elif "POSITIVE_BREADTH" in column:
                    measure = "share of the fixed .SI universe with a positive adjusted-close monthly return"
                else:
                    measure = "median adjusted-close monthly return across DBS, OCBC, and UOB"
                if column.endswith("_CURRENT"):
                    availability = "competition-only current month"
                    timing = "for the approval month"
                    expected_missingness = "January 2021 only because December 2020 prices are unavailable."
                elif column.endswith("_LAG1"):
                    availability = "lagged/deployment"
                    timing = "from the previous calendar month"
                    expected_missingness = "Only months where STOCK_LAG1_AVAILABLE=0."
                else:
                    availability = "lagged/deployment"
                    timing = "mean of the three complete calendar months before approval"
                    expected_missingness = "Only months where STOCK_LAGGED_3M_AVAILABLE=0."
                definition = f"{measure}, {timing}."
            metadata = (role, group, availability, source, definition, expected_missingness)
        if metadata is None:
            raise ValueError(f"Feature dictionary has no exact definition for {column}")
        role, group, availability, source, definition, expected_missingness = metadata
        records.append(
            {
                "feature": column,
                "dtype": str(frame[column].dtype),
                "role": role,
                "group": group,
                "availability": availability,
                "source": source,
                "definition": definition,
                "expected_missingness": expected_missingness,
            }
        )
    return pd.DataFrame(records)


def validate_final_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
) -> dict[str, Any]:
    """Run final acceptance checks and summarize expected missingness."""

    retained_train_mask = ~raw_train.duplicated(keep="first")
    retained_train_row_ids = np.flatnonzero(retained_train_mask.to_numpy())
    duplicate_rows_removed = int((~retained_train_mask).sum())
    if len(train) != len(retained_train_row_ids):
        raise AssertionError("Training output must remove each additional exact duplicate")
    if len(test) != len(raw_test):
        raise AssertionError("The pipeline must preserve every test transaction")
    if not np.array_equal(train["ROW_ID"].to_numpy(), retained_train_row_ids):
        raise AssertionError(
            "Training ROW_ID must preserve source order after exact deduplication"
        )
    if not np.array_equal(test["ROW_ID"].to_numpy(), np.arange(len(test))):
        raise AssertionError("Test ROW_ID no longer matches source order")
    if not train["MONTHLY_RENT"].reset_index(drop=True).equals(
        raw_train.loc[retained_train_mask, "MONTHLY_RENT"].reset_index(drop=True)
    ):
        raise AssertionError("Retained MONTHLY_RENT values changed during preprocessing")
    train_predictors = train.drop(columns="MONTHLY_RENT")
    if list(train_predictors.columns) != list(test.columns):
        raise AssertionError("Train and test predictor schemas differ")
    if {"FURNISHED", "FEE"} & set(train.columns):
        raise AssertionError("Constant columns were not removed")
    if train["ROW_ID"].duplicated().any() or test["ROW_ID"].duplicated().any():
        raise AssertionError("ROW_ID must be unique within each split")

    for name, frame in [("train", train_predictors), ("test", test)]:
        numeric = frame.select_dtypes(include=[np.number])
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise AssertionError(f"{name} contains infinite numeric values")

    flag_controlled_missing = {
        "FLAT_AGE_YEARS": "INVALID_LEASE_DATE_FLAG",
        "REMAINING_LEASE_PROXY_YEARS": "INVALID_LEASE_DATE_FLAG",
        "HDB_POSTAL_CODE": "HDB_POSTAL_CODE_MISSING_FLAG",
        "HDB_MAX_FLOOR": "HDB_MAX_FLOOR_MISSING_FLAG",
        "HDB_YEAR_COMPLETED": "HDB_YEAR_COMPLETED_MISSING_FLAG",
        "HDB_BLOCK_AGE_YEARS": "HDB_YEAR_COMPLETED_MISSING_FLAG",
    }
    macro_groups: dict[str, list[str]] = {}
    for prefix in ["COE", "STOCK"]:
        macro_groups.update(
            {
                f"{prefix}_CURRENT_AVAILABLE": [
                    column
                    for column in train_predictors.columns
                    if column.startswith(f"{prefix}_") and column.endswith("_CURRENT")
                ],
                f"{prefix}_LAG1_AVAILABLE": [
                    column
                    for column in train_predictors.columns
                    if column.startswith(f"{prefix}_") and column.endswith("_LAG1")
                ],
                f"{prefix}_LAGGED_3M_AVAILABLE": [
                    column
                    for column in train_predictors.columns
                    if column.startswith(f"{prefix}_")
                    and column.endswith("_LAGGED_3M_MEAN")
                ],
            }
        )
    allowed_missing = set(flag_controlled_missing)
    allowed_missing.update(column for columns in macro_groups.values() for column in columns)
    unexpected_missing: dict[str, list[str]] = {}
    expected_missing: dict[str, dict[str, int]] = {}
    for name, frame in [("train", train_predictors), ("test", test)]:
        for feature, flag in flag_controlled_missing.items():
            expected_mask = frame[flag].astype(bool)
            if not frame[feature].isna().equals(expected_mask):
                raise AssertionError(
                    f"{name} missingness in {feature} disagrees with {flag}"
                )
        validate_macro_availability_flags(frame)
        for flag, feature_columns in macro_groups.items():
            expected_mask = frame[flag].eq(0)
            for feature in feature_columns:
                if not frame[feature].isna().equals(expected_mask):
                    raise AssertionError(
                        f"{name} missingness in {feature} disagrees with {flag}"
                    )
        missing_counts = frame.isna().sum()
        missing_columns = set(missing_counts[missing_counts.gt(0)].index)
        unexpected = sorted(missing_columns - allowed_missing)
        if unexpected:
            unexpected_missing[name] = unexpected
        expected_missing[name] = {
            column: int(missing_counts[column]) for column in sorted(missing_columns)
        }
    if unexpected_missing:
        raise AssertionError(f"Unexpected missing values: {unexpected_missing}")

    return {
        "raw_train_rows": int(len(raw_train)),
        "train_rows_after_exact_deduplication": int(len(train)),
        "train_exact_duplicate_rows_removed": duplicate_rows_removed,
        "test_rows": int(len(test)),
        "predictor_columns": int(train_predictors.shape[1]),
        "retained_train_target_unchanged": True,
        "train_test_schema_aligned": True,
        "retained_train_row_order_preserved": True,
        "test_row_order_preserved": True,
        "infinite_numeric_values": 0,
        "expected_missing_values": expected_missing,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_pipeline(
    data_dir: str | Path = PROJECT_ROOT / "data",
    processed_dir: str | Path = PROJECT_ROOT / "data" / "processed",
    audit_dir: str | Path = PROJECT_ROOT / "artifacts" / "data_audit",
) -> dict[str, Any]:
    """Run the complete target-free preprocessing workflow and save outputs."""

    data_dir = Path(data_dir)
    processed_dir = Path(processed_dir)
    audit_dir = Path(audit_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_raw_data(data_dir)
    audit = audit_raw_data(datasets)

    train_duplicate_mask = datasets["train"].duplicated(keep="first")
    train = prepare_core(datasets["train"], is_train=True)
    train = train.loc[~train_duplicate_mask.to_numpy()].reset_index(drop=True)
    test = prepare_core(datasets["test"], is_train=False)
    block_features = build_block_geospatial_features(
        datasets["hdb"], datasets["mrt"], datasets["schools"], datasets["malls"]
    )
    train = enrich_with_block_features(train, block_features)
    test = enrich_with_block_features(test, block_features)

    coe_monthly = prepare_coe_monthly(datasets["coe"])
    stock_monthly = prepare_stock_monthly(datasets["stocks"])
    train = add_macro_features(train, coe_monthly, stock_monthly)
    test = add_macro_features(test, coe_monthly, stock_monthly)

    target = train.pop("MONTHLY_RENT")
    test = test[train.columns]
    train["MONTHLY_RENT"] = target
    validation = validate_final_outputs(
        train, test, datasets["train"], datasets["test"]
    )

    train_path = processed_dir / "train_features.parquet"
    test_path = processed_dir / "test_features.parquet"
    audit_path = audit_dir / "audit_summary.json"
    dictionary_path = audit_dir / "feature_dictionary.csv"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    feature_dictionary = build_feature_dictionary(train)
    feature_dictionary.to_csv(dictionary_path, index=False)

    reloaded_train = pd.read_parquet(train_path)
    reloaded_test = pd.read_parquet(test_path)
    pd.testing.assert_frame_equal(
        reloaded_train,
        train,
        check_dtype=True,
        check_exact=True,
        check_categorical=True,
    )
    pd.testing.assert_frame_equal(
        reloaded_test,
        test,
        check_dtype=True,
        check_exact=True,
        check_categorical=True,
    )

    audit["feature_engineering"] = {
        "hdb_blocks_enriched": int(len(block_features)),
        "stock_universe_size": int(
            datasets["stocks"]
            .assign(
                SYMBOL=lambda frame: frame["SYMBOL"].astype("string").str.upper(),
                MONTH=lambda frame: pd.to_datetime(frame["DATE"]).dt.to_period("M"),
            )
            .groupby("SYMBOL")["MONTH"]
            .min()
            .loc[lambda series: series.eq(series.min()) & series.index.str.endswith(".SI")]
            .size
        ),
        "coe_months": int(len(coe_monthly)),
        "stock_months": int(len(stock_monthly)),
        "target_derived_features_saved": False,
        "parquet_round_trip_exact": True,
    }
    audit["cleaning"] = {
        "raw_training_rows": int(len(datasets["train"])),
        "exact_duplicate_training_rows_removed": int(train_duplicate_mask.sum()),
        "training_rows_after_exact_deduplication": int(len(train)),
        "deduplication_policy": "Keep the first exact training row in original order",
        "test_rows_removed": 0,
    }
    audit["final_validation"] = validation
    audit["outputs"] = {
        "train_features": str(train_path.as_posix()),
        "test_features": str(test_path.as_posix()),
        "feature_dictionary": str(dictionary_path.as_posix()),
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=_json_default)

    return {
        "train": train,
        "test": test,
        "audit": audit,
        "feature_dictionary": feature_dictionary,
        "paths": {
            "train": train_path,
            "test": test_path,
            "audit": audit_path,
            "feature_dictionary": dictionary_path,
        },
        "monthly": {"coe": coe_monthly, "stocks": stock_monthly},
    }


__all__ = [
    "SmoothedGroupTargetEncoder",
    "add_macro_features",
    "audit_raw_data",
    "build_block_geospatial_features",
    "build_feature_dictionary",
    "canonicalize_flat_type",
    "enrich_with_block_features",
    "load_raw_data",
    "nearest_distance_and_counts",
    "normalize_text",
    "parse_numeric_text",
    "prepare_coe_monthly",
    "prepare_core",
    "prepare_hdb_blocks",
    "prepare_mrt_stations",
    "prepare_stock_monthly",
    "run_pipeline",
    "validate_auxiliary_data",
    "validate_core_schema",
    "validate_final_outputs",
    "validate_macro_availability_flags",
    "validate_monthly_timing",
]
