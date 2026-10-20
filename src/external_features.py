"""Features from the three team-added sources: CPI, rail lines and tertiary campuses.

These sources were shown in the progress report but were not yet in the saved
parquet files. The data freeze (decision 4 of the final report plan) keeps the
source list as it is now, so this module adds exactly these three and nothing new.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import AUXILIARY_DIR, DERIVED_DIR

BLOCK_EXTRAS_CSV = DERIVED_DIR / "block_extra_features.csv"
RAIL_GEOJSON = AUXILIARY_DIR / "sg-rail-lines.geojson"
TERTIARY_CSV = AUXILIARY_DIR / "sg-tertiary-institutions.csv"
CPI_CSV = AUXILIARY_DIR / "sg-cpi.csv"

CPI_COLUMNS = ["CPI_ALL_ITEMS", "CPI_ACCOMMODATION", "CPI_ACTUAL_RENTALS_HOUSING"]
BLOCK_EXTRA_COLUMNS = [
    "NEAREST_RAIL_LINE_KM",
    "NEAREST_ABOVEGROUND_RAIL_KM",
    "NEAREST_TERTIARY_KM",
    "NEAREST_UNIVERSITY_KM",
    "TERTIARY_COUNT_2KM",
]

# A local flat projection is accurate to well under 1% across Singapore.
_KM_PER_DEG_LAT = 110.574
_REF_LAT = 1.35
_KM_PER_DEG_LON = 111.320 * np.cos(np.radians(_REF_LAT))
_RAIL_STEP_KM = 0.025


def _project(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.column_stack([lon * _KM_PER_DEG_LON, lat * _KM_PER_DEG_LAT])


def _densified_rail_points(features: list[dict]) -> np.ndarray:
    """Turn rail polylines into points no more than 25 m apart."""

    points = []
    for feature in features:
        geometry = feature["geometry"]
        parts = geometry["coordinates"]
        if geometry["type"] == "LineString":
            parts = [parts]
        for part in parts:
            coords = np.asarray(part, dtype=float)[:, :2]
            xy = _project(coords[:, 1], coords[:, 0])
            points.append(xy[:1])
            for start, end in zip(xy[:-1], xy[1:]):
                steps = max(1, int(np.ceil(np.linalg.norm(end - start) / _RAIL_STEP_KM)))
                fractions = np.arange(1, steps + 1)[:, None] / steps
                points.append(start + fractions * (end - start))
    return np.vstack(points)


def compute_block_extras(blocks: pd.DataFrame) -> pd.DataFrame:
    """Compute rail and tertiary distances for unique ADDRESS_KEY coordinates.

    The rail distance is to the nearest point on the densified line, so it is
    within about 13 m of the exact distance to the polyline.
    """

    with RAIL_GEOJSON.open(encoding="utf-8") as stream:
        rail = json.load(stream)["features"]
    aboveground = [f for f in rail if (f.get("properties") or {}).get("GRND_LEVEL") == "ABOVEGROUND"]
    origins = _project(blocks["HDB_LATITUDE"].to_numpy(), blocks["HDB_LONGITUDE"].to_numpy())

    result = blocks[["ADDRESS_KEY"]].copy()
    result["NEAREST_RAIL_LINE_KM"] = cKDTree(_densified_rail_points(rail)).query(origins)[0]
    result["NEAREST_ABOVEGROUND_RAIL_KM"] = cKDTree(_densified_rail_points(aboveground)).query(origins)[0]

    tertiary = pd.read_csv(TERTIARY_CSV)
    campus_xy = _project(tertiary["LATITUDE"].to_numpy(), tertiary["LONGITUDE"].to_numpy())
    distances = np.linalg.norm(origins[:, None, :] - campus_xy[None, :, :], axis=2)
    university = tertiary["type"].eq("university").to_numpy()
    result["NEAREST_TERTIARY_KM"] = distances.min(axis=1)
    result["NEAREST_UNIVERSITY_KM"] = distances[:, university].min(axis=1)
    result["TERTIARY_COUNT_2KM"] = (distances <= 2.0).sum(axis=1).astype("int16")
    for column in BLOCK_EXTRA_COLUMNS[:-1]:
        result[column] = result[column].astype("float32").round(4)
    return result


def load_block_extras(frames: list[pd.DataFrame], refresh: bool = False) -> pd.DataFrame:
    """Load the cached block table, computing it first if it is missing.

    The cache is small and is committed to git, so teammates do not need the
    24 MB rail GeoJSON unless they want to rebuild it.
    """

    keys = (
        pd.concat([f[["ADDRESS_KEY", "HDB_LATITUDE", "HDB_LONGITUDE"]] for f in frames])
        .drop_duplicates("ADDRESS_KEY")
        .sort_values("ADDRESS_KEY")
        .reset_index(drop=True)
    )
    if BLOCK_EXTRAS_CSV.exists() and not refresh:
        cached = pd.read_csv(BLOCK_EXTRAS_CSV)
        if set(keys["ADDRESS_KEY"]) <= set(cached["ADDRESS_KEY"]):
            return cached
    if not RAIL_GEOJSON.exists():
        raise FileNotFoundError(
            f"{BLOCK_EXTRAS_CSV} is missing or stale and {RAIL_GEOJSON} is not there to rebuild it. "
            "See data/README.md."
        )
    extras = compute_block_extras(keys)
    BLOCK_EXTRAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    extras.to_csv(BLOCK_EXTRAS_CSV, index=False)
    return extras


def load_cpi_monthly() -> pd.DataFrame:
    """CPI by calendar month, matched to the approval month (current values)."""

    cpi = pd.read_csv(CPI_CSV)
    cpi["RENT_APPROVAL_DATE"] = pd.to_datetime(cpi.pop("MONTH"), format="%Y-%m")
    return cpi[["RENT_APPROVAL_DATE", *CPI_COLUMNS]]


def add_external_features(frame: pd.DataFrame, block_extras: pd.DataFrame) -> pd.DataFrame:
    """Join CPI by month and the block table by ADDRESS_KEY, keeping row order."""

    rows = len(frame)
    out = frame.merge(load_cpi_monthly(), on="RENT_APPROVAL_DATE", how="left", validate="many_to_one")
    out = out.merge(block_extras, on="ADDRESS_KEY", how="left", validate="many_to_one")
    if len(out) != rows or out[CPI_COLUMNS + BLOCK_EXTRA_COLUMNS].isna().any().any():
        raise ValueError("External feature join lost rows or left gaps")
    return out
