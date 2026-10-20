"""Reproducible exploratory checks for the added CS5228 auxiliary sources."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.data_preprocessing import canonicalize_flat_type, nearest_distance_and_counts


def load_new_auxiliary(data_dir: str | Path) -> dict:
    """Load the three added sources and validate their basic structure."""
    root = Path(data_dir) / "auxiliary"
    cpi = pd.read_csv(root / "sg-cpi.csv")
    tertiary = pd.read_csv(root / "sg-tertiary-institutions.csv")
    with (root / "sg-rail-lines.geojson").open(encoding="utf-8") as stream:
        rail = json.load(stream)
    expected_cpi = {"MONTH", "CPI_ALL_ITEMS", "CPI_ACCOMMODATION", "CPI_ACTUAL_RENTALS_HOUSING"}
    expected_tertiary = {"name", "type", "LATITUDE", "LONGITUDE"}
    if set(cpi) != expected_cpi or set(tertiary) != expected_tertiary:
        raise ValueError("New auxiliary source schema changed")
    months = pd.to_datetime(cpi["MONTH"], format="%Y-%m", errors="raise")
    expected_months = pd.date_range(months.min(), months.max(), freq="MS")
    if cpi["MONTH"].duplicated().any() or not months.sort_values().reset_index(drop=True).equals(pd.Series(expected_months)):
        raise ValueError("CPI months must be unique and consecutive")
    if cpi.isna().any().any() or (cpi.drop(columns="MONTH") <= 0).any().any():
        raise ValueError("CPI values must be complete and positive")
    if tertiary.isna().any().any() or tertiary["name"].duplicated().any():
        raise ValueError("Tertiary institution records must be complete and unique")
    if not tertiary["LATITUDE"].between(1.0, 1.6).all() or not tertiary["LONGITUDE"].between(103.5, 104.2).all():
        raise ValueError("Tertiary institution coordinates are outside Singapore")
    if rail.get("type") != "FeatureCollection" or not rail.get("features"):
        raise ValueError("Rail source must be a nonempty FeatureCollection")
    rail_types = {}
    geometries = {}
    for feature in rail["features"]:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        rail_type = properties.get("RAIL_TYPE")
        kind = geometry.get("type")
        if kind not in {"LineString", "MultiLineString"}:
            raise ValueError(f"Unexpected rail geometry: {kind}")
        rail_types[rail_type] = rail_types.get(rail_type, 0) + 1
        geometries[kind] = geometries.get(kind, 0) + 1
    return {"cpi": cpi, "tertiary": tertiary, "rail": rail,
            "rail_types": rail_types, "rail_geometries": geometries}


def tertiary_distance_summary(blocks: pd.DataFrame, institutions: pd.DataFrame) -> pd.DataFrame:
    """Describe block distance to supplied institutions without using rent values."""
    origins = blocks[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)
    records = []
    for group, subset in [("any tertiary", institutions),
                          ("university", institutions.loc[institutions["type"].eq("university")]),
                          ("polytechnic", institutions.loc[institutions["type"].eq("polytechnic")])]:
        nearest, counts = nearest_distance_and_counts(
            origins, subset[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float), [2.0]
        )
        records.append({"group": group, "institutions": len(subset),
                        "median_nearest_km": float(np.median(nearest)),
                        "blocks_within_2km": int((counts[2.0] > 0).sum()),
                        "block_count": len(blocks)})
    return pd.DataFrame(records)


def summarize_outliers(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag descriptive rent tails within flat type and approval year."""
    clean = train.drop_duplicates(keep="first").copy()
    clean["approval_year"] = pd.to_datetime(clean["RENT_APPROVAL_DATE"], format="%Y-%m").dt.year
    clean["flat_type_group"] = canonicalize_flat_type(clean["FLAT_TYPE"])
    group = clean.groupby(["flat_type_group", "approval_year"])["MONTHLY_RENT"]
    q1 = group.transform(lambda x: x.quantile(0.25))
    q3 = group.transform(lambda x: x.quantile(0.75))
    spread = q3 - q1
    lower = (q1 - 1.5 * spread).clip(lower=0)
    upper = q3 + 1.5 * spread
    clean["iqr_low_flag"] = clean["MONTHLY_RENT"].lt(lower)
    clean["iqr_high_flag"] = clean["MONTHLY_RENT"].gt(upper)
    summary = pd.DataFrame({
        "metric": ["Retained training rows", "Low rents below SGD 1,000", "Within-group IQR low flags", "Within-group IQR high flags", "Any IQR flag"],
        "rows": [len(clean), int(clean["MONTHLY_RENT"].lt(1000).sum()),
                 int(clean["iqr_low_flag"].sum()), int(clean["iqr_high_flag"].sum()),
                 int((clean["iqr_low_flag"] | clean["iqr_high_flag"]).sum())],
    })
    return summary, clean


def rent_correlations(train_features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute descriptive rank and linear associations with observed rent."""
    chosen = [column for column in columns if column in train_features]
    data = train_features[chosen + ["MONTHLY_RENT"]].select_dtypes(include="number")
    return pd.DataFrame({
        "feature": chosen,
        "pearson_r": [data[column].corr(data["MONTHLY_RENT"], method="pearson") for column in chosen],
        "spearman_rho": [data[column].corr(data["MONTHLY_RENT"], method="spearman") for column in chosen],
        "missing_rows": [int(data[column].isna().sum()) for column in chosen],
    }).sort_values("spearman_rho", key=lambda values: values.abs(), ascending=False).reset_index(drop=True)
