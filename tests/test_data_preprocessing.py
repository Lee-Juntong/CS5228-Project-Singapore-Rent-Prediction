from pathlib import Path

import numpy as np
import nbformat
import pandas as pd
import pytest

from src.data_preprocessing import (
    SmoothedGroupTargetEncoder,
    canonicalize_flat_type,
    nearest_distance_and_counts,
    normalize_text,
    parse_numeric_text,
    prepare_core,
    prepare_coe_monthly,
    prepare_hdb_blocks,
    prepare_stock_monthly,
    run_pipeline,
    validate_macro_availability_flags,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
needs_raw_data = pytest.mark.skipif(not (DATA_DIR / "train.csv").exists(), reason="Kaggle files are not in data/, see data/README.md")


def test_text_normalization_and_flat_type_mapping() -> None:
    values = pd.Series(["  Bedok  South ", "ANG MO KIO"], dtype="string")
    assert normalize_text(values).tolist() == ["bedok south", "ang mo kio"]

    flat_types = pd.Series(["1 room", "2-room", " 5 ROOM ", "executive"])
    assert canonicalize_flat_type(flat_types).tolist() == [
        "1-room",
        "2-room",
        "5-room",
        "executive",
    ]


def test_numeric_text_parser() -> None:
    parsed = parse_numeric_text(pd.Series(["$40,609", "1,280", " 35 "]))
    assert parsed.tolist() == [40609, 1280, 35]


def test_invalid_lease_date_is_flagged_without_rewriting_source() -> None:
    frame = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": ["2021-11"],
            "TOWN": ["bedok"],
            "BLOCK": ["153b"],
            "STREET": ["Bedok South Road"],
            "FLAT_TYPE": ["5 room"],
            "FLAT_MODEL": ["improved"],
            "FLOOR_AREA_SQM": [113.0],
            "FURNISHED": ["yes"],
            "LEASE_COMMENCE_DATE": [2022],
            "FEE": [0.0],
            "MONTHLY_RENT": [3200],
        }
    )
    result = prepare_core(frame, is_train=True)
    assert result.loc[0, "LEASE_COMMENCE_DATE"] == 2022
    assert result.loc[0, "INVALID_LEASE_DATE_FLAG"] == 1
    assert pd.isna(result.loc[0, "FLAT_AGE_YEARS"])
    assert pd.isna(result.loc[0, "REMAINING_LEASE_PROXY_YEARS"])


def test_haversine_nearest_distance_and_radius_counts() -> None:
    origins = np.array([[1.3000, 103.8000], [1.3100, 103.8100]])
    destinations = np.array([[1.3000, 103.8000]])
    nearest, counts = nearest_distance_and_counts(origins, destinations, [0.5, 2.0])
    assert nearest[0] == pytest.approx(0.0)
    assert 1.0 < nearest[1] < 2.0
    assert counts[0.5].tolist() == [1, 0]
    assert counts[2.0].tolist() == [1, 1]


@needs_raw_data
def test_coe_lags_exclude_the_current_month() -> None:
    raw = pd.read_csv(DATA_DIR / "auxiliary" / "sg-coe-prices.csv")
    monthly = prepare_coe_monthly(raw).set_index("RENT_APPROVAL_DATE")
    february = pd.Timestamp("2021-02-01")
    january = pd.Timestamp("2021-01-01")
    assert monthly.loc[february, "COE_A_PREMIUM_LAG1"] == pytest.approx(
        monthly.loc[january, "COE_A_PREMIUM_CURRENT"]
    )
    assert pd.isna(monthly.loc[january, "COE_A_PREMIUM_LAG1"])
    assert pd.isna(monthly.loc[pd.Timestamp("2021-03-01"), "COE_A_PREMIUM_LAGGED_3M_MEAN"])
    expected = monthly.loc[
        pd.Timestamp("2021-01-01") : pd.Timestamp("2021-03-01"),
        "COE_A_PREMIUM_CURRENT",
    ].mean()
    assert monthly.loc[pd.Timestamp("2021-04-01"), "COE_A_PREMIUM_LAGGED_3M_MEAN"] == pytest.approx(expected)


@needs_raw_data
def test_stock_lags_and_fixed_universe() -> None:
    raw = pd.read_csv(DATA_DIR / "auxiliary" / "sg-stock-prices.csv")
    monthly = prepare_stock_monthly(raw).set_index("RENT_APPROVAL_DATE")
    assert pd.isna(monthly.loc[pd.Timestamp("2021-01-01"), "STOCK_SG_MARKET_RETURN_CURRENT"])
    assert monthly.loc[pd.Timestamp("2021-03-01"), "STOCK_SG_MARKET_RETURN_LAG1"] == pytest.approx(
        monthly.loc[pd.Timestamp("2021-02-01"), "STOCK_SG_MARKET_RETURN_CURRENT"]
    )


@needs_raw_data
def test_auxiliary_sentinels_and_internal_time_gaps_are_validated() -> None:
    hdb = pd.read_csv(
        DATA_DIR / "auxiliary" / "sg-hdb-block.csv",
        dtype={"POSTAL_CODE": "string"},
    )
    cleaned = prepare_hdb_blocks(hdb)
    sentinel_rows = hdb["YEAR_COMPLETED"].eq(-1)
    assert sentinel_rows.sum() == 11
    assert cleaned.loc[sentinel_rows, "HDB_YEAR_COMPLETED"].isna().all()
    assert cleaned.loc[sentinel_rows, "HDB_MAX_FLOOR"].isna().all()
    assert cleaned.loc[sentinel_rows, "HDB_YEAR_COMPLETED_MISSING_FLAG"].eq(1).all()
    assert cleaned.loc[sentinel_rows, "HDB_MAX_FLOOR_MISSING_FLAG"].eq(1).all()

    invalid_hdb = hdb.copy()
    invalid_hdb.loc[0, "MAX_FLOOR"] = 0
    with pytest.raises(ValueError, match="unsupported nonpositive"):
        prepare_hdb_blocks(invalid_hdb)

    coe = pd.read_csv(DATA_DIR / "auxiliary" / "sg-coe-prices.csv")
    coe_gap = coe.loc[~((coe["YEAR"] == 2021) & (coe["MONTH"] == "feb"))]
    with pytest.raises(ValueError, match="internal month gap"):
        prepare_coe_monthly(coe_gap)

    stocks = pd.read_csv(DATA_DIR / "auxiliary" / "sg-stock-prices.csv")
    stock_dates = pd.to_datetime(stocks["DATE"])
    stock_gap = stocks.loc[
        ~(
            stocks["SYMBOL"].eq("D05.SI")
            & stock_dates.dt.to_period("M").eq(pd.Period("2021-02"))
        )
    ]
    with pytest.raises(ValueError, match="missing monthly closes"):
        prepare_stock_monthly(stock_gap)


def test_smoothed_target_encoder_uses_leave_one_out_and_unseen_fallback() -> None:
    frame = pd.DataFrame({"group": ["a", "a", "b"]})
    target = pd.Series([10.0, 30.0, 100.0])
    encoder = SmoothedGroupTargetEncoder(["group"], alpha=2.0)
    encoded = encoder.fit_transform(frame, target)

    assert encoded.iloc[0] == pytest.approx((30.0 + 2 * 65.0) / 3)
    assert encoded.iloc[1] == pytest.approx((10.0 + 2 * 55.0) / 3)
    assert encoded.iloc[2] == pytest.approx(20.0)
    unseen = encoder.transform(pd.DataFrame({"group": ["unseen"]}))
    assert unseen.iloc[0] == pytest.approx(target.mean())


@needs_raw_data
def test_full_pipeline_removes_exact_train_duplicates_and_preserves_test(tmp_path: Path) -> None:
    result = run_pipeline(
        DATA_DIR,
        tmp_path / "processed",
        tmp_path / "audit",
    )
    train = result["train"]
    test = result["test"]
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    retained_mask = ~raw_train.duplicated(keep="first")
    retained_row_ids = raw_train.index[retained_mask].tolist()

    assert len(train) == 149_575
    assert len(test) == 50_000
    assert train["ROW_ID"].tolist() == retained_row_ids
    assert test["ROW_ID"].tolist() == list(range(50_000))
    assert train["MONTHLY_RENT"].tolist() == raw_train.loc[
        retained_mask, "MONTHLY_RENT"
    ].tolist()
    assert not raw_train.loc[train["ROW_ID"]].duplicated().any()
    assert list(train.drop(columns="MONTHLY_RENT").columns) == list(test.columns)
    assert train["HDB_POSTAL_CODE_MISSING_FLAG"].sum() > 0
    assert test["HDB_POSTAL_CODE_MISSING_FLAG"].sum() > 0
    assert train["HDB_POSTAL_CODE"].isna().equals(
        train["HDB_POSTAL_CODE_MISSING_FLAG"].astype(bool)
    )
    assert test["HDB_POSTAL_CODE"].isna().equals(
        test["HDB_POSTAL_CODE_MISSING_FLAG"].astype(bool)
    )
    assert train["HDB_BLOCK_AGE_YEARS"].max() < 100
    assert test["HDB_BLOCK_AGE_YEARS"].max() < 100
    assert train.loc[
        train["HDB_YEAR_COMPLETED_MISSING_FLAG"].eq(1), "HDB_BLOCK_AGE_YEARS"
    ].isna().all()
    assert "SMOOTHED_TARGET_MEAN" not in train.columns
    assert not ({"FURNISHED", "FEE"} & set(train.columns))
    validate_macro_availability_flags(train.drop(columns="MONTHLY_RENT"))
    validate_macro_availability_flags(test)

    model_features = train.drop(columns=["ROW_ID", "MONTHLY_RENT"])
    categorical_columns = model_features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    datetime_columns = model_features.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    numeric_columns = [
        column
        for column in model_features.columns
        if column not in categorical_columns + datetime_columns
    ]
    categorical_levels = {
        column: int(train[column].nunique(dropna=False))
        for column in categorical_columns
    }
    assert len(model_features.columns) == 77
    assert len(categorical_columns) == 10
    assert len(numeric_columns) == 66
    assert datetime_columns == ["RENT_APPROVAL_DATE"]
    assert sum(categorical_levels.values()) == 22_068
    assert len(numeric_columns) + sum(categorical_levels.values()) == 22_134
    reduced_levels = sum(
        levels
        for column, levels in categorical_levels.items()
        if column not in {"ADDRESS_KEY", "HDB_POSTAL_CODE", "BLOCK"}
    )
    assert len(numeric_columns) + reduced_levels == 886
    assert int(train["HDB_POSTAL_CODE"].isna().sum()) == 5_241
    unseen_levels = {
        column: len(
            set(test[column].dropna().astype(str).unique())
            - set(train[column].dropna().astype(str).unique())
        )
        for column in categorical_columns
    }
    assert unseen_levels["ADDRESS_KEY"] == 50
    assert unseen_levels["HDB_POSTAL_CODE"] == 50
    assert unseen_levels["BLOCK"] == 11

    reloaded_train = pd.read_parquet(result["paths"]["train"])
    reloaded_test = pd.read_parquet(result["paths"]["test"])
    pd.testing.assert_frame_equal(reloaded_train, train, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(reloaded_test, test, check_dtype=True, check_exact=True)
    assert result["audit"]["final_validation"]["infinite_numeric_values"] == 0
    assert result["audit"]["cleaning"] == {
        "raw_training_rows": 150_000,
        "exact_duplicate_training_rows_removed": 425,
        "training_rows_after_exact_deduplication": 149_575,
        "deduplication_policy": "Keep the first exact training row in original order",
        "test_rows_removed": 0,
    }
    assert result["audit"]["final_validation"][
        "train_exact_duplicate_rows_removed"
    ] == 425
    assert result["audit"]["feature_engineering"]["parquet_round_trip_exact"] is True
    dictionary = result["feature_dictionary"]
    assert dictionary["feature"].is_unique
    assert dictionary["definition"].str.len().gt(20).all()
    assert dictionary["expected_missingness"].notna().all()


def test_notebook_inventory_uses_reader_facing_source_roles() -> None:
    notebook = nbformat.read(
        PROJECT_ROOT / "notebooks" / "01_preprocessing_feature_engineering.ipynb",
        as_version=4,
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type in {"markdown", "code"}
    )
    assert "## 2. Source tables and raw audit" in source
    assert "Primary modeling data" in source
    assert "Location reference tables" in source
    assert "Economic time series" in source
    assert "Submission template" in source
    assert "HDB block reference" in source
    assert ".rename_axis('dataset')" not in source


def test_notebook_duplicate_and_category_audits_are_self_explanatory() -> None:
    notebook = nbformat.read(
        PROJECT_ROOT / "notebooks" / "01_preprocessing_feature_engineering.ipynb",
        as_version=4,
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type in {"markdown", "code"}
    )
    assert "### Duplicate diagnostics" in source
    assert "'context'" in source
    assert "'meaning'" in source
    assert "'pipeline treatment'" in source
    assert "Exact duplicate training rows removed" in source
    assert "preserves every transaction" not in source
    assert "### Flat-type label normalization" in source
    assert "'canonical label'" in source
    assert "'training rows'" in source
    assert "'test rows'" in source
    assert "pd.Series(core['train_flat_type_labels'])" not in source


def test_notebook_explains_model_specific_feature_preparation() -> None:
    notebook = nbformat.read(
        PROJECT_ROOT / "notebooks" / "01_preprocessing_feature_engineering.ipynb",
        as_version=4,
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type in {"markdown", "code"}
    )
    assert "## 8. Using the features with different prediction methods" in source
    assert "one_hot_indicator_columns" in source
    assert "CatBoost" in source
    assert "Ridge or Elastic Net" in source
    assert "k-nearest neighbours or kernel SVR" in source
    assert "SmoothedGroupTargetEncoder" in source
    assert "fit every encoder" in source.lower()
    assert "handle_unknown='ignore'" in source
    assert "theoretical full-width estimates" in source
    assert "`__MISSING__`" in source
    assert "`__UNKNOWN__`" in source
    assert "frequency zero" in source
    assert "ADDRESS_KEY" in source
    assert "_CURRENT" in source
    assert "_LAG1" in source
    assert "_LAGGED_3M_MEAN" in source
