"""Descriptive EDA and repeated-profile dispersion, without model fits.

Inputs are the cleaned, row-identified training data returned by the shared
loader. Exact training duplicates have already been removed upstream; this
module never removes additional rows or supplies target-derived model inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TARGET, TIME_VALID_END, TIME_VALID_START
from src.data_preprocessing import canonicalize_flat_type, normalize_text
from src.features import feature_columns


def profile_keys(frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Exact profile keys; missing/empty address components make a row ineligible.

    ADDRESS_KEY is the shared normalized town|block|street identifier. If the
    column is absent, build that identifier from all three source components.
    Areas and lease years are numeric with no rounding; dates are calendar
    months. The stricter sensitivity adds flat model and lease commence year.
    """
    keys = pd.DataFrame(index=frame.index)
    if "ADDRESS_KEY" in frame:
        parts = frame["ADDRESS_KEY"].astype("string").str.split("|", regex=False)
        valid = parts.map(lambda value: isinstance(value, list) and len(value) == 3)
        components = [normalize_text(parts.str.get(i)).replace("", pd.NA) for i in range(3)]
        address = components[0].str.cat(components[1:], sep="|")
        keys["ADDRESS_KEY"] = address.where(valid)
    else:
        components = [normalize_text(frame[name]).replace("", pd.NA)
                      for name in ["TOWN", "BLOCK", "STREET"]]
        keys["ADDRESS_KEY"] = components[0].str.cat(components[1:], sep="|")
    flat_type = normalize_text(frame["FLAT_TYPE"]).replace("", pd.NA)
    keys["FLAT_TYPE"] = canonicalize_flat_type(flat_type)
    keys["FLOOR_AREA_SQM"] = pd.to_numeric(frame["FLOOR_AREA_SQM"], errors="raise")
    keys["FLOOR_AREA_SQM"] = keys["FLOOR_AREA_SQM"].where(
        np.isfinite(keys["FLOOR_AREA_SQM"]) & keys["FLOOR_AREA_SQM"].gt(0)
    )
    dates = pd.to_datetime(frame["RENT_APPROVAL_DATE"], errors="raise")
    keys["APPROVAL_MONTH"] = dates.dt.to_period("M").dt.to_timestamp()
    if strict:
        keys["FLAT_MODEL"] = normalize_text(frame["FLAT_MODEL"]).replace("", pd.NA)
        lease = pd.to_numeric(frame["LEASE_COMMENCE_DATE"], errors="raise")
        keys["LEASE_COMMENCE_DATE"] = lease.where(np.isfinite(lease))
    return keys


def repeated_profile_dispersion(
    frame: pd.DataFrame, strict: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return descriptive summary, repeated groups (n>=2) and all-group sizes.

    Let SSE_g=sum_i(y_ig-ybar_g)^2. The pooled SD uses sum_g(n_g-1)
    degrees of freedom; the self-inclusive RMSE uses sum_g n_g. Leave-one-out
    errors multiply each centered rent by n_g/(n_g-1), avoiding self-inclusion.
    All spread statistics describe the selected repeated-profile rows only.
    These are descriptive proxies, not validation models or Bayes-error bounds.
    """
    keys = profile_keys(frame, strict)
    target = pd.to_numeric(frame[TARGET], errors="raise")
    if target.isna().any() or not np.isfinite(target).all() or not target.gt(0).all():
        raise ValueError("Dispersion requires positive, finite rent targets")
    if "ROW_ID" in frame and (frame["ROW_ID"].isna().any() or frame["ROW_ID"].duplicated().any()):
        raise ValueError("ROW_ID must be nonmissing and unique; use cleaned training rows")
    eligible = keys.notna().all(axis=1)
    data = keys.loc[eligible].copy()
    data[TARGET] = target.loc[eligible]
    groups = data.groupby(list(keys), observed=True, dropna=False)[TARGET].agg(
        n="size", mean="mean", median="median", sample_sd="std", minimum="min", maximum="max"
    ).reset_index()
    sizes = groups.groupby("n").size().rename("group_count").reset_index()
    sizes["row_count"] = sizes["n"] * sizes["group_count"]
    repeated = groups.loc[groups["n"].ge(2)].copy()
    repeated["range_sgd"] = repeated["maximum"] - repeated["minimum"]
    repeated["within_group_sse"] = repeated["sample_sd"].pow(2) * (repeated["n"] - 1)
    row_count = int(repeated["n"].sum())
    group_count = len(repeated)
    sse = float(repeated["within_group_sse"].sum())
    summary = {
        "grouping": "strict" if strict else "plan",
        "keys": list(keys),
        "normalization": "Shared lowercase, trim and whitespace normalization; canonical flat types; exact numeric area; calendar month",
        "duplicate_policy": "Input must be cleaned training data; shared pipeline removes raw exact duplicates keeping first ROW_ID; no further removal here",
        "minimum_group_size_for_spread": 2,
        "missing_key_policy": "Exclude rows missing any grouping key (including empty/malformed address or nonpositive/nonfinite area)",
        "input_rows": len(frame),
        "eligible_rows": int(eligible.sum()),
        "excluded_missing_key_rows": int((~eligible).sum()),
        "all_groups": len(groups),
        "singleton_groups": int(groups["n"].eq(1).sum()),
        "repeated_groups": group_count,
        "repeated_rows": row_count,
        "coverage_fraction_all_rows": row_count / len(frame) if len(frame) else None,
        "coverage_fraction_eligible_rows": row_count / int(eligible.sum()) if eligible.any() else None,
        "unweighted_mean_group_sample_sd_sgd": float(repeated["sample_sd"].mean()) if group_count else None,
        "row_weighted_rms_group_sample_sd_sgd": float(np.sqrt(np.average(repeated["sample_sd"].pow(2), weights=repeated["n"]))) if row_count else None,
        "pooled_within_group_sample_sd_sgd": float(np.sqrt(sse / (row_count - group_count))) if row_count else None,
        "self_inclusive_group_mean_rmse_sgd": float(np.sqrt(sse / row_count)) if row_count else None,
        "leave_one_out_group_mean_rmse_sgd": float(np.sqrt((repeated["within_group_sse"] * (repeated["n"] / (repeated["n"] - 1)).pow(2)).sum() / row_count)) if row_count else None,
        "median_group_size_repeated": float(repeated["n"].median()) if group_count else None,
        "max_group_size": int(groups["n"].max()) if len(groups) else None,
        "caveat": "Selected-subset dispersion, not a proven Bayes error or universal RMSE floor; groups may contain different units and omit recorded attributes",
    }
    return summary, repeated.reset_index(drop=True), sizes


def _rent_summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    data = frame.copy()
    for name in by:
        if name != "RENT_APPROVAL_DATE":
            data[name] = data[name].astype("string").fillna("Missing")
    return data.groupby(by, observed=True, dropna=False)[TARGET].agg(
        n="size", mean="mean", median="median", q25=lambda x: x.quantile(.25),
        q75=lambda x: x.quantile(.75), minimum="min", maximum="max",
    ).reset_index()


def _save_figure(fig, root: Path, name: str) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _figures(train: pd.DataFrame, tables: dict[str, pd.DataFrame], root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import StrMethodFormatter

    blue, gold, gray = "#245A81", "#BE8731", "#4E555B"
    style = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.labelcolor": "#252A2D", "text.color": "#252A2D", "grid.color": "#E4E8EB"}
    with plt.rc_context(style):
        hist, monthly = tables["rent_histogram"], tables["rent_by_month"]
        fig, axes = plt.subplots(1, 2, figsize=(11.3, 3.8), layout="constrained")
        axes[0].bar(hist["lower_sgd"], hist["n"], width=hist["upper_sgd"] - hist["lower_sgd"], align="edge", color=blue, edgecolor="white", linewidth=.3)
        axes[0].set(xlabel="Monthly rent (SGD)", ylabel="Training rows", title=f"Rent distribution (n={len(train):,})")
        axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes[1].fill_between(monthly["RENT_APPROVAL_DATE"], monthly["q25"], monthly["q75"], color=blue, alpha=.15, label="25th–75th percentile")
        axes[1].plot(monthly["RENT_APPROVAL_DATE"], monthly["median"], color=blue, linewidth=1.8, marker="o", markersize=2.2, label="Median")
        axes[1].axvline(TIME_VALID_START, color=gray, linestyle="--", linewidth=1, label="Holdout starts")
        axes[1].set(xlabel="Approval month", ylabel="Monthly rent (SGD)", title="Monthly rent over the cleaned training period")
        tick_months = list(pd.date_range(monthly["RENT_APPROVAL_DATE"].min(), monthly["RENT_APPROVAL_DATE"].max(), freq="12MS"))
        if len(tick_months) > 1 and (monthly["RENT_APPROVAL_DATE"].max() - tick_months[-1]).days < 180:
            tick_months.pop()
        tick_months.append(monthly["RENT_APPROVAL_DATE"].max())
        axes[1].set_xticks(sorted(set(tick_months)))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[1].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes[1].legend(frameon=False, fontsize=8, loc="upper left")
        _save_figure(fig, root, "eda_distribution_trend")

        for table_name, column, title, height in [
            ("rent_by_flat_type", "FLAT_TYPE", "Rent by flat type", 3.6),
            ("rent_by_town", "TOWN", "Rent by town", 8),
        ]:
            table = tables[table_name].sort_values("median")
            labels = table[column].str.replace("-", " ").str.title() + " (n=" + table["n"].map("{:,}".format) + ")"
            fig, ax = plt.subplots(figsize=(8.4, min(height, max(3, .29 * len(table) + 1.4))), layout="constrained")
            positions = np.arange(len(table))
            ax.hlines(positions, table["q25"], table["q75"], color=gold, linewidth=2.3, label="25th–75th percentile")
            ax.scatter(table["median"], positions, color=blue, s=25, zorder=3, label="Median")
            ax.set_yticks(positions, labels)
            ax.set(xlabel="Monthly rent (SGD)", title=f"{title} — cleaned training rows")
            ax.set_xlim(left=0)
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.legend(frameon=False, loc="upper left", fontsize=8)
            _save_figure(fig, root, table_name.replace("rent_by_", "eda_"))

        # Monthly town medians retain local trends without overlaying 26 lines.
        town_month = tables["rent_by_town_month"].pivot(index="TOWN", columns="RENT_APPROVAL_DATE", values="median")
        fig, ax = plt.subplots(figsize=(10, 7.3), layout="constrained")
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad("#E4E8EB")
        image = ax.imshow(town_month, aspect="auto", interpolation="none", cmap=cmap)
        ax.set_yticks(np.arange(len(town_month)), town_month.index.str.title())
        ticks = sorted(set(town_month.columns.get_loc(date) for date in tick_months if date in town_month.columns))
        ax.set_xticks(ticks, town_month.columns[ticks].strftime("%b %Y"))
        ax.set(title="Monthly median rent by town", xlabel="Approval month")
        fig.colorbar(image, ax=ax, label="Monthly rent (SGD)", shrink=.85)
        _save_figure(fig, root, "eda_town_month")

        flat_month = tables["rent_by_flat_type_month"]
        flat_types = [name for name in ["1-room", "2-room", "3-room", "4-room", "5-room", "executive"] if name in set(flat_month["FLAT_TYPE"])]
        columns = min(3, len(flat_types))
        rows = int(np.ceil(len(flat_types) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 2.9 * rows), sharex=True, sharey=True, squeeze=False, layout="constrained")
        for ax, name in zip(axes.flat, flat_types):
            table = flat_month.loc[flat_month["FLAT_TYPE"].eq(name)].sort_values("RENT_APPROVAL_DATE")
            # Sparse types (especially 1-room) have unobserved months. Keep
            # those gaps rather than drawing a continuous interpolated trend.
            table = table.set_index("RENT_APPROVAL_DATE").reindex(monthly["RENT_APPROVAL_DATE"]).reset_index()
            ax.fill_between(table["RENT_APPROVAL_DATE"], table["q25"], table["q75"], color=blue, alpha=.15)
            ax.plot(table["RENT_APPROVAL_DATE"], table["median"], color=blue, linewidth=1.6, marker="o", markersize=2)
            ax.set_title(f"{name.replace('-', ' ').title()} (n={int(table['n'].sum()):,})", fontsize=10)
            ax.set_xticks(monthly["RENT_APPROVAL_DATE"].iloc[sorted(set([0, len(monthly) // 2, len(monthly) - 1]))])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        for ax in list(axes.flat)[len(flat_types):]:
            ax.set_visible(False)
        fig.suptitle("Monthly rent by flat type: median and 25th–75th percentile")
        fig.supxlabel("Approval month")
        fig.supylabel("Monthly rent (SGD)")
        _save_figure(fig, root, "eda_flat_type_month")


def generate_eda(data: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame], output_dir: str | Path) -> dict:
    """Write report source tables/figures and return machine-readable findings.

    This function requires real cleaned data. No numerical result is produced
    when those inputs are unavailable. Call from the analysis report runner.
    """
    train, test = data if isinstance(data, tuple) else (data, None)
    train = train.copy()
    if train.empty:
        raise ValueError("EDA requires nonempty cleaned training data")
    if train["ROW_ID"].isna().any() or train["ROW_ID"].duplicated().any():
        raise ValueError("EDA requires unique, nonmissing ROW_ID")
    train["RENT_APPROVAL_DATE"] = pd.to_datetime(train["RENT_APPROVAL_DATE"], errors="raise")
    dates = train["RENT_APPROVAL_DATE"]
    if dates.isna().any() or not dates.eq(dates.dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Approval dates must be nonmissing first-of-month dates")
    target = pd.to_numeric(train[TARGET], errors="raise")
    if target.isna().any() or not np.isfinite(target).all() or not target.gt(0).all():
        raise ValueError("EDA requires positive, finite rent targets")
    train[TARGET] = target
    root = Path(output_dir)
    tables_root, figures_root = root / "tables", root / "figures"
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, by in [
        ("rent_by_month", ["RENT_APPROVAL_DATE"]), ("rent_by_flat_type", ["FLAT_TYPE"]),
        ("rent_by_town", ["TOWN"]), ("rent_by_year", ["APPROVAL_YEAR"]),
        ("rent_by_flat_type_month", ["FLAT_TYPE", "RENT_APPROVAL_DATE"]),
        ("rent_by_town_month", ["TOWN", "RENT_APPROVAL_DATE"]),
    ]:
        if by == ["APPROVAL_YEAR"]:
            train["APPROVAL_YEAR"] = train["RENT_APPROVAL_DATE"].dt.year
        tables[name] = _rent_summary(train, by)
    edges = np.arange(0, np.ceil(float(target.max()) / 250) * 250 + 250, 250)
    counts, boundaries = np.histogram(target, bins=edges)
    tables["rent_histogram"] = pd.DataFrame({"lower_sgd": boundaries[:-1], "upper_sgd": boundaries[1:], "n": counts})
    records = []
    for column in feature_columns("ALL"):
        if column not in train or not pd.api.types.is_numeric_dtype(train[column]):
            continue
        values = train[column]
        complete = values.notna() & np.isfinite(values)
        usable = complete.any() and values[complete].nunique() > 1 and target[complete].nunique() > 1
        records.append({"feature": column, "n": int(complete.sum()), "missing_or_nonfinite_rows": int((~complete).sum()),
                        "pearson_r": float(values[complete].corr(target[complete], method="pearson")) if usable else np.nan,
                        "spearman_rho": float(values[complete].corr(target[complete], method="spearman")) if usable else np.nan})
    tables["rent_correlations"] = pd.DataFrame(records).sort_values("spearman_rho", key=lambda x: x.abs(), ascending=False)
    summary = {
        "population": "All cleaned labelled training rows, including the shared validation months; descriptive EDA only",
        "training_rows": len(train), "training_start": str(train["RENT_APPROVAL_DATE"].min().date()),
        "training_end": str(train["RENT_APPROVAL_DATE"].max().date()),
        "target_units": "SGD per month", "target_median_sgd": float(target.median()),
        "target_mean_sgd": float(target.mean()), "target_min_sgd": float(target.min()),
        "target_max_sgd": float(target.max()), "low_rent_below_1000_rows": int(target.lt(1000).sum()),
        "duplicate_policy": "Shared preprocessing removes raw exact duplicates keeping first ROW_ID; EDA does not deduplicate again",
        "time_association": tables["rent_correlations"].loc[tables["rent_correlations"]["feature"].eq("MONTH_INDEX")].to_dict("records"),
        "feature_set_counts": {name: len(feature_columns(name)) for name in ["G0", "G0+G1", "G0+G1+G2", "ALL"]},
        "repeated_profiles": {},
    }
    if test is not None:
        test_dates = pd.to_datetime(test["RENT_APPROVAL_DATE"], errors="raise")
        summary.update(test_rows=len(test), test_start=str(test_dates.min().date()), test_end=str(test_dates.max().date()),
                       overlapping_train_test_months=sorted(set(train["RENT_APPROVAL_DATE"].dt.strftime("%Y-%m")) & set(test_dates.dt.strftime("%Y-%m"))))
    holdout = train["RENT_APPROVAL_DATE"].between(TIME_VALID_START, TIME_VALID_END)
    for population, frame in [("all_train", train), ("holdout", train.loc[holdout])]:
        for strict in [False, True]:
            findings, groups, sizes = repeated_profile_dispersion(frame, strict)
            label = f"{population}_{'strict' if strict else 'plan'}"
            summary["repeated_profiles"][label] = findings
            tables[f"repeated_profile_groups_{label}"] = groups
            tables[f"repeated_profile_sizes_{label}"] = sizes
    for name, table in tables.items():
        table.to_csv(tables_root / f"{name}.csv", index=False)
    _figures(train, tables, figures_root)
    return summary
