"""ID-aligned, descriptive Member D diagnostics for the shared time holdout.

No fitting, feature selection, calibration or target-derived prediction features
are performed here. Residuals are prediction minus truth; positive means high.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import TARGET, TIME_FIT_END
from src.evaluation import regression_metrics, time_holdout_masks

MISSING = "<missing>"
BIN_DEFINITIONS = {
    "rent_band": "True rent SGD: (0,1000), [1000,2000), [2000,3000), "
                 "[3000,4000), [4000,5000), [5000,infinity). Retrospective only.",
    "flat_age": "Years: (-infinity,0) invalid; [0,10), [10,20), [20,30), "
                "[30,40), [40,50), [50,infinity); missing/nonfinite age is <missing>.",
    "horizon": "Calendar months since shared TIME_FIT_END: October 2024 = 1, March 2025 = 6.",
    "mrt_distance": "Nearest open MRT km: [0,0.5), [0.5,1), [1,2), [2,infinity); "
                    "negative distance is invalid, missing/nonfinite is <missing>.",
    "categories": "Missing or blank town/flat type is <missing>; no rows are excluded.",
    "rare_segment": "Town x flat type has fewer than 30 rows in this holdout; descriptive flag, not a cause.",
    "calibration": "Up to 10 empirical quantile bins of held-out predictions; duplicate edges are dropped; "
                   "constant predictions use one bin. Bins are descriptive, with no fitted recalibration.",
}


def _train_frame(data):
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, tuple):
        return data[0]
    return data.train


def align_holdout_predictions(data, predictions: pd.DataFrame) -> pd.DataFrame:
    """Require one prediction/truth per exact holdout ROW_ID, then join by ID.

    Metadata order is preserved, but neither prediction order nor frame indices
    are used to identify observations. This rejects random-CV files with extra
    dates; the caller must also establish training provenance of supplied files.
    """
    train = _train_frame(data).copy()
    needed = {"ROW_ID", TARGET, "RENT_APPROVAL_DATE"}
    if not needed <= set(train):
        raise ValueError(f"Missing metadata columns: {sorted(needed - set(train))}")
    if not {"ROW_ID", "y_true", "y_pred"} <= set(predictions):
        raise ValueError("Predictions require ROW_ID, y_true, y_pred")
    for label, frame in (("metadata", train), ("predictions", predictions)):
        if frame.ROW_ID.isna().any() or frame.ROW_ID.duplicated().any():
            raise ValueError(f"{label} ROW_ID must be nonmissing and unique")
    dates = pd.to_datetime(train.RENT_APPROVAL_DATE, errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Metadata approval dates must be nonmissing first-of-month dates")
    train["RENT_APPROVAL_DATE"] = dates
    _, valid = time_holdout_masks(train)
    metadata = train.loc[valid].copy()
    if metadata.empty:
        raise ValueError("No rows in the shared time holdout")
    if set(metadata.ROW_ID) != set(predictions.ROW_ID):
        raise ValueError("Prediction IDs must exactly match all shared time-holdout IDs")
    if "fold" in predictions and not predictions.fold.eq("time").all():
        raise ValueError("Only shared time-holdout predictions may be analyzed")
    aligned = metadata.merge(predictions[["ROW_ID", "y_true", "y_pred"]],
                             on="ROW_ID", how="left", sort=False, validate="one_to_one")
    if len(aligned) != len(metadata):
        raise ValueError("Prediction join changed the holdout row count")
    truth = aligned[TARGET].to_numpy(float)
    if not np.allclose(truth, aligned.y_true.to_numpy(float), rtol=0, atol=1e-8):
        raise ValueError("Saved y_true does not match metadata target at its ROW_ID")
    regression_metrics(truth, aligned.y_pred)  # shared finite/positive/shape checks
    aligned["residual"] = aligned.y_pred - aligned.y_true
    aligned["absolute_error"] = aligned.residual.abs()
    aligned["squared_error"] = aligned.residual**2
    return aligned


def _categories(series):
    values = series.astype("string").str.strip()
    return values.mask(values.eq(""), pd.NA).fillna(MISSING)


def _bands(series, bins, labels):
    values = pd.to_numeric(series, errors="raise")
    values = values.where(np.isfinite(values))
    return pd.cut(values, bins=bins, labels=labels, right=False).astype("string").fillna(MISSING)


def diagnostic_segments(aligned: pd.DataFrame) -> pd.DataFrame:
    """Mutually exclusive segments, including missing values, for every row."""
    required = {"TOWN", "FLAT_TYPE", "FLAT_AGE_YEARS"}
    if not required <= set(aligned):
        raise ValueError(f"Missing segment columns: {sorted(required - set(aligned))}")
    out = pd.DataFrame(index=aligned.index)
    out["town"] = _categories(aligned.TOWN)
    out["flat_type"] = _categories(aligned.FLAT_TYPE)
    out["rent_band"] = _bands(aligned.y_true, [0, 1000, 2000, 3000, 4000, 5000, np.inf],
                               ["<1000", "1000-<2000", "2000-<3000", "3000-<4000", "4000-<5000", "5000+"])
    out["flat_age"] = _bands(aligned.FLAT_AGE_YEARS, [-np.inf, 0, 10, 20, 30, 40, 50, np.inf],
                              ["<0 (invalid)", "0-<10", "10-<20", "20-<30", "30-<40", "40-<50", "50+"])
    out["month"] = aligned.RENT_APPROVAL_DATE.dt.strftime("%Y-%m")
    out["horizon"] = ((aligned.RENT_APPROVAL_DATE.dt.year - TIME_FIT_END.year) * 12
                       + aligned.RENT_APPROVAL_DATE.dt.month - TIME_FIT_END.month).astype(str)
    return out


def summarize_segments(aligned: pd.DataFrame, segments: pd.DataFrame | None = None) -> pd.DataFrame:
    """SSE shares add to one within each segmentation (unless all errors zero)."""
    segments = diagnostic_segments(aligned) if segments is None else segments
    total_sse = float(aligned.squared_error.sum())
    rows = []
    for kind in segments:
        for segment, index in segments.groupby(kind, sort=True, dropna=False).groups.items():
            part = aligned.loc[index]
            sse = float(part.squared_error.sum())
            rows.append({"segment_type": kind, "segment": str(segment), "n": len(part),
                         **regression_metrics(part.y_true, part.y_pred), "squared_error_sum": sse,
                         "row_share": len(part) / len(aligned),
                         "squared_error_share": sse / total_sse if total_sse else 0.0,
                         "contribution_to_overall_mse": sse / len(aligned)})
    table = pd.DataFrame(rows)
    for _, part in table.groupby("segment_type"):
        if part.n.sum() != len(aligned) or not np.isclose(part.squared_error_sum.sum(), total_sse):
            raise AssertionError("Segment counts or squared errors do not reconcile")
    return table


def _top_errors(aligned, segments):
    cases = aligned.copy()
    counts = segments.groupby(["town", "flat_type"], dropna=False).size()
    cases["town_flat_type_holdout_n"] = [int(counts.loc[(town, flat)])
                                         for town, flat in zip(segments.town, segments.flat_type)]
    cases["rare_segment_flag"] = cases.town_flat_type_holdout_n < 30
    issues = pd.Series("", index=cases.index, dtype="string")
    if "INVALID_LEASE_DATE_FLAG" in cases:
        issues.loc[cases.INVALID_LEASE_DATE_FLAG.fillna(0).eq(1)] += "invalid lease-date flag; "
    if "FLOOR_AREA_SQM" in cases:
        area = pd.to_numeric(cases.FLOOR_AREA_SQM, errors="raise")
        issues.loc[area.notna() & ((area <= 0) | ~np.isfinite(area))] += "nonpositive/nonfinite area; "
        issues.loc[area.isna()] += "missing observed area; "
    issues.loc[segments.town.eq(MISSING) | segments.flat_type.eq(MISSING)] += "missing observed town/type; "
    cases["observed_metadata_issue"] = issues.str.rstrip("; ")
    cases["low_rent_under_1000_flag"] = cases.y_true < 1000
    cases["review_category"] = np.select(
        [cases.observed_metadata_issue.ne(""), cases.rare_segment_flag],
        ["observed metadata issue; effect unestablished", "rare observed segment; cause unexplained"],
        default="unexplained; no confirmed data error")
    cases["error_direction"] = np.where(cases.residual > 0, "overprediction",
                                        np.where(cases.residual < 0, "underprediction", "exact"))
    columns = ["ROW_ID", "RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "ADDRESS_KEY",
               "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FLAT_AGE_YEARS",
               "INVALID_LEASE_DATE_FLAG", "NEAREST_OPEN_MRT_KM", "DISTANCE_TO_RAFFLES_PLACE_KM",
               "y_true", "y_pred", "residual", "absolute_error", "squared_error", "error_direction",
               "town_flat_type_holdout_n", "rare_segment_flag", "low_rent_under_1000_flag",
               "observed_metadata_issue", "review_category"]
    return cases[[c for c in columns if c in cases]].sort_values(
        ["absolute_error", "ROW_ID"], ascending=[False, True], kind="stable")


def _save_figure(fig, path):
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_outlier_diagnostics(aligned, segment_table, output_dir):
    """Describe error concentration using the already ID-aligned holdout.

    No rows are removed from the reported validation score. True-rent subsets
    and trimming are retrospective sensitivities, not revised model scores.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    sse = float(aligned.squared_error.sum())
    large = aligned.absolute_error > 1000
    low = aligned.y_true < 2000
    removed = aligned.sort_values(["absolute_error", "ROW_ID"], ascending=[False, True]).head(20).ROW_ID
    retained = ~aligned.ROW_ID.isin(removed)
    rows = []
    for name, mask in [("All validation rows", np.ones(len(aligned), dtype=bool)),
                       ("Absolute error > SGD 1000", large),
                       ("True rent < SGD 2000", low),
                       ("True rent >= SGD 2000 (retrospective)", ~low),
                       (f"Top {len(removed)} absolute errors removed (retrospective)", retained)]:
        part = aligned.loc[mask]
        rows.append({"population": name, "n": len(part), "row_share_pct": 100 * len(part) / len(aligned),
                     "squared_error_share_pct": 100 * float(part.squared_error.sum()) / sse if sse else 0.0,
                     "rmse_sgd": float(np.sqrt(part.squared_error.mean())) if len(part) else None,
                     "bias_sgd": float(part.residual.mean()) if len(part) else None})
    pd.DataFrame(rows).to_csv(output / "outlier_summary.csv", index=False)
    bands = segment_table.loc[segment_table.segment_type.eq("rent_band")].set_index("segment")
    order = ["<1000", "1000-<2000", "2000-<3000", "3000-<4000", "4000-<5000", "5000+"]
    bands = bands.loc[[band for band in order if band in bands.index]].reset_index()
    if bands.n.sum() != len(aligned) or not np.isclose(bands.squared_error_sum.sum(), sse):
        raise AssertionError("Rent-band counts or squared errors do not reconcile")
    blue, orange = "#326b91", "#b76d24"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].scatter(aligned.loc[~low, "y_true"], aligned.loc[~low, "y_pred"], s=5, alpha=.16,
                    color=blue, label="Observed rent >= SGD 2,000", rasterized=True)
    axes[0].scatter(aligned.loc[low, "y_true"], aligned.loc[low, "y_pred"], s=12, marker="x", alpha=.65,
                    color=orange, label="Observed rent < SGD 2,000", linewidths=.6)
    limit = max(aligned.y_true.max(), aligned.y_pred.max()) + 100
    axes[0].plot([0, limit], [0, limit], "--", color="#444444", lw=1, label="Perfect prediction")
    axes[0].set(xlim=(0, limit), ylim=(min(0, aligned.y_pred.min()), limit),
                xlabel="Observed monthly rent (SGD)", ylabel="Predicted monthly rent (SGD)",
                title=f"Validation predictions (n={len(aligned):,})")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    positions = np.arange(len(bands))
    for offset, values, color, label in [(-.18, 100 * bands.row_share, "#999999", "Share of rows"),
                                        (.18, 100 * bands.squared_error_share, blue, "Share of squared error")]:
        bars = axes[1].barh(positions + offset, values, height=.34, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axes[1].set_yticks(positions, [f"{r.segment}\n(n={r.n:,})" for r in bands.itertuples()])
    axes[1].invert_yaxis()
    maximum_share = 100 * max(bands.row_share.max(), bands.squared_error_share.max())
    axes[1].set(xlim=(0, min(100, 1.18 * maximum_share + 1)),
                xlabel="Share of all validation rows / squared error (%)",
                title="Error concentration by observed rent (SGD)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, figures / "validation_outliers")
    return {"large_error_definition": "Absolute residual strictly greater than SGD 1000",
            "populations": rows,
            "caveat": "True-rent subsets and trimming are retrospective sensitivities. All holdout rows "
                      "remain in the reported score; large residuals do not confirm recording errors."}


def _plots(aligned, table, calibration, output):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(aligned.residual, bins=50, color="#3978a8", edgecolor="white", linewidth=0.3)
    axes[0].axvline(0, color="#444444", ls="--", lw=1)
    axes[0].set(xlabel="Prediction minus truth (SGD)", ylabel="Holdout rows",
                title=f"Residual distribution (n={len(aligned):,})")
    axes[1].plot(calibration.mean_prediction, calibration.mean_truth, "o-", color="#3978a8",
                 label="Predicted-rent bins")
    bound = [min(calibration.mean_prediction.min(), calibration.mean_truth.min()),
             max(calibration.mean_prediction.max(), calibration.mean_truth.max())]
    axes[1].plot(bound, bound, "--", color="#444444", label="Equal means")
    axes[1].set(xlabel="Mean prediction (SGD)", ylabel="Mean true rent (SGD)", title="Held-out calibration")
    axes[1].legend(fontsize=8)
    _save_figure(fig, figures / "residual_distribution_calibration")

    horizon = table.loc[table.segment_type.eq("horizon")].copy()
    horizon["horizon"] = horizon.segment.astype(int)
    horizon = horizon.sort_values("horizon")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, metric, label in zip(axes, ("rmse", "bias"), ("RMSE (SGD)", "Mean prediction minus truth (SGD)")):
        ax.plot(horizon.horizon, horizon[metric], "o-", color="#3978a8")
        for row in horizon.itertuples():
            ax.annotate(f"n={row.n:,}", (row.horizon, getattr(row, metric)), xytext=(0, 7),
                        textcoords="offset points", fontsize=7, ha="center")
        ax.set(xlabel="Months ahead of September 2024", ylabel=label, xticks=horizon.horizon)
        ax.grid(alpha=0.18)
        ax.margins(y=0.2)
        ax.axhline(0, color="#777777", lw=0.7)
        if metric == "rmse":
            ax.set_ylim(bottom=0)
    _save_figure(fig, figures / "error_by_horizon")

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    towns = table.loc[table.segment_type.eq("town")].sort_values("squared_error_share")
    labels = [f"{r.segment} (n={r.n:,})" for r in towns.itertuples()]
    axes[0].barh(labels, towns.rmse, color="#3978a8")
    axes[1].barh(labels, towns.squared_error_share * 100, color="#d67932")
    axes[0].set(xlabel="RMSE (SGD)", title="Error by town")
    axes[1].set(xlabel="Share of total squared error (%)", title="Contribution to holdout error")
    axes[1].tick_params(axis="y", labelleft=False)
    _save_figure(fig, figures / "town_errors_contribution")


def generate_residuals(data, predictions, output_dir, *, comparison_predictions=None):
    """Save holdout diagnostics and return a small JSON-serializable summary.

    The optional comparator must use matching features, target mode and fit rows;
    its training provenance is established by the runner, not inferred from rent
    predictions. Only IDs/targets/holdout coverage can be verified here.
    """
    aligned = align_holdout_predictions(data, predictions)
    segments = diagnostic_segments(aligned)
    table = summarize_segments(aligned, segments)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "segment_metrics.csv", index=False)
    metrics = {"n": len(aligned), **regression_metrics(aligned.y_true, aligned.y_pred)}
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    if not np.isfinite(metrics["r2"]):
        metrics["r2"] = None  # R-squared is undefined for constant targets.
    cases = _top_errors(aligned, segments)
    cases.head(20).to_csv(output / "top20_errors.csv", index=False)
    # Keep the actual absolute-error ranking intact even when one sign dominates.
    cases.loc[cases.residual > 0].head(10).to_csv(output / "largest_overpredictions.csv", index=False)
    cases.loc[cases.residual < 0].head(10).to_csv(output / "largest_underpredictions.csv", index=False)
    calibration = aligned.assign(predicted_rent_bin=pd.qcut(aligned.y_pred, q=10, duplicates="drop"))
    if calibration.predicted_rent_bin.isna().all():
        calibration["predicted_rent_bin"] = "all (constant predictions)"
    calibration = calibration.groupby("predicted_rent_bin", observed=True).agg(
        n=("ROW_ID", "size"), mean_prediction=("y_pred", "mean"), mean_truth=("y_true", "mean"),
        bias=("residual", "mean")).reset_index()
    calibration.to_csv(output / "calibration.csv", index=False)
    counts, edges = np.histogram(aligned.residual, bins=50)
    pd.DataFrame({"lower_sgd": edges[:-1], "upper_sgd": edges[1:], "n": counts}).to_csv(
        output / "residual_histogram.csv", index=False)
    summary = {"metrics": metrics, "residual_definition": "prediction minus truth; positive = overprediction",
               "bin_definitions": BIN_DEFINITIONS, "row_join_verified": True,
               "top20_overpredictions": int(cases.head(20).residual.gt(0).sum()),
               "top20_underpredictions": int(cases.head(20).residual.lt(0).sum()),
               "case_review_limit": "Categories are observed flags, not verified causes. Unrecorded unit floor, "
                                    "condition, facing, renovation and tenancy context are plausible omitted "
                                    "information; their effects cannot be assigned to these records."}
    summary["outliers"] = generate_outlier_diagnostics(aligned, table, output)
    if comparison_predictions is not None:
        comparison = align_holdout_predictions(data, comparison_predictions)
        # Both frames have been joined into metadata order; join again by ID explicitly.
        paired = aligned.merge(comparison[["ROW_ID", "y_pred"]].rename(columns={"y_pred": "comparison_pred"}),
                               on="ROW_ID", validate="one_to_one", sort=False)
        other = paired.copy()
        other["y_pred"] = other.comparison_pred
        other["residual"] = other.y_pred - other.y_true
        other["squared_error"] = other.residual**2
        other_segments = diagnostic_segments(other)
        other_segments["town_flat_type"] = other_segments.town + " | " + other_segments.flat_type
        if "NEAREST_OPEN_MRT_KM" in other:
            other_segments["mrt_distance"] = _bands(other.NEAREST_OPEN_MRT_KM,
                [-np.inf, 0, 0.5, 1, 2, np.inf], ["<0 (invalid)", "0-<0.5", "0.5-<1", "1-<2", "2+"])
        main_table = summarize_segments(paired, other_segments)
        other_table = summarize_segments(other, other_segments)
        paired_table = main_table.merge(other_table, on=["segment_type", "segment", "n"],
                                        suffixes=("_boosting", "_ridge"), validate="one_to_one")
        paired_table["rmse_ridge_minus_boosting"] = paired_table.rmse_ridge - paired_table.rmse_boosting
        paired_table["sse_ridge_minus_boosting"] = (paired_table.squared_error_sum_ridge
                                                    - paired_table.squared_error_sum_boosting)
        paired_table.to_csv(output / "boosting_vs_ridge_segments.csv", index=False)
        comparison_metrics = regression_metrics(other.y_true, other.y_pred)
        if not np.isfinite(comparison_metrics["r2"]):
            comparison_metrics["r2"] = None
        summary["comparison"] = {"n": len(paired), **comparison_metrics,
                                 "rmse_ridge_minus_boosting": comparison_metrics["rmse"] - metrics["rmse"],
                                 "paired_rows_lower_absolute_error_boosting": int(
                                     (paired.absolute_error < (paired.comparison_pred - paired.y_true).abs()).sum())}
    _plots(aligned, table, calibration, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
