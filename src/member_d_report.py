"""Report text and provenance helpers for Member D diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

REFERENCE_RUN = "20261025-210800_lightgbm_tuned_G0+G1_time_index"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    """Avoid an unpinned tabulate dependency."""
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in table[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(rows)


def write_full_report(output, eda, residuals, interpretation, profile_comparison, manifest):
    """Compose concise report text from executed tables, with detail in an appendix."""
    output = Path(output)
    metric = residuals["metrics"]
    association = eda["time_association"][0]
    profiles = eda["repeated_profiles"]
    segments = pd.read_csv(output / "residuals/segment_metrics.csv")
    months = segments[segments.segment_type.eq("month")].sort_values("segment")
    types = segments[segments.segment_type.eq("flat_type")].sort_values("rmse", ascending=False)
    towns = segments[segments.segment_type.eq("town")].sort_values("squared_error_share", ascending=False)
    largest_type = types.iloc[0]
    largest_town = towns.iloc[0]
    correlations = pd.read_csv(output / "eda/tables/rent_correlations.csv")
    importance = pd.read_csv(output / "interpretation/permutation_importance.csv")
    joined = importance.merge(correlations, on="feature", how="left", validate="one_to_one")
    joined.to_csv(output / "correlation_vs_importance.csv", index=False)
    dispersion_rows = []
    for label, values in profiles.items():
        dispersion_rows.append({"population_profile": label,
            **{key: values[key] for key in ("input_rows", "excluded_missing_key_rows", "repeated_groups",
                "repeated_rows", "coverage_fraction_all_rows", "unweighted_mean_group_sample_sd_sgd",
                "row_weighted_rms_group_sample_sd_sgd", "pooled_within_group_sample_sd_sgd",
                "self_inclusive_group_mean_rmse_sgd", "leave_one_out_group_mean_rmse_sgd")}})
    dispersion = pd.DataFrame(dispersion_rows)
    dispersion.to_csv(output / "repeated_profile_summary.csv", index=False)
    # Missing repeated groups must stay missing, not become a zero-SGD floor.
    def amount(value):
        return "unavailable (no qualifying groups)" if value is None else f"SGD {value:.2f}"

    profile = profiles["all_train_plan"]
    holdout_profile = profiles["holdout_plan"]
    same_population = profile_comparison[0]
    annual = pd.read_csv(output / "eda/tables/rent_by_year.csv")
    monthly = pd.read_csv(output / "eda/tables/rent_by_month.csv")
    years = pd.to_datetime(monthly.RENT_APPROVAL_DATE).dt.year.value_counts()
    full_years = annual[annual.APPROVAL_YEAR.isin(years[years.eq(12)].index)].sort_values("APPROVAL_YEAR")
    trend_text = (f"Annual median rent rose from SGD {full_years.iloc[0]['median']:.0f} in "
                  f"{int(full_years.iloc[0].APPROVAL_YEAR)} to SGD {full_years.iloc[-1]['median']:.0f} in "
                  f"{int(full_years.iloc[-1].APPROVAL_YEAR)}. " if len(full_years) >= 2 else "")
    type_rents = pd.read_csv(output / "eda/tables/rent_by_flat_type.csv")
    town_rents = pd.read_csv(output / "eda/tables/rent_by_town.csv")
    rent_band = segments[segments.segment_type.eq("rent_band")]
    low = rent_band[rent_band.segment.isin(["<1000", "1000-<2000"])]
    low_text = (f"Retrospectively, true rents below SGD 2,000 form {100*low.n.sum()/metric['n']:.2f}% of rows "
                f"but {100*low.squared_error_share.sum():.2f}% of squared error; this low-rent tail is an "
                "observed limitation, not evidence that the records are wrong. " if len(low) else "")
    strict_same = all(profiles[f"{population}_plan"]["repeated_rows"] == profiles[f"{population}_strict"]["repeated_rows"]
                      and profiles[f"{population}_plan"]["repeated_groups"] == profiles[f"{population}_strict"]["repeated_groups"]
                      for population in ("all_train", "holdout"))
    strict_text = ("Adding flat model and lease year leaves group counts, coverage and dispersion unchanged."
                   if strict_same else "Adding flat model and lease year changes the profile definition; full sensitivity results are in the appendix.")
    month_importance = importance.loc[importance.feature.eq("MONTH_INDEX"), "importance_mean_sgd"]
    month_text = (f"Despite time's full-training Spearman association, its permutation effect in this "
                  f"six-month index-adjusted sample is only {month_importance.iloc[0]:.2f} SGD RMSE. "
                  if len(month_importance) else "")
    pdp = pd.read_csv(output / "interpretation/partial_dependence.csv")
    mrt_curves = pdp[(pdp.feature == "NEAREST_OPEN_MRT_KM") & (pdp.intervention == "standard")]
    curve_deviation = {}
    for model, part in mrt_curves.groupby("model"):
        if len(part) > 2 and part.value.nunique() > 1:
            line = np.polyval(np.polyfit(part.value, part.mean_prediction_sgd, 1), part.value)
            curve_deviation[model] = float(np.max(np.abs(part.mean_prediction_sgd - line)))
    curve_text = ("Boosting has a nonlinear MRT response while Ridge's response is linear over the inspected grid. "
                  if curve_deviation.get("primary", 0) > 1 and curve_deviation.get("ridge", np.inf) < 1e-6 else "")
    cases = pd.read_csv(output / "residuals/top20_errors.csv")
    outliers = pd.DataFrame(residuals["outliers"]["populations"])
    large_errors = outliers.loc[outliers.population.eq("Absolute error > SGD 1000")].iloc[0]
    large_error_text = (f"Absolute errors above SGD 1,000 occur in {int(large_errors.n):,} rows "
                        f"({large_errors.row_share_pct:.2f}%) and contribute "
                        f"{large_errors.squared_error_share_pct:.2f}% of squared error. ")
    case_text = (f"Of these, {int(cases.y_true.lt(1000).sum())} have true rent below SGD 1,000 and "
                 f"{int(cases.rare_segment_flag.sum())} belong to a town/type with fewer than 30 holdout rows. "
                 "These flags do not establish a cause. ")
    reviews = [("error cases", "residuals/case_review.md"),
               ("interpretation notes", "interpretation/interpretation_notes.md")]
    reviewed_links = [f"[{label}]({path})" for label, path in reviews if (output / path).is_file()]
    review_text = ("Detailed reviews are in " + " and ".join(reviewed_links) +
                   ". These retain supported evidence, the limits of plausible explanations and response-curve comparisons."
                   if reviewed_links else "Detailed case attributes and curve values are retained in the source CSVs.")
    top = importance.head(3)
    top_text = ", ".join(f"{r.feature} (+{r.importance_mean_sgd:.1f} SGD RMSE, repeat SD {r.importance_std_sgd:.1f})"
                         for r in top.itertuples())
    horizon_claim = ("increases monotonically" if months.rmse.is_monotonic_increasing
                     else "does not increase monotonically")
    audit_path = PROJECT_ROOT / "artifacts/data_audit/audit_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_matches = audit["cleaning"]["training_rows_after_exact_deduplication"] == eda["training_rows"]
    audit_note = (f"The saved audit records {audit['cleaning']['raw_training_rows']:,} original rows and "
                  f"{audit['cleaning']['exact_duplicate_training_rows_removed']:,} exact training duplicates removed, "
                  "keeping the first original ROW_ID. " if audit_matches else
                  "The present population differs from the saved cleaning audit; its raw/deduplicated counts are not reused here. ")
    overlap = ", ".join(eda["overlapping_train_test_months"]) or "none"
    eda_text = f"""The target is monthly rent in SGD for Singapore HDB flats. The cleaned labelled population contains {eda['training_rows']:,} rows from {eda['training_start']} to {eda['training_end']}. {audit_note}All {eda['test_rows']:,} test rows, including duplicate profiles, are retained in their original order, covering {eda['test_start']} to {eda['test_end']}. Overlapping approval months are {overlap}; this is predominantly future-month prediction rather than a perfectly disjoint calendar split.

Available attributes describe approval month, town, block/street, flat type and model, area and lease commencement. Furnishing and fee are constant and removed upstream. Normalized town/block/street identify the block for enrichment; the chosen G0+G1 input set has {len(manifest['columns'])} core and location columns. Missing unit storey and condition remain limitations. Unusual rents are retained: {eda['low_rent_below_1000_rows']:,} cleaned rows are below SGD 1,000, a descriptive threshold distinct from the team's suspicious-row flags.

Across all cleaned labelled rows, median rent is SGD {eda['target_median_sgd']:.0f}, mean SGD {eda['target_mean_sgd']:.0f}, and observed range SGD {eda['target_min_sgd']:.0f}–{eda['target_max_sgd']:.0f}. {trend_text}Flat-type medians range from SGD {type_rents['median'].min():.0f} to SGD {type_rents['median'].max():.0f}; town medians range from SGD {town_rents['median'].min():.0f} to SGD {town_rents['median'].max():.0f}. These descriptive comparisons mix approval dates, flat types and locations and do not isolate a town or size effect. Time has a strong monotonic association with rent (Spearman rho {association['spearman_rho']:.3f}; Pearson r {association['pearson_r']:.3f}) on {association['n']:,} complete rows. Pearson measures linear association; Spearman measures rank association. Neither establishes causation or exhausts nonlinear relationships, and shared trends and correlated attributes complicate rankings. Held-out permutation importance provides a complementary model-based nonlinear view."""
    error_text = f"""Using the shared October 2024–March 2025 holdout, the reproduced selected LightGBM configuration has RMSE SGD {metric['rmse']:.2f}, MAE SGD {metric['mae']:.2f}, MAPE {metric['mape']:.2f}% and bias SGD {metric['bias']:.2f} on {metric['n']:,} rows. Residual means prediction minus truth; positive values indicate overprediction. This holdout also guided tuning and model selection, so it is not an independent post-selection test.

RMSE {horizon_claim} from horizon 1 to 6 (SGD {months.iloc[0].rmse:.2f} to SGD {months.iloc[-1].rmse:.2f}); this window does not establish worsening trend drift. The highest-error flat type is {largest_type.segment} (n={int(largest_type.n):,}, RMSE SGD {largest_type.rmse:.2f}), contributing {100*largest_type.squared_error_share:.1f}% of squared error. The largest town contribution is {largest_town.segment}: {100*largest_town.squared_error_share:.1f}% from {int(largest_town.n):,} rows. {large_error_text}{low_text}Complete segment counts, biases and contributions are retained in the appendix assets.

For profiles defined by normalized block identity, flat type, exact area and approval month, {profile['repeated_groups']:,} repeated groups cover {profile['repeated_rows']:,} rows ({100*profile['coverage_fraction_all_rows']:.1f}% of cleaned training). Their unweighted mean group sample SD is {amount(profile['unweighted_mean_group_sample_sd_sgd'])}; pooled sample SD is {amount(profile['pooled_within_group_sample_sd_sgd'])}. On groups formed only inside the holdout, leave-one-out group-mean RMSE is {amount(holdout_profile['leave_one_out_group_mean_rmse_sgd'])}, versus model RMSE {amount(same_population['model_rmse_sgd'])} on the same {same_population['n']:,} rows. Small groups inflate leave-one-out error. This selected-subset dispersion is neither a proven universal RMSE floor nor a deployable validation model. {strict_text} The appendix distinguishes all SD/RMSE denominators and self-inclusion.

With identical fit rows, 39 inputs and index handling, Ridge RMSE is SGD {residuals['comparison']['rmse']:.2f}; boosting improves this by {100*(residuals['comparison']['rmse']-metric['rmse'])/residuals['comparison']['rmse']:.1f}%. Paired distance and town/type tables describe where predictions differ. {curve_text}The largest permutation effects are {top_text}, measured on {interpretation['sample_n']:,} fixed held-out rows over {interpretation['n_repeats']} repeats. SD describes repeat variability, not a confidence interval. {month_text}Correlated inputs can share or mask importance; rankings are not causal. The four PDPs describe model behavior. Area/age sensitivity curves update their engineered derivatives; month curves include index restoration but hold seasonal and age columns fixed. Implausible combinations remain a limitation.

The 20 largest absolute errors include {residuals['top20_overpredictions']} overpredictions and {residuals['top20_underpredictions']} underpredictions. {case_text}Separate signed extremes cover both directions. Case review retains observed metadata flags and unexplained cases; no price alone is called a confirmed error."""
    limitations = """Unrecorded storey, renovation, condition, facing, furnishing detail and tenancy context plausibly limit accuracy, but cannot be assigned as causes for specific records. Same-month CPI, COE and stock inputs may be unavailable at forecast time; the selected G0+G1 model excludes them. Static rail/POI snapshots do not establish historical opening dates. Repeated-profile groups may mix different units, omit recorded attributes and have very small sizes. Correlated inputs and implausible synthetic combinations limit interpretation. The six-month holdout does not establish behavior across the full test horizon to July 2026, and using it repeatedly for model selection introduces selection optimism."""
    appendix = f"""## Appendix and reproducibility

Source run: `{REFERENCE_RUN}`. The refit uses the stored tuned configuration, seed 5228, G0+G1, index targets and exactly the shared fit mask; Ridge uses `make_ridge`, not Member B's narrower feature selector. Reference verification: {manifest['reference_agreement']['level']}. See manifest.json for exact tolerances, timing, versions, input/code/config hashes and the generation command. Metric agreement alone does not prove original row-level prediction agreement.

### Repeated-profile denominators

{markdown_table(dispersion, list(dispersion.columns))}

Within repeated groups let SSE_g=sum((rent-group_mean)^2). Unweighted mean SD averages sample SD_g; row-weighted RMS SD is sqrt(sum(n_g*SD_g^2)/sum(n_g)); pooled SD is sqrt(sum(SSE_g)/sum(n_g-1)); self-inclusive RMSE is sqrt(sum(SSE_g)/sum(n_g)); leave-one-out RMSE is sqrt(sum(SSE_g*(n_g/(n_g-1))^2)/sum(n_g)). None are interchangeable. Missing/empty/malformed profile keys are excluded and counted; area is not rounded. Group and size-distribution tables cover full training and holdout under both definitions. No target-derived values enter the prediction features.

The all-training plan definition has {profile['all_groups']:,} eligible groups, including {profile['singleton_groups']:,} singletons; the holdout has {holdout_profile['all_groups']:,} groups including {holdout_profile['singleton_groups']:,} singletons. Missing-key exclusions are {profile['excluded_missing_key_rows']} and {holdout_profile['excluded_missing_key_rows']}, respectively. Median repeated-group sizes are {profile['median_group_size_repeated']} and {holdout_profile['median_group_size_repeated']}; maximum sizes are {profile['max_group_size']} and {holdout_profile['max_group_size']}. Detailed counts by group size are in eda/tables/repeated_profile_sizes_*.csv. These small groups explain why leave-one-out dispersion should not be read as a universal lower bound. The one-room EDA trajectory is sparse and its gaps are retained; sample counts are shown in the figures and tables.

### Additional figures and source tables

{markdown_table(outliers, ['population', 'n', 'row_share_pct', 'squared_error_share_pct', 'rmse_sgd', 'bias_sgd'])}

These overlapping diagnostic populations must not be added together. Rent-based cuts and removal of the largest errors are retrospective sensitivities, not revised validation scores. All holdout rows remain in the official metrics.

![Validation predictions and outliers](residuals/figures/validation_outliers.png)
![Monthly error and bias](residuals/figures/error_by_horizon.png)
![Held-out permutation importance](interpretation/permutation_importance.png)
![Rent by town](eda/figures/eda_town.png)
![Monthly town medians](eda/figures/eda_town_month.png)
![Monthly type medians](eda/figures/eda_flat_type_month.png)
![Residuals and calibration](residuals/figures/residual_distribution_calibration.png)
![Error by town and contribution](residuals/figures/town_errors_contribution.png)

{review_text}

All segment metrics are in residuals/segment_metrics.csv; bins and missing handling are in residuals/summary.json. Squared-error shares reconcile within each segmentation, which must not be summed across overlapping segmentations. True-rent bands are retrospective diagnostics, never inference-time inputs. Case attributes and evidence labels are in residuals/top20_errors.csv and both signed-extreme tables. Correlations versus model importance are aligned in correlation_vs_importance.csv; every repeat, sampled ROW_ID and PDP ordinate is retained under interpretation/. PDF figures are exported beside PNGs for the course LaTeX report. Main-report allocations remain approximately 1.5 pages for Data/EDA and 1.25 pages for Error Analysis/Interpretation; check the final integrated report against the eight-page limit. Names, student IDs, team name and the full workload table remain team-level completion checks; the final deadline is unspecified.
"""
    report = ("# Member D: data, error analysis and interpretation\n\n## Data and EDA\n\n" + eda_text +
        "\n\n![Rent distribution and monthly medians](eda/figures/eda_distribution_trend.png)\n\n"
        "![Rent by flat type](eda/figures/eda_flat_type.png)\n\n## Error analysis and interpretation\n\n" + error_text +
        "\n\n![Partial dependence and sensitivity](interpretation/partial_dependence.png)\n\n"
        "## Limitations contribution\n\n" + limitations + "\n\n" + appendix)
    (output / "report.md").write_text(report, encoding="utf-8")
    # Pasteable prose fragments preserve the same values as the reviewed Markdown.
    def latex(text):
        replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "–": "--"}
        return "".join(replacements.get(char, char) for char in text)
    sections = ("% Generated prose fragments; figures and appendix tables are separate assets.\n"
                "\\section{Data and EDA}\n" + latex(eda_text) +
                "\n\n\\section{Error Analysis and Interpretation}\n" + latex(error_text) +
                "\n\n% Contribution to the shared limitations section\n" + latex(limitations) + "\n")
    (output / "sections.tex").write_text(sections, encoding="utf-8")
