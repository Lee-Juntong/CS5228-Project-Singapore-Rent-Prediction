"""Compact report and file helpers for rent analysis."""

import hashlib
import json
import os
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
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in table[columns].itertuples(index=False, name=None):
        cells = ["Unavailable" if pd.isna(v) else f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
                 for v in values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_full_report(output, eda, residuals, interpretation, profile_comparison, manifest):
    """Summarize calculated tables; leave detailed explanations to the reviewed notes."""
    output = Path(output)
    metrics = pd.DataFrame([{"model": "LightGBM", **residuals["metrics"]},
                            {"model": "Matched Ridge", **residuals["comparison"]}])
    importance = pd.read_csv(output / "interpretation/permutation_importance.csv")
    correlations = pd.read_csv(output / "eda/tables/rent_correlations.csv")
    importance.merge(correlations, on="feature", how="left", validate="one_to_one").to_csv(
        output / "correlation_vs_importance.csv", index=False)
    profile_fields = ["input_rows", "excluded_missing_key_rows", "repeated_groups", "repeated_rows",
                      "coverage_fraction_all_rows", "unweighted_mean_group_sample_sd_sgd",
                      "row_weighted_rms_group_sample_sd_sgd", "pooled_within_group_sample_sd_sgd",
                      "self_inclusive_group_mean_rmse_sgd", "leave_one_out_group_mean_rmse_sgd"]
    profiles = pd.DataFrame([{"population_profile": label, **{key: values[key] for key in profile_fields}}
                             for label, values in eda["repeated_profiles"].items()])
    profiles.to_csv(output / "repeated_profile_summary.csv", index=False)
    association = eda["time_association"][0]
    eda_text = (f"Cleaned training: {eda['training_rows']:,} rows, {eda['training_start']} to {eda['training_end']}. "
                f"Test: {eda['test_rows']:,} rows, {eda['test_start']} to {eda['test_end']}. "
                f"Overlapping months: {', '.join(eda['overlapping_train_test_months']) or 'none'}. "
                f"Median rent is SGD {eda['target_median_sgd']:.0f}; observed range is SGD "
                f"{eda['target_min_sgd']:.0f}–{eda['target_max_sgd']:.0f}. "
                f"There are {eda['low_rent_below_1000_rows']:,} rents below SGD 1,000. "
                f"Time has Spearman rho {association['spearman_rho']:.3f} on {association['n']:,} rows.")
    error_text = (f"LightGBM holdout RMSE is SGD {residuals['metrics']['rmse']:.2f}, compared with "
                  f"SGD {residuals['comparison']['rmse']:.2f} for matched Ridge. Both use identical fit rows, "
                  "G0+G1 inputs and index targets. Residuals are prediction minus truth; positive means overprediction.")
    report = ["# Rent analysis", "## Data and EDA", eda_text,
              "[EDA source tables](eda/tables/rent_by_month.csv)", "## Validation errors", error_text,
              markdown_table(metrics, ["model", "n", "rmse", "mae", "mape", "bias"]),
              "RMSE, MAE and bias are SGD; MAPE is a percentage. "
              "[Segments](residuals/segment_metrics.csv) · [Largest errors](residuals/top20_errors.csv) · "
              "[Signed underpredictions](residuals/largest_underpredictions.csv)",
              "## Repeated-profile dispersion",
              markdown_table(profiles, ["population_profile", "repeated_groups", "repeated_rows",
                  "pooled_within_group_sample_sd_sgd", "leave_one_out_group_mean_rmse_sgd"]),
              "Profiles group normalized block identity, flat type, exact area and approval month; strict profiles "
              "also include flat model and lease year. Spread uses groups with at least two rows. Pooled sample SD "
              "and leave-one-out group-mean RMSE have different denominators. Leave-one-out uses other holdout labels; "
              "it is descriptive, not a deployable model or a proven universal error floor.",
              "[Full dispersion summary](repeated_profile_summary.csv) · "
              "[Model RMSE on the same repeated holdout rows](profile_model_comparison.csv)",
              "Model performance on those repeated holdout rows:",
              markdown_table(pd.DataFrame(profile_comparison), ["profile", "n", "model_rmse_sgd"]),
              "## Model interpretation",
              markdown_table(importance.head(10), ["feature", "importance_mean_sgd", "importance_std_sgd"]),
              f"Permutation importance is the increase in SGD RMSE on {interpretation['sample_n']:,} fixed holdout "
              f"rows over {interpretation['n_repeats']} repeats. SD describes permutation variation, not a confidence "
              "interval. Correlated inputs can share importance. PDPs describe fitted predictions, not causal effects; "
              "area/age sensitivity updates engineered derivatives, while month curves hold seasonality and age fixed.",
              "[Correlations versus importance](correlation_vs_importance.csv) · "
              "[Partial-dependence values](interpretation/partial_dependence.csv)", "## Figures"]
    for title, path in [("Rent distribution and trend", "eda/figures/eda_distribution_trend"),
                        ("Rent by flat type", "eda/figures/eda_flat_type"),
                        ("Error by forecast month", "residuals/figures/error_by_horizon"),
                        ("Validation outliers", "residuals/figures/validation_outliers"),
                        ("Permutation importance", "interpretation/permutation_importance"),
                        ("Partial dependence", "interpretation/partial_dependence")]:
        report.append(f"![{title}]({path}.png)")
    report.extend(["## Limits and reproduction",
                   "The holdout also guided tuning and selection, and covers only six forecast months. "
                   "Unrecorded unit and tenancy attributes are plausible limitations, not established case causes. "
                   "Unusual rents remain in the metrics. Static POI snapshots and hypothetical PDP combinations limit interpretation.",
                   f"Reference: `{REFERENCE_RUN}`. Agreement: {manifest['reference_agreement']['level']}. "
                   "Aggregate agreement does not establish identical original row predictions. "
                   "[Run details](manifest.json) record the command, environment and configuration. "
                   "Figures also have PDF exports; detailed case explanations require manual review."])
    notes = PROJECT_ROOT / "docs/analysis.md"
    if notes.is_file():
        report.append(f"[Reviewed findings]({os.path.relpath(notes, output).replace(os.sep, '/')})")
    (output / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    # Small pasteable handoff; tables, figures and reviewed explanations stay separate.
    prose = eda_text + "\n\n" + error_text
    for character, replacement in [("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("–", "--")]:
        prose = prose.replace(character, replacement)
    (output / "sections.tex").write_text("% Generated summary; add reviewed interpretation and figures.\n" +
                                         "\\section{Data and error analysis}\n" + prose + "\n", encoding="utf-8")
