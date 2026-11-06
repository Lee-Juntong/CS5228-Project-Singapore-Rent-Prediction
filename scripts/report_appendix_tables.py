"""Build the four extended-experiment appendix tables from saved results."""
import pandas as pd

from scripts.report_tables import MODELS, ROW_END, TABLES, signed, table
from src.config import RESULTS_DIR


def main():
    source = RESULTS_DIR / "member_b" / "extensions"
    models = ["global_median", "group_median", "linear_regression", "ridge",
              "knn", "decision_tree", "random_forest"]
    horizon = pd.read_csv(source / "horizon.csv")
    horizon[["origin", "band"]] = horizon["case"].str.split("/", expand=True)
    windows = horizon.pivot(index=["model", "band", "mode"], columns="origin", values="rmse")
    if windows.shape != (28, 5) or windows.isna().any().any():
        raise ValueError("Expected five complete training cutoffs for each baseline, band and target.")
    means = windows.mean(axis=1)
    rows = []
    for name in models:
        values = [means.loc[name, band, mode] for band in ("1-6", "7-16") for mode in ("raw", "index")]
        rows.append(" & ".join([MODELS[name], *(f"{x:.1f}" for x in values)]) + ROW_END)
    header = (r" & \multicolumn{2}{c}{\textbf{Near}} & \multicolumn{2}{c}{\textbf{Far}}" + ROW_END
              + "\n" + r"\textbf{Model} & Raw & Index & Raw & Index")
    (TABLES / "baseline_horizon.tex").write_text(table("lrrrr", header, rows), encoding="utf-8")

    lag = pd.read_csv(source / "lags.csv").pivot(index="model", columns="case", values="rmse")
    rows = []
    for name in models[2:]:
        base, augmented = lag.loc[name, ["base", "plus16"]]
        rows.append(f"{MODELS[name]} & {base:.1f} & {augmented:.1f} & {signed(augmented-base)}" + ROW_END)
    header = r"\textbf{Model} & \shortstack{Without rent\\history} & \shortstack{With rent\\history} & Change"
    (TABLES / "baseline_lags.tex").write_text(table("lrrr", header, rows), encoding="utf-8")

    macro = pd.read_csv(source / "macro_index" / "median_metrics.csv")
    scores = macro.pivot(index="method", columns="protocol", values="rmse")
    methods = {"linear12": "Original 12-month linear index", "anchored_history": "Recency-weighted history",
               "anchored_macro": "Recency-weighted history + GDP/CPI", "macro": "Ridge with GDP/CPI forecasts"}
    rows = [f"{label} & {scores.loc[key, 'fixed']:.1f} & {scores.loc[key, 'monthly_update']:.1f}" + ROW_END
            for key, label in methods.items()]
    header = r"\textbf{Index method} & \shortstack{Fixed at\\Sep 2024} & \shortstack{Updated\\monthly}"
    (TABLES / "macro_medians.tex").write_text(table("@{}lrr@{}", header, rows), encoding="utf-8")

    audit = pd.read_csv(source / "flag_threshold_audit.csv").set_index("flag")
    rules = {"m2": r"Low rent for floor area (residual $z<-3$ of a rent\,$\sim$\,area line)",
             "m1": r"Extreme within its flat type or flat type\,$\times$\,town group (IQR fence and $|z|>3$)",
             "both": "Flagged by both rules"}
    rows = [f"{label} & {audit.loc[flag, 'all_rows_flags']} & {audit.loc[flag, 'fit_only_threshold_drops']}" + ROW_END
            for flag, label in rules.items()]
    header = r"\textbf{Rule} & \shortstack{\textbf{All}\\\textbf{rows}} & \shortstack{\textbf{Fit}\\\textbf{rows}}"
    (TABLES / "outliers.tex").write_text(table(r"@{}p{0.62\columnwidth}rr@{}", header, rows), encoding="utf-8")


if __name__ == "__main__":
    main()
