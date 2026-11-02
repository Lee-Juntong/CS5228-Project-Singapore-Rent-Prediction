"""Decision 5 (trend handling) and the split comparison, built from logged runs.

Reads results/results.csv and results/breakdowns/, refits nothing except the
monthly rent index, and writes:

* results/member_a/trend/*.csv           tables behind the report
* report/tables/{trend,split}.tex        LaTeX tables for the Methods section
* report/figures/{rent_index,monthly_rmse}.pdf

Run the experiments first, for example:
    python -m scripts.run_experiment --model ridge random_forest --target-mode raw index --member A
then:
    python -m scripts.trend_analysis
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, TARGET
from src.evaluation import BREAKDOWNS_DIR, RESULTS_CSV, MonthlyRentIndex, split_comparison, time_holdout_masks
from src.features import load_modeling_data

OUT = PROJECT_ROOT / "results" / "member_a" / "trend"
TABLES = PROJECT_ROOT / "report" / "tables"
FIGURES = PROJECT_ROOT / "report" / "figures"

# Display names, in the order of the baseline ladder.
MODELS = {
    "global_median": "Global median",
    "group_median": "Group median",
    "linear_regression": "Linear regression",
    "ridge": "Ridge",
    "knn": "KNN",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "lightgbm_default": "LightGBM (default)",
    "catboost_default": "CatBoost (default)",
    "lightgbm_tuned": "LightGBM (tuned)",
    "catboost_tuned": "CatBoost (tuned)",
}
MONTHLY_PLOT = ["ridge", "random_forest", "lightgbm_tuned"]


def latest_time_runs(results: pd.DataFrame) -> pd.DataFrame:
    runs = results[(results["split"] == "time") & (results["feature_set"] == "G0+G1")
                   & results["model"].isin(MODELS)]
    return runs.sort_values("timestamp").groupby(["model", "target_mode"]).tail(1)


def trend_table(runs: pd.DataFrame) -> pd.DataFrame:
    wide = runs.pivot(index="model", columns="target_mode", values=["rmse", "bias"])
    wide.columns = [f"{metric}_{mode}" for metric, mode in wide.columns]
    wide = wide.reindex([m for m in MODELS if m in wide.index]).dropna()
    wide["change"] = wide["rmse_index"] - wide["rmse_raw"]
    return wide.reset_index()


def monthly_table(runs: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for run in runs.itertuples():
        path = BREAKDOWNS_DIR / f"{run.run_id}.csv"
        if not path.exists():
            continue
        months = pd.read_csv(path).query("segment_type == 'month'")
        parts.append(months.assign(model=run.model, target_mode=run.target_mode)
                     [["model", "target_mode", "segment", "n", "rmse", "bias"]].rename(columns={"segment": "month"}))
    return pd.concat(parts, ignore_index=True)


def rent_index_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Observed monthly medians next to the index fitted on the fit window only."""

    fit, _ = time_holdout_masks(train)
    index = MonthlyRentIndex().fit(train.loc[fit, "MONTH_INDEX"], train.loc[fit, TARGET])
    observed = train.groupby("MONTH_INDEX")[TARGET].median()
    months = pd.Series(observed.index)
    return pd.DataFrame({
        "month_index": observed.index,
        "month": pd.to_datetime(train.groupby("MONTH_INDEX")["RENT_APPROVAL_DATE"].min().to_numpy()).strftime("%Y-%m"),
        "observed_median": observed.to_numpy(),
        "index": index.transform(months),
        "in_fit": observed.index <= index.last_month_,
    })


def write_latex(trend: pd.DataFrame, split: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lrrr}", r"\hline",
             r"\textbf{Model} & \textbf{Raw} & \textbf{Index} & \textbf{Change} \\", r"\hline"]
    for row in trend.itertuples():
        lines.append(f"{MODELS[row.model]} & {row.rmse_raw:.1f} & {row.rmse_index:.1f} & {row.change:+.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (TABLES / "trend.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{lrrr}", r"\hline",
             r"\textbf{Model} & \textbf{Time} & \textbf{Random} & \textbf{Gap} \\", r"\hline"]
    for row in split.itertuples():
        name = MODELS[row.model] + (" + index" if row.target_mode == "index" else "")
        lines.append(f"{name} & {row.rmse_time:.1f} & {row.rmse_random5_window:.1f} & {row.optimism:+.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (TABLES / "split.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(index_frame: pd.DataFrame, monthly: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8})

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = index_frame["month_index"]
    ax.plot(x, index_frame["observed_median"], "o", ms=2.5, color="0.35", label="Observed monthly median")
    fit = index_frame["in_fit"]
    ax.plot(x[fit], index_frame["index"][fit], color="tab:blue", lw=1, label="Index (fit window)")
    ax.plot(x[~fit], index_frame["index"][~fit], color="tab:orange", lw=1.5, label="Index (extrapolated)")
    ax.axvline(index_frame.loc[fit, "month_index"].max() + 0.5, color="0.6", ls=":", lw=1)
    ticks = index_frame[index_frame["month"].str.endswith("-01")]
    ax.set_xticks(ticks["month_index"], ticks["month"].str[:4])
    ax.set_ylabel("Monthly rent (SGD)")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "rent_index.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharex=True)
    colors = dict(zip(MONTHLY_PLOT, ["tab:red", "tab:green", "tab:blue"]))
    for model in MONTHLY_PLOT:
        for mode, style in (("raw", "--"), ("index", "-")):
            part = monthly[(monthly["model"] == model) & (monthly["target_mode"] == mode)].sort_values("month")
            if part.empty:
                continue
            label = f"{MODELS[model]}, {mode}"
            axes[0].plot(part["month"], part["rmse"], style, color=colors[model], marker="o", ms=2.5, lw=1, label=label)
            axes[1].plot(part["month"], part["bias"], style, color=colors[model], marker="o", ms=2.5, lw=1)
    axes[0].set_ylabel("RMSE (SGD)")
    axes[1].set_ylabel("Mean prediction minus rent (SGD)")
    axes[1].axhline(0, color="0.6", lw=0.8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "monthly_rmse.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS_CSV)
    runs = latest_time_runs(results)

    trend = trend_table(runs)
    monthly = monthly_table(runs)
    split = split_comparison(results)
    split = split[(split["feature_set"] == "G0+G1") & split["model"].isin(MODELS)]
    split = split.assign(order=split["model"].map(list(MODELS).index)).sort_values(["order", "target_mode"], ascending=[True, False]).drop(columns="order")

    train, _ = load_modeling_data()
    index_frame = rent_index_frame(train)

    trend.to_csv(OUT / "trend_table.csv", index=False)
    monthly.to_csv(OUT / "monthly_rmse.csv", index=False)
    split.to_csv(OUT / "split_comparison.csv", index=False)
    index_frame.to_csv(OUT / "rent_index.csv", index=False)
    write_latex(trend, split)
    plot(index_frame, monthly)

    print(trend.round(1).to_string(index=False))
    print()
    print(split.round(1).to_string(index=False))
    valid = index_frame[~index_frame["in_fit"]]
    gap = valid["index"] - valid["observed_median"]
    print(f"\nIndex minus observed median in validation months: mean {gap.mean():.1f}, range {gap.min():.1f} to {gap.max():.1f}")


if __name__ == "__main__":
    main()
