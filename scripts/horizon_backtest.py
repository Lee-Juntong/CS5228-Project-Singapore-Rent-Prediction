"""Backtest the raw and index targets on earlier periods, up to 16 months ahead.

The time holdout has flat rents and a 6 month horizon, while the test set runs
up to 16 months past the training data. This script moves the end of the fit
window back so the forecast covers the 2022 and 2023 rent rise and the flattening
after it. Nothing here enters results/results.csv. It writes:

* results/member_a/trend/horizon_backtest.csv         one row per origin, horizon band, model and target
* report/tables/horizon.tex                           LaTeX table for the Limitations section

Takes about 15 minutes on Juntong's PC. Rebuild only the table from the CSV with
    python -m scripts.horizon_backtest --table-only
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.baselines import make_random_forest
from src.config import PROJECT_ROOT, TARGET
from src.evaluation import MonthlyRentIndex, fit_predict, regression_metrics
from src.features import feature_columns, load_modeling_data
from src.models import make_lightgbm_tuned, make_xgboost_tuned

OUT = PROJECT_ROOT / "results" / "member_a" / "trend" / "horizon_backtest.csv"
TABLE = PROJECT_ROOT / "report" / "tables" / "horizon.tex"

FEATURE_SET = "G0+G1"
# Last fit month of each backtest and its horizon in months. Data ends in 2025-03,
# so 2024-03 only has 12 months. It is the one origin with a low slope, like the final fit.
ORIGINS = {"2022-03": 16, "2022-09": 16, "2023-03": 16, "2023-09": 16, "2024-03": 12}
BANDS = {"1-6": (1, 6), "7-16": (7, 16)}
MODELS = {
    "lightgbm_tuned": make_lightgbm_tuned,
    "xgboost_tuned": make_xgboost_tuned,
    "random_forest": make_random_forest,
}
NAMES = {"lightgbm_tuned": "LightGBM", "xgboost_tuned": "XGBoost", "random_forest": "Random forest"}


def backtest(train: pd.DataFrame) -> pd.DataFrame:
    columns = feature_columns(FEATURE_SET)
    month = train["RENT_APPROVAL_DATE"].dt.to_period("M")
    rows = []
    for origin, horizon in ORIGINS.items():
        last = pd.Period(origin, "M")
        fit = train[month <= last]
        ahead = (month - last).map(lambda offset: offset.n)
        valid = train[(ahead >= 1) & (ahead <= horizon)]
        valid_ahead = ahead[valid.index].to_numpy()
        slope = MonthlyRentIndex().fit(fit["MONTH_INDEX"], fit[TARGET]).slope_
        for name, make_model in MODELS.items():
            for mode in ("raw", "index"):
                start = time.perf_counter()
                prediction, _, _ = fit_predict(make_model, fit, valid, columns, mode)
                print(f"{origin} {name} {mode}: {time.perf_counter() - start:.0f}s", flush=True)
                for band, (low, high) in BANDS.items():
                    part = (valid_ahead >= low) & (valid_ahead <= high)
                    metrics = regression_metrics(valid[TARGET].to_numpy()[part], prediction[part])
                    rows.append({"origin": origin, "band": band, "model": name, "target_mode": mode,
                                 "n_fit": len(fit), "n_valid": int(part.sum()),
                                 "index_slope": slope, "rmse": metrics["rmse"], "bias": metrics["bias"]})
    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame) -> None:
    """One row per origin and horizon band: raw and index RMSE for each model."""

    wide = table.pivot(index=["origin", "band"], columns=["model", "target_mode"], values="rmse")
    lines = [r"\begin{tabular}{llrrrrrr}", r"\hline",
             r" & & \multicolumn{2}{c}{\textbf{LightGBM}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{Forest}} \\",
             r"\textbf{Fit to} & \textbf{Months} & Raw & Index & Raw & Index & Raw & Index \\", r"\hline"]
    for (origin, band), row in wide.iterrows():
        cells = [f"{row[(model, mode)]:.0f}" for model in MODELS for mode in ("raw", "index")]
        lines.append(f"{origin} & {band.replace('-', ' to ')} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table-only", action="store_true", help="rebuild the table from the saved CSV")
    args = parser.parse_args()
    if args.table_only:
        table = pd.read_csv(OUT)
    else:
        train, _ = load_modeling_data()
        table = backtest(train)
        table.to_csv(OUT, index=False)
    write_table(table)
    summary = table.pivot_table(index=["origin", "band", "model"], columns="target_mode", values=["rmse", "bias"])
    print(summary.round(0).to_string())


if __name__ == "__main__":
    main()
