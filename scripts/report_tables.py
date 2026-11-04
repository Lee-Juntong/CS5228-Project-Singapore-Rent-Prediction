"""Build the Evaluation tables of the report from the logged results.

Reads results/results.csv, results/kaggle_log.csv and
results/member_a/repeat_rent/validation.csv, and writes
report/tables/{results,ablation,blend}.tex.

    python -m scripts.report_tables
"""

from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT
from src.evaluation import RESULTS_CSV

TABLES = PROJECT_ROOT / "report" / "tables"
VALIDATION = PROJECT_ROOT / "results" / "member_a" / "repeat_rent" / "validation.csv"
KAGGLE_LOG = PROJECT_ROOT / "results" / "kaggle_log.csv"

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
    "xgboost_default": "XGBoost (default)",
    "lightgbm_tuned": "LightGBM (tuned)",
    "catboost_tuned": "CatBoost (tuned)",
    "xgboost_tuned": "XGBoost (tuned)",
}
ROW_END = r" \\"


def signed(value: float, digits: int = 1) -> str:
    return f"${value:+.{digits}f}$"


def latest(results: pd.DataFrame, model: str, feature_set: str, target_mode: str) -> pd.Series:
    rows = results[
        (results.model == model)
        & (results.feature_set == feature_set)
        & (results.target_mode == target_mode)
        & (results.split == "time")
    ]
    return rows.sort_values("timestamp").iloc[-1]


def table(spec: str, header: str, rows: list[str]) -> str:
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\hline", header + ROW_END, r"\hline", *rows, r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def results_table(results: pd.DataFrame, validation: pd.DataFrame) -> str:
    rows = sorted((latest(results, model, "G0+G1", "index") for model in MODELS), key=lambda row: row.rmse)
    blend = validation[(validation.window == "holdout") & (validation.model == "lag_blend5")].iloc[0]
    lines = [rf"\textbf{{Five model blend}} & \textbf{{{blend.rmse:.1f}}} & {blend.mae:.1f} & {blend.mape:.2f} & {signed(blend.bias)} & " + ROW_END]
    for row in rows:
        cells = [MODELS[row.model], f"{row.rmse:.1f}", f"{row.mae:.1f}", f"{row.mape:.2f}", signed(row.bias), f"{row.train_seconds:.1f}"]
        lines.append(" & ".join(cells) + ROW_END)
    header = r"\textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{MAPE} & \textbf{Bias} & \textbf{Fit}"
    return table("lrrrrr", header, lines)


def ablation_table(results: pd.DataFrame) -> str:
    rows = []
    for feature_set, label in [("G0", "G0 (13)"), ("G0+G1", "G0+G1 (39)"), ("G0+G1+G2", "G0+G1+G2 (44)"), ("ALL", "All groups (54)")]:
        light = latest(results, "lightgbm_tuned", feature_set, "raw").rmse
        cat = latest(results, "catboost_tuned", feature_set, "raw").rmse
        rows.append(f"{label} & {light:.1f} & {cat:.1f}" + ROW_END)
    return table("lrr", r"\textbf{Features} & \textbf{LightGBM} & \textbf{CatBoost}", rows)


def blend_table(results: pd.DataFrame, validation: pd.DataFrame, log: pd.DataFrame) -> str:
    scores = validation.pivot(index="model", columns="window", values="rmse")

    def public(model: str, mode: str = "index") -> float:
        return log[(log.model == model) & (log.target_mode == mode)].public_lb.iloc[0]

    def row(label: str, model: str, public_score: float, bold: bool = False) -> str:
        cells = [f"{scores.loc[model, 'holdout']:.1f}", f"{scores.loc[model, '16_month']:.1f}", f"{public_score:.2f}"]
        if bold:
            label, cells = rf"\textbf{{{label}}}", [rf"\textbf{{{cell}}}" for cell in cells]
        return " & ".join([label, *cells]) + ROW_END

    raw = latest(results, "lightgbm_tuned", "G0+G1", "raw").rmse
    rows = [
        f"LightGBM (tuned), raw & {raw:.1f} & & {public('lightgbm_tuned', 'raw'):.2f}" + ROW_END,
        row("LightGBM (tuned)", "lightgbm_tuned_index", public("lightgbm_tuned")),
        row("Two model blend", "lag_blend2", public("lightgbm_lag_blend")),
        row("Index and raw, four models", "lag_blend_index_raw", public("lag_blend_index_raw", "index+raw")),
        row("Five model blend", "lag_blend5", public("lag_blend5"), bold=True),
    ]
    return table("lrrr", r"\textbf{Model} & \textbf{Holdout} & \textbf{16 month} & \textbf{Public}", rows)


def main() -> None:
    results = pd.read_csv(RESULTS_CSV)
    validation = pd.read_csv(VALIDATION)
    log = pd.read_csv(KAGGLE_LOG)
    (TABLES / "results.tex").write_text(results_table(results, validation), encoding="utf-8")
    (TABLES / "ablation.tex").write_text(ablation_table(results), encoding="utf-8")
    (TABLES / "blend.tex").write_text(blend_table(results, validation, log), encoding="utf-8")


if __name__ == "__main__":
    main()
