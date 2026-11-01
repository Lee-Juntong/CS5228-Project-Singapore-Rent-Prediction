"""Reusable plots, learning curves and ID-aligned evaluation for any member."""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import SEED, TARGET
from src.evaluation import fit_predict, regression_metrics, segment_metrics, time_holdout_masks
from src.features import feature_columns

LABELS = {"global_median": "Global median", "group_median": "Group median",
          "linear_regression": "Linear regression", "ridge": "Ridge", "knn": "KNN",
          "decision_tree": "Decision tree", "random_forest": "Random forest"}
METRICS = {"rmse": "RMSE (SGD; lower is better)", "mae": "MAE (SGD; lower is better)",
           "mape": "MAPE (%; lower is better)", "r2": "R-squared (higher is better)",
           "bias": "Mean prediction minus rent (SGD; ideal = 0)"}


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_comparison(table, output_dir):
    """One figure per metric; compare only matching split/window/protocol."""
    output_dir = Path(output_dir)
    for column in ("split", "feature_set", "target_mode", "n_valid"):
        if column in table and table[column].nunique() > 1:
            raise ValueError(f"Cannot plot different {column} values as one comparison")
    ordered = table.sort_values("rmse", ascending=False)
    for metric, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        values = ordered[metric].to_numpy()
        bars = ax.barh([LABELS.get(x, x) for x in ordered["model"]], values, color="#3978a8")
        ax.bar_label(bars, fmt="%.3f" if metric == "r2" else "%.2f", padding=5, fontsize=10)
        ax.axvline(0, color="#777777", lw=0.8)
        ax.margins(x=0.22)
        ax.set_xlabel(label)
        ax.set_title("Baseline ladder | Oct 2024 - Mar 2025 holdout", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        fig.tight_layout()
        save_figure(fig, output_dir / metric)


def plot_segments(table, output_dir):
    for segment_type in ("month", "flat_type"):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for model, part in table[table.segment_type == segment_type].groupby("model", sort=False):
            part = part.sort_values("segment")
            ax.plot(part.segment, part.rmse, marker="o", ms=4, label=LABELS.get(model, model))
        ax.set(xlabel="Validation month" if segment_type == "month" else "Flat type", ylabel="RMSE (SGD)",
               title=f"Holdout error by {segment_type.replace('_', ' ')}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_figure(fig, Path(output_dir) / f"rmse_by_{segment_type}")


def learning_curve(model_name, make_model, train, output_dir, fitted_full=None,
                   feature_set="G0+G1", fractions=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Nested month-stratified subsamples of the fit window; fixed future holdout."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_mask, valid_mask = time_holdout_masks(train)
    fit, valid = train.loc[fit_mask], train.loc[valid_mask]
    if not fractions or any(f <= 0 or f > 1 for f in fractions):
        raise ValueError("fractions must lie in (0, 1]")
    rng = np.random.default_rng(SEED)
    month = fit.RENT_APPROVAL_DATE.dt.strftime("%Y-%m")
    groups = [rng.permutation(np.flatnonzero(month.to_numpy() == value)) for value in sorted(month.unique())]
    rows = []
    for fraction in fractions:
        positions = np.sort(np.concatenate([g[:max(1, int(len(g) * fraction))] for g in groups]))
        sample = fit.iloc[positions]
        start = time.perf_counter()
        if fraction == 1 and fitted_full is not None:
            fitted = fitted_full
            prediction = fitted.predict(valid)
            fit_seconds = np.nan
        else:
            prediction, fit_seconds, _, fitted = fit_predict(make_model, sample, valid,
                                                             feature_columns(feature_set), return_fitted=True)
        training = regression_metrics(sample[TARGET], fitted.predict(sample))
        validation = regression_metrics(valid[TARGET], prediction)
        rows.append({"model": model_name, "fraction": fraction, "n_fit": len(sample),
                     "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter() - start,
                     **{f"train_{k}": v for k, v in training.items()},
                     **{f"valid_{k}": v for k, v in validation.items()}})
        pd.DataFrame(rows).to_csv(output_dir / f"{model_name}.csv", index=False)
        print(f"  curve {model_name} {fraction:.0%}: valid RMSE {validation['rmse']:.2f}", flush=True)
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve.n_fit, curve.train_rmse, "o-", label="Training (in-sample)", color="#3978a8")
    ax.plot(curve.n_fit, curve.valid_rmse, "o-", label="Future holdout", color="#d67932")
    ax.set(xlabel="Training rows (nested samples across training months)", ylabel="RMSE (SGD)",
           title=LABELS.get(model_name, model_name) + " | Sample-size learning curve")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, output_dir / model_name)
    return curve


def forest_curve(fitted, train, output_dir, counts=(10, 25, 50, 100, 200, 300)):
    """Score prefixes of one saved forest; no repeated training or early stopping."""
    fit, valid = time_holdout_masks(train)
    model = fitted.model
    matrices = [model.named_steps["prep"].transform(train.loc[mask, fitted.columns]) for mask in (fit, valid)]
    sums = [np.zeros(matrix.shape[0]) for matrix in matrices]
    rows = []
    for n, tree in enumerate(model.named_steps["model"].estimators_, 1):
        for i, matrix in enumerate(matrices):
            sums[i] += tree.predict(matrix)
        if n in counts:
            rows.append({"n_estimators": n,
                         **{f"{label}_{k}": v for label, mask, values in zip(("train", "valid"), (fit, valid), sums)
                            for k, v in regression_metrics(train.loc[mask, TARGET], values / n).items()}})
    table = pd.DataFrame(rows)
    table.to_csv(Path(output_dir) / "random_forest_trees.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table.n_estimators, table.train_rmse, "o-", label="Training (in-sample)")
    ax.plot(table.n_estimators, table.valid_rmse, "o-", label="Future holdout")
    ax.set(xlabel="Number of trees (prefix of the saved 300-tree forest)", ylabel="RMSE (SGD)",
           title="Random forest | Ensemble-size curve")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, Path(output_dir) / "random_forest_trees")
    return table


def evaluate_predictions(valid_frame, predictions, output_dir):
    """Evaluate ROW_ID,y_pred files without requiring access to the estimator."""
    required = {"ROW_ID", "y_pred"}
    if not required <= set(predictions):
        raise ValueError("Predictions require ROW_ID and y_pred")
    if valid_frame.ROW_ID.isna().any() or predictions.ROW_ID.isna().any():
        raise ValueError("ROW_ID cannot be missing")
    if valid_frame.ROW_ID.duplicated().any() or predictions.ROW_ID.duplicated().any():
        raise ValueError("ROW_ID must be unique")
    if set(valid_frame.ROW_ID) != set(predictions.ROW_ID):
        raise ValueError("Prediction IDs must exactly match validation IDs (no missing or extra rows)")
    aligned = predictions.set_index("ROW_ID").loc[valid_frame.ROW_ID, "y_pred"].to_numpy(float)
    metrics = regression_metrics(valid_frame[TARGET], aligned)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    segment_metrics(valid_frame, aligned).to_csv(output_dir / "segment_metrics.csv", index=False)
    return metrics
