"""Held-out interpretation of a retained shared rent model, always in SGD.

No estimator or target transform is fitted here. Call ``fit_predict`` with
``return_fitted=True`` on the agreed fit partition before using this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance

from src.config import SEED, TARGET
from src.evaluation import FittedRentModel, in_time_valid_window, regression_metrics

PDP_FEATURES = ("FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "NEAREST_OPEN_MRT_KM", "MONTH_INDEX")
PDP_LABELS = ("Floor area (sq m)", "Flat age (years)", "Nearest open MRT (km)",
              "Month index (Jan 2021 = 0)")
BLUE, ORANGE = "#3978a8", "#b56e29"


class _PredictionAdapter(RegressorMixin, BaseEstimator):
    """Expose the full SGD prediction to sklearn inspection without refitting."""

    def __init__(self, fitted):
        self.fitted = fitted
        self.is_fitted_ = True

    def fit(self, X, y=None):
        raise RuntimeError("Interpretation must use the already fitted model")

    def predict(self, X):
        return self.fitted.predict(X)


def _negative_rmse(estimator, X, y):
    return -regression_metrics(y, estimator.predict(X))["rmse"]


def heldout_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Choose stable ROW_IDs, independently of input ordering and target values."""
    required = {"ROW_ID", "RENT_APPROVAL_DATE", TARGET}
    if not required <= set(frame):
        raise ValueError(f"Held-out interpretation requires {sorted(required)}")
    if frame.empty or frame.ROW_ID.isna().any() or frame.ROW_ID.duplicated().any():
        raise ValueError("Interpretation needs nonempty data with unique nonmissing ROW_ID")
    if not in_time_valid_window(frame).all():
        raise ValueError("Interpretation rows must all belong to the shared time holdout")
    if not isinstance(sample_size, (int, np.integer)) or sample_size < 2:
        raise ValueError("sample_size must be an integer of at least 2")
    ordered = frame.sort_values("ROW_ID").reset_index(drop=True)
    positions = np.sort(np.random.default_rng(seed).choice(
        len(ordered), min(sample_size, len(ordered)), replace=False))
    sample = ordered.iloc[positions].copy()
    # Also checks positive finite rents, using the team's metric contract.
    regression_metrics(sample[TARGET], sample[TARGET])
    return sample


def _save_figure(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _pdp_grid(values: pd.Series) -> np.ndarray:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError(f"Cannot construct PDP for all-missing feature {values.name}")
    unique = np.unique(finite)
    return unique if len(unique) <= 20 else np.unique(np.quantile(finite, np.linspace(.05, .95, 20)))


def _intervene(sample: pd.DataFrame, feature: str, value: float,
               coherent: bool = False) -> pd.DataFrame:
    changed = sample.copy()
    changed[feature] = value
    if coherent and feature == "FLOOR_AREA_SQM":
        if value <= 0:
            raise ValueError("Area sensitivity requires positive floor area")
        if "LOG_FLOOR_AREA_SQM" in changed:
            changed["LOG_FLOOR_AREA_SQM"] = np.log(value)
        if "FLOOR_AREA_PER_ROOM_PROXY" in changed:
            changed["FLOOR_AREA_PER_ROOM_PROXY"] = value / changed["FLAT_TYPE_ORDINAL"]
    if coherent and feature == "FLAT_AGE_YEARS":
        changed["LEASE_COMMENCE_DATE"] = changed.RENT_APPROVAL_DATE.dt.year - value
        if "INVALID_LEASE_DATE_FLAG" in changed:
            changed["INVALID_LEASE_DATE_FLAG"] = 0
        if "REMAINING_LEASE_PROXY_YEARS" in changed:
            changed["REMAINING_LEASE_PROXY_YEARS"] = max(0, 99 - value)
    return changed


def partial_dependence_table(fitted: FittedRentModel, sample: pd.DataFrame,
                             comparison_fitted: FittedRentModel | None = None) -> pd.DataFrame:
    """Brute-force average full-model predictions; curves are descriptive only.

    Standard curves set one column and retain all other columns. Area and age
    sensitivity curves also update their exact engineered derivatives. Neither
    intervention implies a feasible or causal change in a real flat.
    """
    rows = []
    models = [("primary", fitted)]
    if comparison_fitted is not None:
        models.append(("ridge", comparison_fitted))
    for label, model in models:
        baseline = float(np.mean(model.predict(sample)))
        for feature in PDP_FEATURES:
            if feature not in model.columns:
                raise ValueError(f"Requested PDP feature is not a model input: {feature}")
            for value in _pdp_grid(sample[feature]):
                methods = [False, True] if feature in PDP_FEATURES[:2] else [False]
                for coherent in methods:
                    prediction = model.predict(_intervene(sample, feature, float(value), coherent))
                    rows.append({"model": label, "feature": feature, "value": float(value),
                                 "intervention": "coherent_derivatives" if coherent else "standard",
                                 "mean_prediction_sgd": float(prediction.mean()),
                                 "mean_change_sgd": float(prediction.mean()) - baseline,
                                 "sample_n": len(sample)})
    return pd.DataFrame(rows)


def generate_interpretation(fitted: FittedRentModel, valid_frame: pd.DataFrame,
                            output_dir: str | Path, *,
                            comparison_fitted: FittedRentModel | None = None,
                            sample_size: int = 2000, n_repeats: int = 5,
                            seed: int = SEED) -> dict:
    """Write repeated permutation importance, PDPs and their exact source tables.

    ``comparison_fitted`` must use identical columns and target mode so Ridge
    and boosting curves describe matching information and full-SGD outputs.
    """
    if not isinstance(n_repeats, (int, np.integer)) or n_repeats < 2:
        raise ValueError("Use at least two permutations to report repeat variability")
    if fitted.target_mode == "index" and fitted.rent_index is None:
        raise ValueError("Index interpretation requires the retained training-fitted rent index")
    if comparison_fitted is not None:
        if (fitted.columns != comparison_fitted.columns or
                fitted.target_mode != comparison_fitted.target_mode):
            raise ValueError("Comparison requires identical feature columns and target modes")
        if comparison_fitted.target_mode == "index" and comparison_fitted.rent_index is None:
            raise ValueError("Index comparison requires its training-fitted rent index")
    sample = heldout_sample(valid_frame, sample_size, seed)
    missing = set(fitted.columns) - set(sample)
    if missing:
        raise ValueError(f"Model columns missing from interpretation data: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample[["ROW_ID", "RENT_APPROVAL_DATE"]].to_csv(output / "sample_rows.csv", index=False)
    X = sample[fitted.columns].copy()
    # MONTH_INDEX is required by the full prediction wrapper in index mode.
    if fitted.target_mode == "index" and "MONTH_INDEX" not in X:
        raise ValueError("Index-mode importance requires MONTH_INDEX among the model columns")
    baseline = regression_metrics(sample[TARGET], fitted.predict(sample))["rmse"]
    importance = permutation_importance(
        _PredictionAdapter(fitted), X, sample[TARGET], scoring=_negative_rmse,
        n_repeats=n_repeats, random_state=seed, n_jobs=1)
    table = pd.DataFrame({"feature": fitted.columns,
                          "importance_mean_sgd": importance.importances_mean,
                          "importance_std_sgd": importance.importances_std,
                          "n_repeats": n_repeats, "sample_n": len(sample),
                          "baseline_rmse_sgd": baseline}).sort_values(
                              "importance_mean_sgd", ascending=False)
    table.to_csv(output / "permutation_importance.csv", index=False)
    pd.DataFrame([{"feature": feature, "repeat": j + 1, "delta_rmse_sgd": float(value)}
                  for feature, values in zip(fitted.columns, importance.importances)
                  for j, value in enumerate(values)]).to_csv(
                      output / "permutation_repeats.csv", index=False)
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.importance_mean_sgd, xerr=top.importance_std_sgd,
            color=BLUE, ecolor="#444444", capsize=2)
    ax.axvline(0, color="#666666", lw=.8)
    ax.set(xlabel="Increase in held-out RMSE after permutation (SGD)",
           title=f"Permutation importance | n={len(sample):,}, {n_repeats} repeats")
    ax.text(0, -.16, "Error bars: standard deviation across permutations; not confidence intervals.",
            transform=ax.transAxes, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_figure(fig, output / "permutation_importance")

    pdp = partial_dependence_table(fitted, sample, comparison_fitted)
    pdp.to_csv(output / "partial_dependence.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, feature, xlabel in zip(axes.flat, PDP_FEATURES, PDP_LABELS):
        for (model, method), part in pdp[pdp.feature == feature].groupby(
                ["model", "intervention"], sort=False):
            color = BLUE if model == "primary" else ORANGE
            label = ("Boosting" if model == "primary" else "Ridge") + (
                " (derived columns updated)" if method == "coherent_derivatives" else " (standard PDP)")
            ax.plot(part.value, part.mean_prediction_sgd, color=color,
                    linestyle=":" if method == "coherent_derivatives" else "-",
                    marker="o" if model == "primary" else "s", ms=2.5, label=label)
        ax.set(xlabel=xlabel, ylabel="Mean predicted monthly rent (SGD)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        if feature == "MONTH_INDEX":
            months = sorted(pdp.loc[pdp.feature == feature, "value"].unique())
            ax.set_xticks(months)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(.5, -.01))
    fig.suptitle(f"Partial dependence on the same held-out sample | n={len(sample):,}")
    fig.tight_layout(rect=(0, .08, 1, .96))
    _save_figure(fig, output / "partial_dependence")
    summary = {
        "sample_n": len(sample), "validation_n": len(valid_frame), "seed": int(seed),
        "sample_rule": "Sort ROW_ID; numpy default_rng(seed) sample without replacement; retain sorted positions",
        "n_repeats": int(n_repeats), "baseline_rmse_sgd": baseline,
        "target_mode": fitted.target_mode,
        "importance_definition": "Permuted full-SGD RMSE minus unpermuted full-SGD RMSE; positive means worse",
        "uncertainty": "Population SD across fixed-sample permutations (ddof=0); not sampling uncertainty or a confidence interval",
        "pdp_grid": "All distinct values if <=20; otherwise 20 sample quantiles between 5% and 95%",
        "pdp_definition": "Average full-SGD prediction after setting one feature on every sampled row",
        "pdp_sensitivity": "Area updates log-area and area-per-room; age updates lease commencement, invalid flag and remaining-lease proxy when present",
        "limitations": [
            "Held-out targets affect scoring only; no model, encoder or rent index is fitted here.",
            "The shared holdout also guided model tuning; it is not an independent post-selection test.",
            "Correlated inputs can share or mask permutation importance; rankings are not causal.",
            "Individual permutations and standard PDPs can create implausible combinations with engineered or correlated inputs.",
            "Area/age sensitivity maintains exact derivatives but can still create unrealistic flat-type, location and age combinations.",
            "Month PDP changes MONTH_INDEX (and the retained rent-index restoration), holding seasonality and age columns fixed.",
            "PDPs describe model behavior on a selected held-out population, not real-world interventions."
        ],
        "top_features": table.head(10).to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
