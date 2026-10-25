"""Tune the boosting models on the time holdout with a fixed Optuna budget.

Phase 2, decision "tune with a small fixed budget" from the final report plan.
Each study optimises time-holdout RMSE (the reported metric) over the same
Jan 2021 to Sep 2024 fit / Oct 2024 to Mar 2025 validation split that
`run_experiment` uses, so the tuned numbers are comparable to every other run.

The search runs through the shared `fit_predict`, which calls the model factory
fresh and fixes the seed for every trial. Best parameters are written to
`configs/<model>_tuned.json`; `make_lightgbm_tuned` / `make_catboost_tuned` in
`src/models.py` read them back.

    python -m scripts.tune_boosting                 # both models, 30 trials each
    python -m scripts.tune_boosting --model catboost --trials 50
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import optuna

from src.config import SEED, CONFIG_DIR
from src.evaluation import fit_predict, regression_metrics, time_holdout_masks
from src.features import feature_columns, load_modeling_data

# Tuning uses the Phase 1 feature set and the primary (raw) target, matching the
# headline results table.
FEATURE_SET = "G0+G1"
TARGET_MODE = "raw"


def _lightgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _catboost_space(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 200, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def _lightgbm_factory(params: dict) -> Callable:
    def make():
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, **params)

    return make


def _catboost_factory(params: dict) -> Callable:
    def make():
        from src.models import CatBoostModel

        return CatBoostModel(**params)

    return make


SPACES = {"lightgbm": _lightgbm_space, "catboost": _catboost_space}
FACTORIES = {"lightgbm": _lightgbm_factory, "catboost": _catboost_factory}


def tune(model: str, trials: int, timeout: float | None, data) -> dict:
    """Run one study and return the best parameters."""

    train, _ = data
    fit_mask, valid_mask = time_holdout_masks(train)
    fit_frame, valid_frame = train[fit_mask], train[valid_mask]
    columns = feature_columns(FEATURE_SET)
    y_true = valid_frame["MONTHLY_RENT"].to_numpy(dtype=float)
    build_factory = FACTORIES[model]
    space = SPACES[model]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        prediction, _, _ = fit_predict(
            build_factory(params), fit_frame, valid_frame, columns, TARGET_MODE
        )
        return regression_metrics(y_true, prediction)["rmse"]

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)
    return study.best_params, study.best_value


def save_params(model: str, params: dict, rmse: float) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{model}_tuned.json"
    payload = {
        "model": model,
        "feature_set": FEATURE_SET,
        "target_mode": TARGET_MODE,
        "seed": SEED,
        "holdout_rmse": round(float(rmse), 4),
        "params": params,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path} (holdout RMSE {rmse:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", nargs="+", default=["lightgbm", "catboost"], choices=list(SPACES)
    )
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument(
        "--timeout", type=float, default=None, help="wall-clock cap in seconds per model"
    )
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = load_modeling_data()  # load once, reuse for every study
    for model in args.model:
        print(f"== tuning {model}: {args.trials} trials ==")
        params, rmse = tune(model, args.trials, args.timeout, data)
        save_params(model, params, rmse)


if __name__ == "__main__":
    main()
