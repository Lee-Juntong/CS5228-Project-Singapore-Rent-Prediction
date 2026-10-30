"""Fit the final model on all training rows and save its weights for replay.

The shared `fit_predict` / `make_submission` throw the fitted model away after
predicting, so a reported submission cannot be reloaded later. This script
mirrors `make_submission`'s all-rows fit but keeps the fitted factory object and
`joblib.dump`s it, alongside a JSON sidecar recording everything needed to
reproduce the numbers (model, feature set, target mode, seed, git commit,
library versions). It reuses the exact pipeline primitives (`set_seed`, the
`MonthlyRentIndex` trend handling) so the saved model's predictions match the
logged run bit for bit.

    python -m scripts.save_final_model --model catboost_tuned --features G0+G1+G2
    python -m scripts.save_final_model --model lightgbm_tuned --features G0+G1 --target-mode index

Weights land in `models/` (gitignored). See models/README.md to reload one.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np

from src.config import PROJECT_ROOT, SEED, TARGET, set_seed
from src.evaluation import MonthlyRentIndex, _git_commit, regression_metrics
from src.evaluation import in_time_valid_window, time_holdout_masks
from src.features import feature_columns, load_modeling_data
from src.models import MODELS

MODELS_DIR = PROJECT_ROOT / "models"
_LIBS = ("lightgbm", "catboost", "scikit-learn", "numpy", "pandas")


def save_final_model(model_name: str, feature_set: str, target_mode: str,
                     data=None) -> dict:
    """Fit `model_name` on all training rows and dump the fitted model + sidecar."""

    if model_name not in MODELS:
        raise KeyError(f"Unknown model {model_name}; choose from {list(MODELS)}")
    make_model = MODELS[model_name]
    train, _ = data if data is not None else load_modeling_data()
    columns = feature_columns(feature_set)

    # Same recipe as fit_predict: fix the seed, build the target (optionally
    # trend-removed), fit a fresh model on every training row.
    set_seed()
    y = train[TARGET].to_numpy(dtype=float)
    index = None
    if target_mode == "index":
        index = MonthlyRentIndex().fit(train["MONTH_INDEX"], train[TARGET])
        y = y / index.transform(train["MONTH_INDEX"])
    model = make_model()
    start = time.perf_counter()
    model.fit(train[columns], y)
    fit_seconds = time.perf_counter() - start

    # Holdout RMSE for the sidecar, so the saved model carries its own score.
    # This is an in-sample slice (the model saw these rows), reported only to
    # tie the artifact back to the logged time-holdout number, not as a metric.
    valid = in_time_valid_window(train)
    pred = np.asarray(model.predict(train[valid][columns]), dtype=float)
    if target_mode == "index":
        pred = pred * index.transform(train[valid]["MONTH_INDEX"])
    insample_window_rmse = regression_metrics(train.loc[valid, TARGET], pred)["rmse"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{feature_set}_{target_mode}".replace("+", "")
    weights_path = MODELS_DIR / f"{stem}.joblib"
    sidecar_path = MODELS_DIR / f"{stem}.json"
    joblib.dump({"model": model, "columns": columns, "target_mode": target_mode,
                 "rent_index": index}, weights_path)
    sidecar = {
        "model": model_name,
        "feature_set": feature_set,
        "target_mode": target_mode,
        "seed": SEED,
        "fit_rows": int(len(train)),
        "fit_seconds": round(fit_seconds, 2),
        "insample_time_window_rmse": round(float(insample_window_rmse), 2),
        "git_commit": _git_commit(),
        "libraries": {lib: importlib.metadata.version(lib) for lib in _LIBS},
        "weights_file": weights_path.name,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"saved {weights_path.name} (fit {fit_seconds:.1f}s on {len(train)} rows)")
    print(f"sidecar {sidecar_path.name}")
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--features", default="G0+G1+G2")
    parser.add_argument("--target-mode", default="raw", choices=["raw", "index"])
    args = parser.parse_args()
    save_final_model(args.model, args.features, args.target_mode)


if __name__ == "__main__":
    main()
