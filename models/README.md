# Saved model weights

Fitted final models for replay. The weight files (`*.joblib`) are **not in git**
(large binaries, tied to the pinned library versions in `requirements.txt`);
only this README is committed. Rebuild them from the shared data in one command.

## Rebuild

From the project root, after the processed data is built (see `data/README.md`):

```bash
python -m scripts.save_final_model --model catboost_tuned --features G0+G1+G2
python -m scripts.save_final_model --model lightgbm_tuned --features G0+G1 --target-mode index
```

Each call fits the model on all training rows with the shared seed (5228) and
the same trend handling the evaluation pipeline uses, then writes:

* `<model>_<features>_<target_mode>.joblib` — the fitted model plus the feature
  column list, target mode and (for index mode) the fitted `MonthlyRentIndex`.
* `<model>_<features>_<target_mode>.json` — a sidecar recording the model,
  feature set, target mode, seed, git commit, library versions and the fit's
  in-sample time-window RMSE, so a saved model carries its own provenance.

## Reload and predict

```python
import joblib
from src.features import load_modeling_data

bundle = joblib.load("models/catboost_tuned_G0G1G2_raw.joblib")
model, columns, target_mode, rent_index = (
    bundle["model"], bundle["columns"], bundle["target_mode"], bundle["rent_index"]
)

_, test = load_modeling_data()
pred = model.predict(test[columns])
if target_mode == "index":           # undo the trend removal
    pred = pred * rent_index.transform(test["MONTH_INDEX"])
```

This reproduces the submission predictions exactly, since the model was fit with
the same seed, columns and trend handling as the reported run.
