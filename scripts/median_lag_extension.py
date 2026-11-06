"""Feature-aware variants of the two fixed lookup controls in lags.csv."""
import numpy as np
from scripts.baseline_extensions import BASE, ResultFile
from scripts.lag_blend_submission import add_cell, lag_features
from src.baselines import BASELINE_MODELS
from src.config import DATA_DIR, TARGET
from src.evaluation import MonthlyRentIndex, regression_metrics, time_holdout_masks
from src.features import feature_columns, load_modeling_data


def main():
    out = ResultFile("median_lag_variants")
    names = [n for n in ("global_median","group_median") if not out.has(n,"L24_shrink5","index")]
    if not names:
        return
    train,_ = load_modeling_data()
    add_cell(train,DATA_DIR/"train.csv")
    a,b = time_holdout_masks(train)
    fit,valid = train.loc[a].copy(),train.loc[b]
    index = MonthlyRentIndex().fit(fit.MONTH_INDEX,fit[TARGET])
    fit["a"] = fit[TARGET]/index.transform(fit.MONTH_INDEX)
    lag = lag_features(fit,valid,fit.MONTH_INDEX.max())
    # Same lack of history as the primary models. Five pseudo-observations
    # shrink a sparse cell towards its ordinary median baseline.
    n = lag.L24_n.fillna(0).to_numpy()
    local = lag.L24_mean.fillna(0).to_numpy()
    cols = feature_columns(BASE)
    for name in names:
        ratio = BASELINE_MODELS[name]().fit(fit[cols],fit.a).predict(valid[cols])
        pred = ((5*ratio+n*local)/(5+n))*index.transform(valid.MONTH_INDEX)
        out.add(name,"L24_shrink5","index",regression_metrics(valid[TARGET],pred),
                n_fit=len(fit),n_valid=len(valid),with_history=int(np.sum(n>0)),
                source="median plus local 24-month mean; fixed prior strength 5")


if __name__ == "__main__":
    main()
