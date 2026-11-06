"""Member B holdout ablations and expanding-window backtests; run as a module."""
import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.horizon_backtest import ORIGINS
from scripts.lag_blend_submission import add_cell, lag_features
from scripts.suspicious_row_ablation import FLAGS
from src.baselines import BASELINE_MODELS, KNN_COLUMNS
from src.config import DATA_DIR, RESULTS_DIR, TARGET
from src.evaluation import MonthlyRentIndex, fit_predict, regression_metrics, time_holdout_masks
from src.features import FEATURE_SETS, feature_columns, load_modeling_data
from src.models import MODELS

OUT = RESULTS_DIR / "member_b" / "extensions"
METRICS = ["rmse", "mae", "mape", "bias", "r2"]
BASE = "G0+G1"
LOG = pd.read_csv(RESULTS_DIR / "results.csv").sort_values("timestamp")


def reference(model, mode, group=BASE):
    rows = LOG[(LOG.model == model) & (LOG.target_mode == mode) & (LOG.feature_set == group)
               & (LOG.split == "time") & (LOG.n_fit == 132538) & (LOG.n_valid == 17037)]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return {**row[METRICS].to_dict(), "source": row.run_id}


def factory(name, knn_columns=None):
    def make():
        estimator = MODELS[name]()
        if name == "knn" and knn_columns is not None:
            estimator.set_params(columns__columns=tuple(knn_columns))
        return estimator
    return make


class ResultFile:
    def __init__(self, experiment):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / (experiment + ".csv")
        self.rows = pd.read_csv(self.path).to_dict("records") if self.path.exists() else []

    def has(self, name, case, mode):
        return any((r["model"], r["case"], r["mode"]) == (name, case, mode) for r in self.rows)

    def add(self, name, case, mode, scores, **extra):
        self.rows.append(dict(model=name, case=case, mode=mode, **scores, **extra))
        tmp = self.path.with_suffix(".tmp")
        pd.DataFrame(self.rows).to_csv(tmp, index=False)
        tmp.replace(self.path)
        print(name, case, mode, round(scores["rmse"], 3), flush=True)


def evaluate(out, name, case, mode, fit, valid, cols, cached=None, knn=None, **extra):
    if out.has(name, case, mode):
        return
    if cached is None:
        prediction, _, _ = fit_predict(factory(name, knn), fit, valid, cols, mode)
        cached = dict(**regression_metrics(valid[TARGET], prediction), source="new fit")
    out.add(name, case, mode, cached, n_fit=len(fit), n_valid=len(valid), **extra)


def run(experiment):
    train, _ = load_modeling_data()
    fit_mask, valid_mask = time_holdout_masks(train)
    fit, valid = train.loc[fit_mask].copy(), train.loc[valid_mask].copy()
    cols = feature_columns(BASE)
    out = ResultFile(experiment)
    if experiment == "features":
        for group in FEATURE_SETS:
            offered = feature_columns(group)
            for name in BASELINE_MODELS:
                cached = reference(name, "raw", group)
                knn = [c for c in KNN_COLUMNS if c in offered]
                if cached is None and (name in ("global_median", "group_median")
                                       or (name == "knn" and len(knn) == 4)):
                    cached = reference(name, "raw")
                evaluate(out,name,group,"raw",fit,valid,offered,cached,knn,n_offered=len(offered))
    elif experiment in ("outliers","outliers_fit_only"):
        for flag in ["all", *FLAGS]:
            filtered = fit
            if flag != "all":
                detector = FLAGS[flag][1]
                drop = detector(fit) if experiment=="outliers_fit_only" else detector(train)[fit_mask]
                filtered = fit.loc[~drop]
            names = list(BASELINE_MODELS)
            if experiment=="outliers_fit_only":
                names.append("lightgbm_tuned")
            for name in names:
                evaluate(out,name,flag,"index",filtered,valid,cols,
                         reference(name,"index") if flag == "all" else None,
                         dropped=len(fit)-len(filtered))
    elif experiment == "lags":
        add_cell(train, DATA_DIR / "train.csv")
        fit, valid = train.loc[fit_mask].copy(), train.loc[valid_mask].copy()
        index = MonthlyRentIndex().fit(fit.MONTH_INDEX, fit[TARGET])
        fit["a"] = fit[TARGET] / index.transform(fit.MONTH_INDEX)
        fit_lags = lag_features(fit,fit,fit.MONTH_INDEX.max())
        valid_lags = lag_features(fit,valid,fit.MONTH_INDEX.max())
        extra = list(fit_lags)
        fit, valid = fit.join(fit_lags), valid.join(valid_lags)
        for name in BASELINE_MODELS:
            base = reference(name,"index")
            evaluate(out,name,"base","index",fit,valid,cols,base)
            evaluate(out,name,"plus16","index",fit,valid,cols+extra,
                     base if name in ("global_median","group_median") else None,
                     list(KNN_COLUMNS)+extra, n_lags=16)
    elif experiment == "horizon":
        old = pd.read_csv(RESULTS_DIR / "member_a/trend/horizon_backtest.csv")
        month = train.RENT_APPROVAL_DATE.dt.to_period("M").astype("int64")
        for origin, horizon in ORIGINS.items():
            ahead = month - pd.Period(origin, freq="M").ordinal
            fit = train.loc[ahead <= 0]
            valid = train.loc[ahead.between(1,horizon)]
            for name in BASELINE_MODELS:
                for mode in ("raw","index"):
                    bands = [("1-6",ahead.loc[valid.index]<=6),("7-16",ahead.loc[valid.index]>6)]
                    pending = [(band,mask) for band,mask in bands if not out.has(name,origin+"/"+band,mode)]
                    if not pending:
                        continue
                    records = old[(old.origin==origin)&(old.model==name)&(old.target_mode==mode)]
                    if records.empty:
                        pred,_,_ = fit_predict(factory(name),fit,valid,cols,mode)
                    for band,mask in pending:
                        if records.empty:
                            scores = dict(**regression_metrics(valid.loc[mask,TARGET],pred[mask]),source="new fit")
                        else:
                            r = records.loc[records.band==band].iloc[0]
                            assert (r.n_fit,r.n_valid)==(len(fit),int(mask.sum()))
                            scores = dict(rmse=r.rmse,bias=r.bias,source="member_a/trend/horizon_backtest.csv")
                        out.add(name,origin+"/"+band,mode,scores,n_fit=len(fit),n_valid=mask.sum(),
                                last_horizon=6 if band=="1-6" else horizon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", choices=("features","outliers","outliers_fit_only","lags","horizon"))
    with threadpool_limits(limits=1, user_api="blas"):
        for exp in parser.parse_args().experiments:
            run(exp)
