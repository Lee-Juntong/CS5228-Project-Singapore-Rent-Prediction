"""Vintage-aware monthly rent index and paired holdout evaluation."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from scripts.baseline_extensions import OUT, BASE, reference
from scripts.trend_analysis import MODELS as REPORT_MODELS
from src.config import DATA_DIR, RESULTS_DIR, TARGET
from src.evaluation import MonthlyRentIndex, fit_predict, regression_metrics, time_holdout_masks
from src.features import feature_columns, load_modeling_data
from src.models import MODELS

VINTAGES = pd.read_csv(DATA_DIR / "official_forecasts/sg_forecast_vintages.csv",
                       parse_dates=["published"]).sort_values("published")
DEST = OUT / "macro_index"
FIRST = pd.Period("2021-01", freq="M")


def asof(target, available_before):
    """A target-year forecast can be missing at a fixed multi-month origin."""
    limit = min(target.start_time, available_before)
    subset = VINTAGES[(VINTAGES.target_year == target.year) & (VINTAGES.published < limit)]
    values, provenance = [], {}
    for key in ("gdp", "cpi_all"):
        releases = subset.loc[subset.series == key]
        if releases.empty:
            values.append(np.nan)
            provenance[key + "_release"] = ""
        else:
            r = releases.iloc[-1]
            values.append((r.low_pct + r.high_pct) / 2)
            provenance[key + "_release"] = r.published.date().isoformat()
        provenance[key + "_pct"] = values[-1]
    return values, provenance


def features(history, month, before, use_macro):
    # The full prior calendar year and every earlier month of the current year.
    first = pd.Period(f"{month.year-1}-01", freq="M")
    months = pd.period_range(first, month-1, freq="M")
    x = history.loc[months].to_numpy(dtype=float)
    assert len(x) == 11 + month.month and np.isfinite(x).all()
    anchor = x[-3:].mean()
    normalized = np.log(x / anchor)
    # Put January in a fixed slot; unobserved slots contribute zero.
    inputs = [*normalized, *([0.] * (23-len(x))), np.sin(2*np.pi*month.month/12),
              np.cos(2*np.pi*month.month/12)]
    macro, meta = asof(month, before)
    if use_macro:
        # Convert percent to fractions on the same scale as log-rent changes.
        inputs.extend(np.asarray(macro)/100)
    return inputs, anchor, meta


def fit_small(history, alpha, use_macro):
    X, y = [], []
    for t in history.index:
        if t.year <= history.index.min().year:
            continue
        x, anchor, _ = features(history.loc[:t-1],t,t.start_time,use_macro)
        X.append(x)
        y.append(np.log(history[t]/anchor))
    X = np.asarray(X)
    # If a new year had no official forecast at issuance, use the fit-period mean.
    means = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X),means,X)
    model = Ridge(alpha=alpha).fit(X,y)
    return model, means


def trajectory(history, end, alpha, use_macro, actual_updates=None):
    history = history.copy()
    start = history.index.max()+1
    model, means = fit_small(history,alpha,use_macro)
    rows = []
    for t in pd.period_range(start,end,freq="M"):
        before = start.start_time if actual_updates is None else t.start_time
        x, anchor, meta = features(history,t,before,use_macro)
        x = np.where(np.isnan(x),means,x)
        forecast = anchor * np.exp(model.predict(x.reshape(1,-1))[0])
        assert np.isfinite(forecast) and forecast > 0
        rows.append(dict(month=str(t), forecast=forecast, **meta))
        history.loc[t] = forecast if actual_updates is None else actual_updates.loc[t]
    return pd.DataFrame(rows)


def choose_alpha(history):
    # Choose on complete six-month windows entirely before the evaluation period.
    scores = []
    for alpha in (0.05,0.5,5.0):
        for start in pd.period_range("2023-01",history.index.max()-5,freq="M")[::3]:
            stop = start+5
            if stop > history.index.max():
                continue
            f = trajectory(history.loc[:start-1],stop,alpha,True)
            diff = f.forecast.to_numpy()-history.loc[start:stop].to_numpy()
            scores.append(dict(alpha=alpha,origin=str(start-1),rmse=np.sqrt(np.mean(diff**2))))
    scores = pd.DataFrame(scores)
    winner = scores.groupby("alpha").rmse.apply(lambda x: np.sqrt(np.mean(x**2))).idxmin()
    return float(winner),scores


def anchored(history, end, half_life, use_macro, actual_updates=None):
    """Age each observed rent at a common annual nominal drift, then average.

    The drift is a scenario assumption, with half weight on real GDP growth
    and full weight on headline CPI. Select only the memory length on past origins.
    """
    history = history.copy()
    start = history.index.max()+1
    past = [asof(t,t.start_time)[0] for t in history.index]
    fallback = np.nanmean(past,axis=0)
    rows = []
    for t in pd.period_range(start,end,freq="M"):
        before = t.start_time if actual_updates is not None else start.start_time
        macro,meta = asof(t,before)
        gdp,cpi = np.where(np.isnan(macro),fallback,macro)
        trend = (cpi+0.5*gdp)/100 if use_macro else 0.
        x = history.loc[pd.Period(f"{t.year-1}-01","M"):t-1].to_numpy()
        assert len(x) == t.month+11
        age = np.arange(len(x),0,-1)
        forecast = np.average(x*(1+trend)**(age/12),weights=2**(-age/half_life))
        rows.append(dict(month=str(t),forecast=forecast,**meta))
        history.loc[t] = forecast if actual_updates is None else actual_updates.loc[t]
    return pd.DataFrame(rows)


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    train,_ = load_modeling_data()
    train_month = train.RENT_APPROVAL_DATE.dt.to_period("M")
    observed = train.groupby(train_month)[TARGET].median()
    fit_mask,valid_mask = time_holdout_masks(train)
    fit,valid = train.loc[fit_mask],train.loc[valid_mask]
    months = train_month.loc[valid_mask]
    cutoff = pd.Period("2024-09","M")
    end = observed.index.max()
    history = observed.loc[:cutoff]
    alpha, grid = choose_alpha(history)
    grid.to_csv(DEST/"training_cv.csv",index=False)
    memory_cv = []
    for origin in sorted(grid.origin.unique()):
        t = pd.Period(origin,"M")
        for half in (1.5,3.,6.):
            f = anchored(history.loc[:t],t+6,half,True)
            delta = f.forecast.to_numpy()-history.loc[t+1:t+6].to_numpy()
            memory_cv.append(dict(origin=origin,half_life=half,mse=np.mean(delta**2)))
    memory_cv = pd.DataFrame(memory_cv)
    half = float(memory_cv.groupby("half_life").mse.mean().idxmin())
    memory_cv.to_csv(DEST/"anchored_training_cv.csv",index=False)
    development = []
    for origin in sorted(grid.origin.unique()):
        t = pd.Period(origin,"M")
        past = history.loc[:t]
        actual = history.loc[t+1:t+6].to_numpy()
        forecasts = {
            "last": np.repeat(past.iloc[-1],6),
            "linear12": np.polyval(np.polyfit(np.arange(12),past.iloc[-12:],1),np.arange(12,18)),
            "macro": trajectory(past,t+6,alpha,True).forecast,
            "no_macro": trajectory(past,t+6,alpha,False).forecast,
            "anchored_macro": anchored(past,t+6,half,True).forecast,
            "anchored_history": anchored(past,t+6,half,False).forecast,
        }
        for method, forecast in forecasts.items():
            development.append(dict(origin=origin,method=method,**regression_metrics(actual,forecast)))
    pd.DataFrame(development).to_csv(DEST/"development_median_metrics.csv",index=False)
    old = MonthlyRentIndex().fit(fit.MONTH_INDEX,fit[TARGET])
    levels = []
    for protocol in ("fixed","monthly_update"):
        for spec in ("macro","no_macro","linear12","last","anchored_macro","anchored_history"):
            monthly = protocol == "monthly_update"
            if spec in ("macro","no_macro"):
                df = trajectory(history,end,alpha,spec=="macro",observed if monthly else None)
            elif spec.startswith("anchored"):
                df = anchored(history,end,half,spec=="anchored_macro",observed if monthly else None)
            else:
                rows = []
                for t in pd.period_range(cutoff+1,end,freq="M"):
                    seen = observed.loc[:t-1] if monthly else history
                    delta = t.ordinal - seen.index.max().ordinal
                    p = seen.iloc[-1] if spec=="last" else np.polyval(
                        np.polyfit(np.arange(12),seen.iloc[-12:],1),11+delta)
                    rows.append(dict(month=str(t),forecast=p))
                df = pd.DataFrame(rows)
            df["observed"] = [observed[pd.Period(t,"M")] for t in df.month]
            levels.append(df.assign(protocol=protocol,method=spec))
    levels = pd.concat(levels,ignore_index=True)
    levels.to_csv(DEST/"monthly_holdout.csv",index=False)
    summary = [dict(protocol=p,method=m,**regression_metrics(g.observed,g.forecast))
               for (p,m),g in levels.groupby(["protocol","method"])]
    pd.DataFrame(summary).to_csv(DEST/"median_metrics.csv",index=False)
    template = levels.pivot(index="month",columns=["protocol","method"],values="forecast")
    linear = old.transform(valid.MONTH_INDEX)
    assert np.allclose(template["fixed","linear12"].loc[months.astype(str)],linear)
    # The original and new indices coincide on fit months. Only the validation
    # scale changes. Reuse row predictions if they exist; scalar RMSE is insufficient.
    records=[]
    archive = RESULTS_DIR / "predictions"
    cache = DEST / "ratios.parquet"
    ratios = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(index=valid.ROW_ID)
    assert np.array_equal(ratios.index,valid.ROW_ID)
    for name in REPORT_MODELS:
        old_ref = reference(name,"index")
        if name not in ratios:
            logged = archive / (old_ref["source"]+".parquet")
            if logged.exists():
                preds = pd.read_parquet(logged).set_index("ROW_ID").loc[valid.ROW_ID,"y_pred"].to_numpy()
            else:
                preds,_,_ = fit_predict(MODELS[name],fit,valid,feature_columns(BASE),"index")
            ratios[name] = preds / linear
            tmp = cache.with_suffix(".tmp")
            ratios.to_parquet(tmp); tmp.replace(cache)
        records.append(dict(model=name,protocol="archived",method="raw",**reference(name,"raw")))
        records.append(dict(model=name,protocol="archived",method="linear12",**old_ref))
        for (protocol,method),level in template.items():
            pred = ratios[name].to_numpy() * level.loc[months.astype(str)].to_numpy()
            records.append(dict(model=name,protocol=protocol,method=method,
                                **regression_metrics(valid[TARGET],pred)))
        pd.DataFrame(records).to_csv(DEST/"rent_metrics.csv",index=False)
        print(name,flush=True)
    horizon = pd.Period("2026-07","M")
    future = trajectory(observed,horizon,alpha,True)
    future["anchored_macro"] = anchored(observed,horizon,half,True).forecast
    future.to_csv(DEST/"fixed_origin_2025_03_future_index.csv",index=False)
    dated = []
    for t in pd.period_range(FIRST,horizon,freq="M"):
        _,meta=asof(t,t.start_time)
        dated.append(dict(month=str(t),**meta))
    pd.DataFrame(dated).to_csv(DEST/"asof_monthly_forecasts.csv",index=False)
    (DEST/"settings.json").write_text(json.dumps(dict(alpha=alpha,half_life_months=half,fit_months=len(history)-12,
        fit_end=str(cutoff),macro_series=["annual real GDP growth","annual CPI-All Items inflation"],
        grid_origins=sorted(grid.origin.unique()),fixed_fit_rows=len(fit),valid_rows=len(valid)),indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        run()
