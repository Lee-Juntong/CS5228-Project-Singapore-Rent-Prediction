"""Verify experiment coverage and build the local results overview."""
import platform
import hashlib
import json
from importlib.metadata import version

import numpy as np
import pandas as pd

from scripts.baseline_extensions import OUT, BASELINE_MODELS, RESULTS_DIR, LOG
from src.config import DATA_DIR, PROCESSED_DIR
from scripts.suspicious_row_ablation import FLAGS
from src.evaluation import time_holdout_masks


def table(df):
    def render(x):
        if pd.isna(x):
            return ""
        return f"{x:.2f}" if isinstance(x, (float,np.floating)) else str(x)
    return "\n".join("| "+" | ".join(render(c) for c in row)+" |"
                     for row in [df.columns,["---"]*len(df.columns),*df.itertuples(index=False,name=None)])


def main():
    text = ["# Member B experiment results", "",
            "RMSE is in SGD per rental unless the table explicitly scores monthly medians. Lower is better. "
            "Each experiment CSV records the source of reused rows. Figures are rounded here only."]
    for name, expected, cases, title in [
        ("features",28,["G0","G0+G1","G0+G1+G2","ALL"],"Feature groups, raw target"),
        ("outliers",28,["all","m1","m2","both"],"Dropping flagged fitting rows, index target"),
        ("lags",14,["base","plus16"],"Repeat-rent features, index target"),
    ]:
        df=pd.read_csv(OUT/(name+".csv"))
        assert len(df)==expected and not df.duplicated(["model","case"]).any()
        assert set(df.model)==set(BASELINE_MODELS) and (df.n_valid==17037).all()
        wide=df.pivot(index="model",columns="case",values="rmse").loc[list(BASELINE_MODELS),cases]
        if name == "features":
            for other in ("lightgbm_tuned","catboost_tuned"):
                ref=LOG[(LOG.model==other)&(LOG.split=="time")&(LOG.target_mode=="raw")]
                wide.loc[other+" (archived)"]=[ref.loc[ref.feature_set==c].iloc[-1].rmse for c in cases]
        if name == "outliers":
            ref=LOG[(LOG.split=="time")&(LOG.target_mode=="index")&(LOG.feature_set=="G0+G1")]
            keys=["lightgbm_tuned"]+["lightgbm_tuned_no_"+c for c in cases[1:]]
            wide.loc["lightgbm_tuned (archived)"]=[ref.loc[ref.model==key].iloc[-1].rmse for key in keys]
        text.extend(["","## "+title,"",table(wide.reset_index())])
    clean=pd.read_csv(OUT/"outliers_fit_only.csv")
    assert len(clean)==32 and not clean.duplicated(["model","case"]).any() and (clean.n_valid==17037).all()
    prospective=clean.pivot(index="model",columns="case",values="rmse")[["all","m1","m2","both"]]
    text.extend(["","## Outlier sensitivity with strictly training-only thresholds","",
                 "All rules are estimated again on fitting rows only. The tuned LightGBM control uses the same flags.",
                 "",table(prospective.reset_index())])
    med=pd.read_csv(OUT/"median_lag_variants.csv")
    assert len(med)==2
    text.extend(["","The fixed median definitions ignore added columns. Separate lookup variants give "
                 "the L24 mean a weight of n/(n+5), with n earlier matching transactions and "
                 "the ordinary median model providing the remainder.","",table(med[["model","rmse","with_history"]])])
    blend=pd.read_csv(RESULTS_DIR/"member_a/repeat_rent/validation.csv").query("window == 'holdout'")
    text.extend(["","Previously scored holdout comparisons. The lagged LightGBM also changes regularisation, "
                 "and the five-model blend changes the estimator combination.", "",table(blend[["model","rmse"]])])
    horizon=pd.read_csv(OUT/"horizon.csv")
    assert len(horizon)==140 and not horizon.duplicated(["model","case","mode"]).any()
    older=pd.read_csv(RESULTS_DIR/"member_a/trend/horizon_backtest.csv")
    controls=older[older.model.isin(["lightgbm_tuned","xgboost_tuned"])].copy()
    controls["case"]=controls.origin+"/"+controls.band
    both=pd.concat([horizon,controls.rename(columns={"target_mode":"mode"})],ignore_index=True)
    rows=[]
    for (name,mode),g in both.groupby(["model","mode"]):
        by=g[g["case"].str.endswith("/7-16")]
        rows.append(dict(model=name,mode=mode,near=np.average(g.loc[g["case"].str.endswith("/1-6"),"rmse"],
                     weights=None),far=by.rmse.mean()))
    av=pd.DataFrame(rows).pivot(index="model",columns="mode",values=["near","far"])
    av.columns=["_".join(c) for c in av.columns]
    text.extend(["","## Earlier origins","","Equal-weight mean of the five window RMSEs, so overlapping windows "
                 "are not independent observations. Near covers months 1–6. Far covers months 7–16 "
                 "except the last origin, where labels end at month 12. Per-origin counts, bias and scores: horizon.csv.",
                 "",table(av.reset_index())])
    dest=OUT/"macro_index"
    med=pd.read_csv(dest/"median_metrics.csv")
    rent=pd.read_csv(dest/"rent_metrics.csv")
    assert len(med)==12 and rent.model.nunique()==13 and len(rent)==13*14
    assert np.isfinite(med.rmse).all() and np.isfinite(rent.rmse).all()
    monthly=pd.read_csv(dest/"monthly_holdout.csv")
    assert len(monthly)==72 and not monthly.duplicated(["protocol","method","month"]).any()
    for (protocol,method), frame in monthly.groupby(["protocol","method"]):
        expected=med.loc[(med.protocol==protocol)&(med.method==method),"rmse"].item()
        assert np.isclose(np.sqrt(np.mean((frame.observed-frame.forecast)**2)),expected)
        for series in ("gdp","cpi_all"):
            usable=frame.dropna(subset=[series+"_release"])
            released=pd.to_datetime(usable[series+"_release"])
            assert (released < pd.to_datetime(usable.month)).all()
            if protocol=="fixed":
                assert (released < pd.Timestamp("2024-10-01")).all()
    for n in BASELINE_MODELS:
        masks=pd.read_csv(OUT/"outliers.csv").query("model == @n").set_index("case")
        assert masks.dropped.to_dict()=={"all":0,"m1":276,"m2":44,"both":10}
    text.extend(["","## Macro index: monthly median RMSE","",
                 table(med.pivot(index="method",columns="protocol",values="rmse").reset_index()),
                 "","## Macro index: per-rental RMSE on the same 17,037 holdout rows",""])
    pairs=rent[rent.protocol=="fixed"].pivot(index="model",columns="method",values="rmse")
    archive=rent[rent.protocol=="archived"].pivot(index="model",columns="method",values="rmse")
    pairs.insert(0,"archived_raw",archive.raw)
    pairs.insert(1,"archived_index",archive.linear12)
    text.extend([table(pairs.reset_index()),"",
                "The paired linear12 column scores the same local model and predictions as the alternative "
                "indices. Compare each alternative with that column to isolate the index change. "
                "Monthly-update scores by rental and all metrics are in macro_index/rent_metrics.csv."])
    env=["numpy","pandas","scikit-learn","lightgbm","xgboost","catboost","pyarrow"]
    text.extend(["","Environment: Python "+platform.python_version()+", "+", ".join(k+" "+version(k) for k in env)+"."])
    (OUT/"RESULTS.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    inputs=[PROCESSED_DIR/"train_features.parquet",DATA_DIR/"train.csv",
            DATA_DIR/"official_forecasts/sg_forecast_vintages.csv", RESULTS_DIR/"results.csv",
            RESULTS_DIR/"member_a/trend/horizon_backtest.csv"]
    # Quantify the difference between reproducing the published labels and
    # learning new outlier thresholds from the fit period alone.
    flags_input=pd.read_parquet(inputs[0],columns=["RENT_APPROVAL_DATE","FLAT_TYPE","TOWN",
                                                  "FLOOR_AREA_SQM","MONTHLY_RENT"])
    fm,_=time_holdout_masks(flags_input)
    audit=[]
    for flag,(_,detector) in FLAGS.items():
        retrospective=detector(flags_input)[fm]
        prospective=detector(flags_input.loc[fm])
        audit.append(dict(flag=flag,published_fit_drops=int(retrospective.sum()),
                          fit_only_threshold_drops=int(prospective.sum()),
                          different_fit_flags=int((retrospective!=prospective).sum()),
                          all_rows_flags=int(detector(flags_input).sum())))
    pd.DataFrame(audit).to_csv(OUT/"flag_threshold_audit.csv",index=False)
    (OUT/"verification.json").write_text(json.dumps({
        "python":platform.python_version(),"packages":{k:version(k) for k in env},
        "input_sha256":{str(p.relative_to(DATA_DIR.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "coverage":{"feature_rows":28,"outlier_rows":28,"fit_only_outlier_rows":32,"lag_rows":14,"feature_aware_medians":2,
                    "horizon_rows":140,"macro_monthly_forecasts_scored":len(monthly),
                    "macro_rental_comparison_rows":len(rent)},
        "checked":"Unique configurations, validation counts, outlier fit counts, monthly score arithmetic, and forecast release cutoffs."
    },indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
