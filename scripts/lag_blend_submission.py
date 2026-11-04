"""Build the five model repeat-rent blend for Kaggle. See docs/repeat_rent_blend.md."""

import sys, json, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from src.features import load_modeling_data, feature_columns
from src.evaluation import MonthlyRentIndex
tr,te=load_modeling_data()
raw=pd.read_csv('data/train.csv').iloc[tr.ROW_ID.values].reset_index(drop=True)
for c in ['BLOCK','STREET','FLAT_TYPE']: raw[c]=raw[c].astype(str).str.lower().str.strip()
raw['FLAT_TYPE']=raw.FLAT_TYPE.str.replace('-',' ')
tr['BT']=(raw.BLOCK+'|'+raw.STREET+'|'+raw.FLAT_TYPE).values
cols=feature_columns('G0+G1');Y='MONTHLY_RENT'
P=dict(json.load(open('configs/lightgbm_tuned.json'))['params'],random_state=5228,verbose=-1,n_jobs=-1)
rmse=lambda a,b:float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def lag_feats(hist,dst,cut):
    # hist: rows with 'a', known; only rows with month<dst month and <=cut used
    out=pd.DataFrame(index=dst.index);h=hist[['BT','MONTH_INDEX','a']].copy()
    d=dst[['BT','MONTH_INDEX']].reset_index()
    for name,ks in {'L12':(11,12,13),'L24':(23,24,25),'L36':(35,36,37),'OFF':(6,7,8,9,10,14,15,16,17,18,19,20,21,22,26,27,28,29,30)}.items():
        parts=[]
        for k in ks:
            t=h.assign(MONTH_INDEX=h.MONTH_INDEX+k);t=t[t.MONTH_INDEX-k<=cut];parts.append(t)
        t=pd.concat(parts).groupby(['BT','MONTH_INDEX']).a.agg(['mean','count','min','max']).reset_index()
        j=d.merge(t,on=['BT','MONTH_INDEX'],how='left').set_index('index')
        out[name+'_mean']=j['mean'];out[name+'_n']=j['count'];out[name+'_min']=j['min'];out[name+'_max']=j['max']
    return out.loc[dst.index]
rt=pd.read_csv('data/test.csv').iloc[te.ROW_ID.values].reset_index(drop=True)
for c in ['BLOCK','STREET','FLAT_TYPE']: rt[c]=rt[c].astype(str).str.lower().str.strip()
rt['FLAT_TYPE']=rt.FLAT_TYPE.str.replace('-',' ')
te['BT']=(rt.BLOCK+'|'+rt.STREET+'|'+rt.FLAT_TYPE).values
ix=MonthlyRentIndex().fit(tr.MONTH_INDEX,tr[Y]);tr['a']=tr[Y].values/ix.transform(tr.MONTH_INDEX);iv=ix.transform(te.MONTH_INDEX)
cut=int(tr.MONTH_INDEX.max());tf=lag_feats(tr,tr,cut);tv=lag_feats(tr,te,cut)
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
F,V=pd.concat([tr[cols],tf],axis=1),pd.concat([te[cols],tv],axis=1)
cat=[c for c in cols if str(tr[c].dtype)=='category']
def num(X):
    X=X.copy()
    for c in cat: X[c]=X[c].cat.codes
    return X
XP=json.load(open('configs/xgboost_tuned.json'))['params'];CP=json.load(open('configs/catboost_tuned.json'))['params']
Pr={**P,'num_leaves':31,'min_child_samples':200,'feature_fraction':.5,'n_estimators':800,'random_state':4}
preds={'base':lgb.LGBMRegressor(**P).fit(tr[cols],tr.a).predict(te[cols])*iv,
 'lag_reg':lgb.LGBMRegressor(**Pr).fit(F,tr.a).predict(V)*iv,
 'xgb':xgb.XGBRegressor(**{**XP,'tree_method':'hist','random_state':6,'n_jobs':-1}).fit(num(F),tr.a).predict(num(V))*iv,
 'cat':CatBoostRegressor(**{**CP,'iterations':800,'random_seed':7,'verbose':0}).fit(num(F),tr.a).predict(num(V))*iv,
 'rf':RandomForestRegressor(n_estimators=200,min_samples_leaf=5,max_features=.33,n_jobs=-1,random_state=8).fit(num(F).fillna(-1),tr.a).predict(num(V).fillna(-1))*iv}
out=pd.DataFrame({'Id':te.ROW_ID.values,'Predicted':np.mean(list(preds.values()),axis=0)}).sort_values('Id')
out.to_csv('submissions/lag_blend5_G0G1_index.csv',index=False)
print(out.shape,out.Id.is_unique,out.Predicted.describe().round(1).to_dict(),{k:round(float(v.mean()),1) for k,v in preds.items()})
old=pd.read_csv('submissions/lightgbm_lag_blend_G0G1_index.csv');print('corr with 419.71 file',np.corrcoef(old.Predicted,out.Predicted)[0,1],'rms diff',rmse(old.Predicted,out.Predicted))
