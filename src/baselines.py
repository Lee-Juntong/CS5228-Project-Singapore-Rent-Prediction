"""Member B: the first six ladder items (seven fitted estimators)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import SEED
from src.models import make_global_median, sklearn_tabular

LINEAR_DROP = (
    "FLOOR_AREA_SQM", "LOG_FLOOR_AREA_SQM", "FLOOR_AREA_PER_ROOM_PROXY",
    "FLAT_TYPE_ORDINAL", "LEASE_COMMENCE_DATE",
)
KNN_COLUMNS = ("HDB_LATITUDE", "HDB_LONGITUDE", "FLOOR_AREA_SQM", "MONTH_INDEX")


class SelectColumns(TransformerMixin, BaseEstimator):
    def __init__(self, columns=None, drop=()):
        self.columns = columns
        self.drop = drop

    def fit(self, X, y=None):
        self.columns_ = list(self.columns) if self.columns is not None else [c for c in X if c not in self.drop]
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model inputs: {sorted(missing)}")
        return self

    def transform(self, X):
        return X.loc[:, self.columns_]


class GroupMedian(RegressorMixin, BaseEstimator):
    """Exact calendar quarter lookup with a training-only fallback hierarchy."""

    levels = (("TOWN", "FLAT_TYPE", "QUARTER"), ("TOWN", "FLAT_TYPE"),
              ("FLAT_TYPE",), ("TOWN",))

    @staticmethod
    def keys(X):
        frame = X[["TOWN", "FLAT_TYPE"]].astype(object).copy()
        frame = frame.fillna("__MISSING__")
        frame["QUARTER"] = np.floor(X["MONTH_INDEX"].to_numpy(float) / 3).astype(int)
        return frame

    def fit(self, X, y):
        frame = self.keys(X)
        frame["rent"] = np.asarray(y, dtype=float)
        self.global_median_ = float(np.median(y))
        self.tables_ = [frame.groupby(list(level), observed=True)["rent"].median() for level in self.levels]
        return self

    def predict_with_levels(self, X):
        frame = self.keys(X)
        prediction = np.full(len(X), np.nan)
        used = np.full(len(X), "global", dtype=object)
        for level, table in zip(self.levels, self.tables_):
            key = pd.MultiIndex.from_frame(frame[list(level)]) if len(level) > 1 else pd.Index(frame[level[0]])
            candidate = table.reindex(key).to_numpy(float)
            fill = np.isnan(prediction) & np.isfinite(candidate)
            prediction[fill] = candidate[fill]
            used[fill] = "+".join(level)
        prediction[np.isnan(prediction)] = self.global_median_
        return prediction, used

    def predict(self, X):
        return self.predict_with_levels(X)[0]


def make_group_median():
    return GroupMedian()


def make_linear():
    return Pipeline([("columns", SelectColumns(drop=LINEAR_DROP)),
                     ("tabular", sklearn_tabular(LinearRegression(), onehot_drop="first"))])


def make_baseline_ridge():
    return Pipeline([("columns", SelectColumns(drop=LINEAR_DROP)),
                     ("tabular", sklearn_tabular(Ridge(alpha=10.0, solver="lsqr", tol=1e-6,
                                                      max_iter=10000), onehot_drop="first"))])


def make_knn():
    return Pipeline([("columns", SelectColumns(columns=KNN_COLUMNS)),
                     ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()),
                     ("model", KNeighborsRegressor(n_neighbors=50, weights="distance",
                                                   algorithm="kd_tree", p=2, n_jobs=8))])


def make_decision_tree():
    return sklearn_tabular(DecisionTreeRegressor(max_depth=16, min_samples_leaf=20,
                                                random_state=SEED), scale=False)


def make_random_forest():
    return sklearn_tabular(RandomForestRegressor(n_estimators=300, max_depth=20,
                          min_samples_leaf=5, max_features=0.7, bootstrap=True,
                          random_state=SEED, n_jobs=8), scale=False, dense=True)


BASELINE_MODELS = {
    "global_median": make_global_median,
    "group_median": make_group_median,
    "linear_regression": make_linear,
    "ridge": make_baseline_ridge,
    "knn": make_knn,
    "decision_tree": make_decision_tree,
    "random_forest": make_random_forest,
}

MODEL_NOTES = {
    "global_median": "Training median; no predictors.",
    "group_median": "Town/type/absolute calendar quarter; fallback town+type, type, town, global.",
    "linear_regression": "G0+G1 minus all area-derived columns, ordinal type and redundant lease year; retain one-hot type.",
    "ridge": "Same columns as linear_regression; alpha=10 fixed without tuning.",
    "knn": "Only scaled latitude, longitude, area and month; k=50, distance weights.",
    "decision_tree": "G0+G1; depth=16, minimum leaf=20.",
    "random_forest": "G0+G1; 300 trees, depth=20, minimum leaf=5, max_features=0.7.",
}
