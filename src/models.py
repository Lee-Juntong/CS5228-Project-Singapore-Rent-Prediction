"""Model factories for the shared interface.

Add your model as a zero-argument function that returns an object with
`fit(X, y)` and `predict(X)`, then register it in MODELS so the command line
runner can find it. Pass random_state=SEED (from src.config) to every model that has one, or results will not reproduce. Torch models also need torch.manual_seed(SEED).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CONFIG_DIR, SEED


class GlobalMedian:
    """Predicts the median training rent for every row. The sanity floor."""

    def fit(self, X: pd.DataFrame, y) -> "GlobalMedian":
        self.value_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value_)


def sklearn_tabular(estimator, scale: bool = True) -> Pipeline:
    """Wrap a scikit-learn estimator so it accepts the shared X directly.

    Categorical columns are one hot encoded (unseen levels become all zeros).
    Numeric columns get median imputation and, when `scale` is true, standard
    scaling. Columns are detected by dtype at fit time.
    """

    def numeric(X):
        return [c for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype)]

    def categorical(X):
        return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]

    numeric_steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    transformer = ColumnTransformer(
        [
            ("num", make_pipeline(*numeric_steps), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("prep", transformer), ("model", estimator)])


def make_global_median():
    return GlobalMedian()


def make_lightgbm():
    """LightGBM with default settings. It reads pandas categories natively."""

    from lightgbm import LGBMRegressor

    return LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1)


class CatBoostModel:
    """CatBoost wrapper that declares categorical columns by dtype at fit time.

    CatBoost needs its categorical features named rather than inferred, and it
    cannot take NaN in a categorical column. We pick the pandas `category`
    columns of the incoming X (the same four across every feature set), fill
    their missing values with a constant string, and let CatBoost handle numeric
    NaN natively. The same columns are rebuilt on predict.
    """

    _CAT_FILL = "__missing__"

    def __init__(self, **params):
        self.params = params
        self.cat_features_: list[str] = []

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cat_features_:
            X[col] = X[col].astype("object").where(X[col].notna(), self._CAT_FILL).astype(str)
        return X

    def fit(self, X: pd.DataFrame, y) -> "CatBoostModel":
        from catboost import CatBoostRegressor

        self.cat_features_ = [
            c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)
        ]
        self.model_ = CatBoostRegressor(
            random_seed=SEED, verbose=0, allow_writing_files=False, **self.params
        )
        self.model_.fit(self._prepare(X), y, cat_features=self.cat_features_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._prepare(X)), dtype=float)


def make_catboost():
    """CatBoost with default settings, categorical columns declared by dtype."""

    return CatBoostModel()


def make_ridge():
    """Ridge (alpha 1) on every column, one hot categories, scaled numerics.

    A simple linear reference for the split comparison. Phase 1 linear models
    with chosen columns belong to their owner and can replace it.
    """

    from sklearn.linear_model import Ridge

    return sklearn_tabular(Ridge(alpha=1.0))


def _load_tuned_params(name: str) -> dict:
    """Read the params written by scripts/tune_boosting.py, or {} if absent.

    Falling back to an empty dict means a tuned factory reverts to library
    defaults until the tuner has run, so the model is always importable.
    """

    path = CONFIG_DIR / f"{name}_tuned.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("params", {})


def make_lightgbm_tuned():
    """LightGBM with the parameters from configs/lightgbm_tuned.json."""

    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        random_state=SEED, n_jobs=-1, verbose=-1, **_load_tuned_params("lightgbm")
    )


def make_catboost_tuned():
    """CatBoost with the parameters from configs/catboost_tuned.json."""

    return CatBoostModel(**_load_tuned_params("catboost"))


MODELS = {
    "global_median": make_global_median,
    "lightgbm_default": make_lightgbm,
    "catboost_default": make_catboost,
    "lightgbm_tuned": make_lightgbm_tuned,
    "catboost_tuned": make_catboost_tuned,
    "ridge_default": make_ridge,
}
