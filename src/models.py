"""Model factories for the shared interface.

Add your model as a zero-argument function that returns an object with
`fit(X, y)` and `predict(X)`, then register it in MODELS so the command line
runner can find it. Use SEED for any random state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import SEED


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


def make_ridge():
    """Ridge (alpha 1) on every column, one hot categories, scaled numerics.

    A simple linear reference for the split comparison. Phase 1 linear models
    with chosen columns belong to their owner and can replace it.
    """

    from sklearn.linear_model import Ridge

    return sklearn_tabular(Ridge(alpha=1.0))


MODELS = {
    "global_median": make_global_median,
    "lightgbm_default": make_lightgbm,
    "ridge_default": make_ridge,
}
