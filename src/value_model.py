"""Fair-price model: predicts what a phone 'should' cost from its specs.

value_gap = (predicted - actual) / predicted. Positive means cheaper than its specs suggest.
Out-of-fold predictions are used so a phone is never priced by a model that saw its own price.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SPEC_FEATURES


def candidate_models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                       subsample=0.8, random_state=0),
    }


def fit_value_model(phones):
    X, y = phones[SPEC_FEATURES], np.log(phones.price_inr)
    est = candidate_models()["Gradient Boosting"]
    oof = np.exp(cross_val_predict(est, X, y, cv=KFold(5, shuffle=True, random_state=0)))
    est.fit(X, y)
    return est, oof


def explain_price(est, phones, phone_id):
    """Local explanation: % effect of each spec on predicted price versus a typical phone.

    Each spec is swapped for the catalogue median, one at a time, and the change in the
    model's predicted (log) price is measured (an occlusion-style attribution).
    """
    X = phones[SPEC_FEATURES]
    row = X[phones.phone_id == phone_id].iloc[[0]]
    base = est.predict(row)[0]
    med = X.median()
    effects = {}
    for f in SPEC_FEATURES:
        r = row.copy()
        r[f] = med[f]
        effects[f] = (np.exp(base - est.predict(r)[0]) - 1) * 100
    return pd.Series(effects).sort_values()
