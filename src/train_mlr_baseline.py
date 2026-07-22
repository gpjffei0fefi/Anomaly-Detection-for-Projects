"""
train_mlr_baseline.py
Multiple Linear Regression baseline with 5-fold cross-validation.

Reported alongside XGBoost. Antoniou & Aretoulis (2025) found MLR can
outperform XGBoost on comparable highway cost data - the thesis reports
the comparison honestly either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import FEATURE_COLUMNS, TARGET


def cross_validate_mlr(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> dict:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes, rmses, r2s = [], [], []
    oof_pred = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        oof_pred[test_idx] = pred
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
        r2s.append(r2_score(y[test_idx], pred))

    final_model = LinearRegression().fit(X, y)
    return {
        "model_name": "MLR",
        "cv_mae_mean": float(np.mean(maes)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_r2_mean": float(np.mean(r2s)),
        "oof_predictions": oof_pred,
        "model": final_model,
        "coefficients": dict(zip(FEATURE_COLUMNS, final_model.coef_.tolist())),
        "intercept": float(final_model.intercept_),
    }


def cross_validate_mlr_log(df: pd.DataFrame, n_splits: int = 5,
                           seed: int = 42) -> dict:
    """MLR on log(cost). Construction costs are right-skewed, so a log target
    often fits better. Metrics are computed on the ORIGINAL peso scale after
    exp() back-transform, so they are directly comparable to the other models."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    ylog = np.log(y)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes, rmses, r2s = [], [], []
    oof_pred = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], ylog[train_idx])
        pred = np.exp(model.predict(X[test_idx]))
        oof_pred[test_idx] = pred
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
        r2s.append(r2_score(y[test_idx], pred))

    return {
        "model_name": "MLR (log cost)",
        "cv_mae_mean": float(np.mean(maes)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_r2_mean": float(np.mean(r2s)),
        "oof_predictions": oof_pred,
    }