"""
train_xgboost.py
XGBoost cost-prediction model with small-sample discipline:
shallow trees, strong regularization, 5-fold CV (no single holdout,
since n is only ~70-90 rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from features import FEATURE_COLUMNS, TARGET

SMALL_N_PARAMS = dict(
    n_estimators=300,
    max_depth=3,           # shallow: guards against memorizing a small dataset
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,        # L2 regularization
    min_child_weight=3,
    random_state=42,
)


def cross_validate_xgb(df: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                       params: dict | None = None) -> dict:
    """params overrides SMALL_N_PARAMS keys (used by the website sliders)."""
    run_params = {**SMALL_N_PARAMS, **(params or {})}
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes, rmses, r2s = [], [], []
    oof_pred = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X):
        model = XGBRegressor(**run_params)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        oof_pred[test_idx] = pred
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
        r2s.append(r2_score(y[test_idx], pred))

    final_model = XGBRegressor(**run_params).fit(X, y)
    importances = dict(zip(FEATURE_COLUMNS,
                           final_model.feature_importances_.astype(float).tolist()))
    return {
        "model_name": "XGBoost",
        "cv_mae_mean": float(np.mean(maes)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_r2_mean": float(np.mean(r2s)),
        "oof_predictions": oof_pred,
        "model": final_model,
        "feature_importances": importances,
    }


def shap_contributions(model: XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Per-row SHAP values via XGBoost's native TreeSHAP (pred_contribs).
    Returns array of shape (n_rows, n_features + 1); the last column is the
    bias (base value). Each value is that feature's push, in pesos, on the
    prediction for that specific row - the mathematically grounded answer to
    'why did the model expect this cost?'"""
    import xgboost as xgblib
    dm = xgblib.DMatrix(np.asarray(X, dtype=float),
                        feature_names=FEATURE_COLUMNS)
    return model.get_booster().predict(dm, pred_contribs=True)