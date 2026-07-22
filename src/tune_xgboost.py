"""
tune_xgboost.py
Exhaustive grid search for XGBoost hyperparameters with 5-fold CV.
Run this once your REAL dataset is loaded; report the winning parameters in
the thesis methodology chapter instead of defending arbitrary defaults.

On 70-90 rows the full grid (~144 fits x 5 folds) takes a few minutes.

Usage:
  python src/tune_xgboost.py --input data/raw/albay_projects.csv
  python src/tune_xgboost.py --input data/sample/sample_projects_SYNTHETIC.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from dpwh_schema_qc import run_qc
from features import build_features, FEATURE_COLUMNS, TARGET

PARAM_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.10],
    "n_estimators": [150, 300],
    "min_child_weight": [1, 3],
    "reg_lambda": [1.0, 2.0],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


def main(input_csv: Path) -> None:
    qc = run_qc(input_csv, Path("data/processed"))
    feat = build_features(qc[qc["quality_tier"] != "UNUSABLE"])
    X = feat[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = feat[TARGET].to_numpy(dtype=float)
    print(f"\nTuning on {len(feat)} rows, "
          f"{int(np.prod([len(v) for v in PARAM_GRID.values()]))} combinations x 5 folds...")

    search = GridSearchCV(
        XGBRegressor(random_state=42),
        PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    search.fit(X, y)

    print("\nBest parameters (report these in the thesis):")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"\nBest CV RMSE: PHP {-search.best_score_:,.0f}")

    # Fold-level variance of the winner - report this too (small-n honesty).
    idx = search.best_index_
    fold_scores = [-search.cv_results_[f"split{i}_test_score"][idx]
                   for i in range(5)]
    print(f"Per-fold RMSE: {[f'{s:,.0f}' for s in fold_scores]}")
    print(f"RMSE std across folds: PHP {np.std(fold_scores):,.0f} "
          f"(report as +/- uncertainty)")
    print("\nCopy the best parameters into SMALL_N_PARAMS in src/train_xgboost.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    main(ap.parse_args().input)
