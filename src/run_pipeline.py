"""
run_pipeline.py
End-to-end orchestrator:
  raw CSV -> QC -> features -> MLR + XGBoost (5-fold CV) -> Isolation Forest
  -> data/processed/results.csv + metrics.json (consumed by viz_dash_app.py)

Usage:
  python src/run_pipeline.py --input data/sample/sample_projects_SYNTHETIC.csv
  python src/run_pipeline.py --input data/raw/albay_projects.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpwh_schema_qc import run_qc
from features import build_features
from train_mlr_baseline import cross_validate_mlr
from train_xgboost import cross_validate_xgb
from anomaly_iforest import score_anomalies

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main(input_csv: Path) -> None:
    # 1. Quality control
    qc_df = run_qc(input_csv, PROCESSED)
    usable = qc_df[qc_df["quality_tier"] != "UNUSABLE"]

    # 2. Features
    feat_df = build_features(usable)
    print(f"\nModeling rows: {len(feat_df)}")

    # 3. Models (both reported - no cherry-picking)
    mlr = cross_validate_mlr(feat_df)
    xgb = cross_validate_xgb(feat_df)
    for m in (mlr, xgb):
        print(f"{m['model_name']:>8}:  MAE {m['cv_mae_mean']:,.0f}  "
              f"RMSE {m['cv_rmse_mean']:,.0f}  R2 {m['cv_r2_mean']:.3f}")

    # 4. Anomaly scoring on XGBoost out-of-fold residuals
    results = score_anomalies(feat_df, xgb["oof_predictions"])
    n_high = int((results["flag_level"] == "HIGH").sum())
    n_mod = int((results["flag_level"] == "MODERATE").sum())
    print(f"\nFlags: HIGH={n_high}  MODERATE={n_mod}  of {len(results)} projects")

    # 5. Persist for the website
    results.to_csv(PROCESSED / "results.csv", index=False)
    metrics = {
        "n_projects": len(results),
        "flags_high": n_high,
        "flags_moderate": n_mod,
        "mlr": {k: mlr[k] for k in ("cv_mae_mean", "cv_rmse_mean", "cv_r2_mean")},
        "mlr_coefficients": mlr["coefficients"],
        "xgboost": {k: xgb[k] for k in ("cv_mae_mean", "cv_rmse_mean", "cv_r2_mean")},
        "xgboost_feature_importances": xgb["feature_importances"],
    }
    (PROCESSED / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {PROCESSED/'results.csv'} and metrics.json")
    print("Launch the website:  python src/viz_dash_app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    main(ap.parse_args().input)
