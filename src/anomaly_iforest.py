"""
anomaly_iforest.py
Anomaly scoring with three independent lenses:

  1. Isolation Forest  - primary method: features + abc_ratio + cost residual.
  2. Local Outlier Factor (LOF) - second unsupervised opinion; a project
     flagged by BOTH methods is a stronger signal than either alone.
  3. Univariate auditor's baseline - the simple rule an auditor could apply
     in a spreadsheet: cost per m2 more than 1.5 SD above the dataset median.
     Reported so the thesis can show what ML adds beyond the simple rule.

Contamination sensitivity: flags at two thresholds (HIGH / MODERATE) so the
thesis does not hinge on one arbitrary setting.

IMPORTANT FRAMING: a flag is a *statistical* deviation, not evidence of
irregularity. Legitimate causes (terrain, right-of-way, ancillary works)
are common.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, ANOMALY_EXTRA_COLUMNS

CONTAMINATION_LEVELS = (0.05, 0.10)
BASELINE_SD = 1.5  # auditor's rule: cost/m2 > median + 1.5 SD


def score_anomalies(df: pd.DataFrame, oof_predictions: np.ndarray,
                    seed: int = 42,
                    contamination_levels: tuple = CONTAMINATION_LEVELS) -> pd.DataFrame:
    """contamination_levels: (high_threshold, moderate_threshold), high < moderate.
    Attaches the fitted scaler and primary forest to the returned frame's attrs
    so the website can score hypothetical new projects."""
    contamination_levels = tuple(sorted(contamination_levels))
    out = df.copy()
    out["predicted_cost_php"] = oof_predictions
    out["residual_php"] = out["cost_deflated_php"] - out["predicted_cost_php"]
    out["residual_pct"] = 100.0 * out["residual_php"] / out["predicted_cost_php"]

    # residual_pct (not raw pesos) so a 60% overprice on a small road counts
    # as much as on a large one - scale-invariant anomaly signal.
    anomaly_cols = (FEATURE_COLUMNS
                    + [c for c in ANOMALY_EXTRA_COLUMNS if c in out.columns]
                    + ["residual_pct"])
    scaler = StandardScaler()
    X = scaler.fit_transform(out[anomaly_cols].to_numpy(dtype=float))

    # ---- 1. Isolation Forest (primary) -----------------------------------
    flag_cols = []
    for cont in contamination_levels:
        iso = IsolationForest(n_estimators=300, contamination=cont,
                              random_state=seed)
        labels = iso.fit_predict(X)               # -1 = anomaly
        col = f"flag_c{int(round(cont*100)):02d}"
        flag_cols.append(col)
        out[col] = (labels == -1)
        if cont == contamination_levels[0]:
            # score_samples: lower = more anomalous. Negate so higher = worse.
            out["anomaly_score"] = -iso.score_samples(X)
            out.attrs["iso_model"] = iso
            out.attrs["scaler"] = scaler
            out.attrs["anomaly_cols"] = anomaly_cols

    out["flag_level"] = np.select(
        [out[flag_cols[0]], out[flag_cols[1]]],
        ["HIGH", "MODERATE"],
        default="NONE",
    )

    # ---- 2. LOF second opinion -------------------------------------------
    n_neighbors = int(min(20, max(5, len(out) // 4)))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=contamination_levels[1])
    out["lof_flag"] = (lof.fit_predict(X) == -1)
    out["flagged_by_both"] = out["lof_flag"] & (out["flag_level"] != "NONE")

    # ---- 3. Univariate auditor's baseline --------------------------------
    cps = out["cost_per_sqm_php"]
    med, sd = cps.median(), cps.std()
    out["baseline_flag"] = cps > (med + BASELINE_SD * sd)

    return out