"""
features.py
Feature engineering for the cost-prediction and anomaly-detection models.

PLACEHOLDER DEFLATOR: `YEAR_PRICE_INDEX` below is a placeholder.
Replace with actual PSA CMWPI annual averages (psa.gov.ph) before final runs,
and cite the PSA table used. Values here are illustrative only so the pipeline
runs end-to-end during development.
"""

from __future__ import annotations

import pandas as pd

# TODO(Enrique): replace with verified PSA CMWPI annual averages, base year 2019 = 100.
YEAR_PRICE_INDEX = {2019: 100.0, 2020: 101.5, 2021: 104.0,
                    2022: 110.0, 2023: 114.0, 2024: 117.0}

FEATURE_COLUMNS = [
    "geometry_sqm",
    "surface_thickness_mm",
    "duration_days",
    "publish_year",
    "scope_flag_ancillary",
    "surface_is_pccp",
]
# Anomaly-stage-only features. abc_ratio (contract / ABC) is a bid-behavior
# signal deliberately EXCLUDED from the cost regression: it contains the
# target in its numerator, so using it there would leak the answer and make
# residual-based anomaly detection meaningless.
ANOMALY_EXTRA_COLUMNS = ["abc_ratio"]
TARGET = "cost_deflated_php"


def build_features(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Take QC-passed rows and return a modeling table."""
    df = qc_df.copy()
    df = df[df["quality_tier"].isin(["COMPLETE", "USABLE_PARTIAL", "USABLE_SPARSE"])]

    # Deflate contract cost to base-year pesos.
    df["price_index"] = df["publish_year"].map(YEAR_PRICE_INDEX)
    if df["price_index"].isna().any():
        missing_years = sorted(df.loc[df["price_index"].isna(), "publish_year"].unique())
        raise ValueError(f"No price index for years: {missing_years}. Update YEAR_PRICE_INDEX.")
    df[TARGET] = df["contract_amount_php"] * 100.0 / df["price_index"]

    # Simple encodings / imputations, kept transparent for the thesis write-up.
    df["surface_is_pccp"] = (df["surface_type"] == "PCCP").astype(int)
    df["scope_flag_ancillary"] = df["scope_flag_ancillary"].astype(int)
    med_thick = df["surface_thickness_mm"].median()
    df["thickness_imputed"] = df["surface_thickness_mm"].isna().astype(int)
    df["surface_thickness_mm"] = df["surface_thickness_mm"].fillna(med_thick)
    df["duration_days"] = df["duration_days"].fillna(df["duration_days"].median())

    # Interpretability helper (not a model feature; derived from target).
    df["cost_per_sqm_php"] = df["contract_amount_php"] / df["geometry_sqm"]

    # Bid-behavior signal for the anomaly stage: contract amount vs ABC.
    # ~1.0 is typical; >1.0 is legally prohibited under RA 9184.
    import numpy as np
    df["abc_ratio"] = np.where(df["abc_php"] > 0,
                               df["contract_amount_php"] / df["abc_php"], np.nan)
    df["abc_ratio"] = df["abc_ratio"].fillna(df["abc_ratio"].median())

    return df.reset_index(drop=True)