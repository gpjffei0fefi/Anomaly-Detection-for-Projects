"""
generate_synthetic_data.py
Generates a SYNTHETIC demo dataset so the full pipeline and website can be
developed and demonstrated before real PhilGEPS extraction is finished.

*** EVERY ROW IS FABRICATED. ***
Project titles, contractors, and costs are invented. Ref numbers use the
prefix SYN- so they can never be confused with real PhilGEPS references.
Replace with the real extraction (data/raw/) for actual thesis results.

Cost model used to fabricate data (illustrative only):
  cost ~ (unit_rate x area x thickness_factor) x noise, plus a fixed ancillary
  add-on when scope_flag_ancillary is true. ~7% of rows get an injected
  overprice multiplier so the anomaly detector has something to find in demos.

Usage: python src/generate_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N = 90

MUNICIPALITIES = ["Legazpi City", "Daraga", "Sto. Domingo", "Tabaco City",
                  "Ligao City", "Guinobatan", "Camalig", "Tiwi", "Malinao",
                  "Polangui", "Oas", "Libon", "Manito", "Rapu-Rapu"]
CONTRACTORS = [f"Synthetic Builders {c}" for c in "ABCDEFGHIJ"]


def main(out_path: Path) -> None:
    year = RNG.integers(2019, 2025, N)
    length = RNG.uniform(300, 3000, N).round(0)          # meters
    width = RNG.choice([5.0, 6.1, 6.7, 7.3], N)          # typical carriageway widths
    area = (length * width).round(1)
    thickness = RNG.choice([200, 230, 250, 280, 300], N).astype(float)
    surface = RNG.choice(["PCCP", "ASPHALT"], N, p=[0.85, 0.15])
    ancillary = RNG.random(N) < 0.25

    # Fabricated base rate (PHP per m2), scaled by thickness; noise ~ lognormal.
    base_rate = np.where(surface == "PCCP", 2400.0, 1800.0)
    thickness_factor = thickness / 250.0
    infl = 1.0 + 0.035 * (year - 2019)                    # fabricated inflation
    cost = base_rate * area * thickness_factor * infl
    cost *= RNG.lognormal(mean=0.0, sigma=0.10, size=N)
    cost += np.where(ancillary, RNG.uniform(3e6, 15e6, N), 0.0)

    # Inject anomalies: ~7% of rows overpriced 1.5x-2.2x.
    is_anom = RNG.random(N) < 0.07
    cost = np.where(is_anom, cost * RNG.uniform(1.5, 2.2, N), cost)

    abc = cost * RNG.uniform(1.00, 1.12, N)
    duration = (area / RNG.uniform(45, 90, N)).round(0) + 30

    df = pd.DataFrame({
        "ref_no": [f"SYN-{y}{i:04d}" for i, y in enumerate(year)],
        "project_title": [f"[SYNTHETIC] Concreting of Demo Road, {m}"
                          for m in RNG.choice(MUNICIPALITIES, N)],
        "procuring_entity": "SYNTHETIC - DPWH Albay Demo DEO",
        "municipality": RNG.choice(MUNICIPALITIES, N),
        "publish_year": year,
        "abc_php": abc.round(2),
        "contract_amount_php": cost.round(2),
        "awardee": RNG.choice(CONTRACTORS, N),
        "duration_days": duration,
        "surface_type": surface,
        "surface_thickness_mm": thickness,
        "total_paving_area_sqm": area,
        "length_m": length,
        "width_m": width,
        "scope_flag_ancillary": ancillary,
        "_synthetic_injected_anomaly": is_anom,   # ground truth for demo only
    })

    # Simulate real-world missingness for QC-tier demonstration.
    miss = RNG.random(N)
    df.loc[miss < 0.10, "total_paving_area_sqm"] = np.nan   # forces L x W fallback
    df.loc[(miss >= 0.10) & (miss < 0.16), "surface_thickness_mm"] = np.nan

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows -> {out_path}")
    print(f"Injected anomalies: {int(is_anom.sum())}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "data" / "sample" /
         "sample_projects_SYNTHETIC.csv")
