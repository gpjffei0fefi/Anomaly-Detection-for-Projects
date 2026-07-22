# Data Dictionary — `Albay_DPWH_Road_Projects_Dataset.xlsx`

**Grain:** one row = one whole DPWH road project (contract). Not per pavement layer.

| Column | Type | Unit | Source | Notes |
|---|---|---|---|---|
| `ref_no` | str | — | PhilGEPS | Procurement reference, e.g., `22FA0025`, `24FB0046` |
| `project_title` | str | — | PhilGEPS | As published |
| `procuring_entity` | str | — | PhilGEPS | e.g., DPWH Albay 2nd DEO |
| `municipality` | str | — | PhilGEPS / title | Location within Albay |
| `publish_year` | int | year | PhilGEPS | 2019–2024 window |
| `abc_php` | float | ₱ | PhilGEPS | Approved Budget for the Contract. **Capture this for every row** — it feeds `abc_ratio`, a bid-behavior signal in the anomaly stage |
| `contract_amount_php` | float | ₱ | Award notice | Awarded contract amount |
| `awardee` | str | — | Award notice | Winning contractor |
| `duration_days` | int | days | Award notice | Contract duration |
| `surface_type` | cat | — | BOQ/Plans | `PCCP` / `Asphalt` / other |
| `surface_thickness_mm` | float | mm | BOQ/Plans | Wearing/surface course only (e.g., 280 for 22FA0025, 24FB0046) |
| `total_paving_area_sqm` | float | m² | BOQ | **Primary geometry feature** |
| `length_m` | float | m | Plans | Fallback geometry |
| `width_m` | float | m | Plans | Fallback geometry |
| `geometry_source` | cat | — | QC script | `BOQ_AREA` / `PLANS_LXW` / `MISSING` |
| `scope_flag_ancillary` | bool | — | BOQ scope review | **True if contract includes major non-road scope** (bridge, slope protection, major drainage). Required to prevent false anomaly flags |
| `cost_per_sqm_php` | float | ₱/m² | derived | `contract_amount_php / total_paving_area_sqm` |
| `price_index_year` | float | index | PSA CMWPI | Deflator feature (base year TBD) |
| `quality_tier` | cat | — | `dpwh_schema_qc.py` | `COMPLETE` / `USABLE_PARTIAL` / `USABLE_SPARSE` / `UNUSABLE` |
| `qc_notes` | str | — | manual | Missing fields, assumptions, provenance |

## Derived columns (produced by the pipeline — do not enter manually)

| Column | Produced by | Meaning |
|---|---|---|
| `abc_ratio` | `features.py` | `contract_amount_php / abc_php`. Anomaly-stage feature ONLY — excluded from the cost regression to avoid target leakage. ~1.0 typical; >1.0 prohibited under RA 9184 |
| `cost_deflated_php` | `features.py` | Contract cost deflated to base-year pesos via `YEAR_PRICE_INDEX` (replace placeholder with PSA CMWPI before final runs) |
| `predicted_cost_php` | pipeline | XGBoost out-of-fold expected cost |
| `residual_pct` | pipeline | (actual − predicted) / predicted × 100. Scale-invariant anomaly signal |
| `anomaly_score` | `anomaly_iforest.py` | Isolation Forest score; higher = more anomalous |
| `flag_level` | `anomaly_iforest.py` | `HIGH` / `MODERATE` / `NONE` at the two contamination thresholds |
| `lof_flag` | `anomaly_iforest.py` | Local Outlier Factor second opinion (independent method) |
| `flagged_by_both` | `anomaly_iforest.py` | IF ∩ LOF — strongest signals in the dataset |
| `baseline_flag` | `anomaly_iforest.py` | Auditor's rule: cost/m² > median + 1.5 SD |
| `shap_<feature>` | `train_xgboost.py` | Per-project TreeSHAP contribution of each feature, in pesos |

## Recommended future columns (add if extractable)

| Column | Source | Why |
|---|---|---|
| `road_classification` | Project title / procuring entity | National / provincial / barangay roads have different normal unit costs; stratifies the model |
| `road_scope_share` | BOQ line items | Fraction of contract value that is road-surface work; the proper fix for scope contamination (stretch goal) |

**Derived model targets/features are defined in `src/features.py`; this file is the source of truth for raw columns.**