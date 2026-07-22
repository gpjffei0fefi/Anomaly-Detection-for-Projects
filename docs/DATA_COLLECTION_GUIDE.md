# Data Collection Guide

## Part A — PhilGEPS Extraction Protocol

1. Go to `https://notices.philgeps.gov.ph` → **Detailed Search**.
2. Filter: **Category** = Construction Projects (road-related), **Area of Delivery** = Albay, **Publish Date** = 2019-01-01 to 2024-12-31.
3. The portal is an ASP.NET postback application with **no public API** — extraction is manual. Record each project directly into `Albay_DPWH_Road_Projects_Dataset.xlsx`.
4. For each award notice, capture: Reference No. (e.g., 22FA0025), project title, procuring entity, **Approved Budget for the Contract (ABC — mandatory, feeds the abc_ratio anomaly signal)**, contract amount, awardee, award date, contract duration.
5. Retrieve geometry (length, width, surface thickness, Total Paving Area) from the attached Plans/BOQ where available. BOQ Total Paving Area is primary; Length × Width from Plans is fallback.
6. **Road classification (recommended):** note whether the project title / procuring entity indicates a national highway, provincial road, or barangay road. Different classes have different normal unit costs; this can become a stratifying feature.
7. Log every missing field in the data quality log sheet; assign the row a quality tier via `src/dpwh_schema_qc.py`.

**Scope flag (required):** inspect the BOQ/scope of works and record whether the contract includes major non-road items (bridge, slope protection, major drainage, buildings). Without this flag, total-cost models will falsely flag heavy-ancillary contracts as anomalies. **Stretch goal:** if the BOQ is itemized, also estimate `road_scope_share` — the fraction of contract value that is road-surface work — which is the proper fix for scope contamination.

**COA cross-referencing (start early — this is the study's validation):** for every project entered, note whether it appears in a COA Annual Audit Report finding (`coa.gov.ph` → Annual Audit Reports → DPWH, and Region V reports). After the pipeline runs, every HIGH-flagged project must be manually checked against COA findings and credible news coverage. Even 2–3 documented matches converts the study from "the algorithm flagged X projects" into "the algorithm flagged projects with documented audit issues."

---

## Part B — Resources for Material Prices & Equipment Costs

These support feature engineering (e.g., regional price context per project year) and sanity-checking unit costs. Verification status is noted per the honesty policy of this project.

### Confident / well-established sources

1. **DPWH Construction Materials Price Data (CMPD)**
   DPWH publishes region-by-region construction materials price bulletins (cement, reinforcing steel, aggregates, etc.), used internally for cost estimates. Check `www.dpwh.gov.ph` under Business → Construction Materials Price Data, or request from DPWH Region V (Bicol) / Albay District Engineering Offices via FOI. *Availability of every year/region online should be verified — coverage is uneven.*

2. **DPWH Detailed Unit Price Analysis (DUPA) / Approved Budget for the Contract documents**
   DPWH cost estimates are built from DUPAs, which break each pay item (e.g., Item 311 PCCP) into materials, labor, and equipment components. DUPAs for specific projects can be requested through FOI (`www.foi.gov.ph`) and are sometimes included in bid documents on PhilGEPS. This is the most directly relevant source for what a project's cost *should* decompose into.

3. **Philippine Statistics Authority (PSA) — Construction Materials Price Indices**
   - **Construction Materials Wholesale Price Index (CMWPI)** — NCR-based wholesale index, published monthly.
   - **Construction Materials Retail Price Index (CMRPI)** — retail counterpart.
   Available at `psa.gov.ph`. Useful for normalizing costs across the 2019–2024 window (inflation adjustment feature). *Note these indices are NCR-centric; treat as a proxy for Bicol, and state that limitation in the thesis.*

4. **ACEL, Inc. (Association of Carriers and Equipment Lessors) Equipment Rental Rates Guidebook**
   The standard industry reference for heavy-equipment rental rates in the Philippines; DPWH DUPAs conventionally reference ACEL rates for equipment cost components. The guidebook is a paid publication — check whether your school, a local engineering office, or a contractor contact can provide access. *I do not have a verified free online source for current ACEL rates.*

5. **PhilGEPS bid documents themselves**
   Bid Data Sheets, BOQs, and sometimes contractors' detailed estimates attached to notices contain actual unit prices per pay item for your exact projects. This is your highest-fidelity, project-specific price source.

6. **COA Annual Audit Reports (DPWH)**
   `www.coa.gov.ph` → Annual Audit Reports. Useful for cross-validating flagged anomalies and for cost-disallowance context, not for raw prices.

### Needs verification before citing

7. **DPWH Department Orders on Standardized Pay Item Costs / Cost Estimation Guidelines** — DPWH periodically issues DOs governing cost estimation (markup rates, OCM, profit percentages). The specific DO numbers in force for 2019–2024 should be looked up on `www.dpwh.gov.ph` issuances before citing; do not cite from memory.
8. **NEDA / DBM infrastructure cost references** — may exist for benchmark unit costs per km of road; I do not have a verified specific document title, so verify before relying on this.
9. **Regional DTI/DPWH price monitoring bulletins for Region V** — plausible but availability unverified.

### Practical strategy

- Use **PhilGEPS BOQ unit prices** as the primary price signal (project-specific, verifiable).
- Use **PSA CMWPI** to build the year-index deflator: compute annual averages, base year 2019 = 100, and replace the placeholder `YEAR_PRICE_INDEX` in `src/features.py` with the verified values (cite the exact PSA table). State the NCR-proxy limitation in the thesis.
- Capture the **ABC for every project** — the pipeline computes `abc_ratio` from it as an anomaly-stage bid-behavior signal.
- Use **DUPA structure** (materials + labor + equipment + OCM + profit) as the conceptual framework in your methodology chapter, with ACEL as the equipment-rate convention.
- File **FOI requests early** (foi.gov.ph) for DPWH Region V CMPD bulletins and sample DUPAs — turnaround can take 15+ working days.
- Once ~70+ usable rows exist, run `python src/tune_xgboost.py --input data/raw/<file>.csv`, copy the winning hyperparameters into `SMALL_N_PARAMS` in `src/train_xgboost.py`, and record them (plus per-fold RMSE ±) for the methodology chapter.