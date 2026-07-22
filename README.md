# AI-Driven Parametric 3D Visualization using XGBoost and Isolation Forest for Detecting Anomalies and Transparency Analysis in DPWH Road Projects of Albay Province

A public transparency tool that uses machine learning to detect cost anomalies in Department of Public Works and Highways (DPWH) road construction projects across Albay Province, Philippines, paired with parametric 3D visualization of road geometry.

> **Status:** Research in progress — Senior High School thesis, Mathematical and Computational Science strand, Legazpi City Science High School. Target completion: July 2026.

---

## 1. Motivation

Road construction is the single largest budget allocation within the DPWH, yet road projects are among the least evaluated for physical compliance. Commission on Audit (COA) reports have repeatedly flagged defective, unfinished, and idle infrastructure projects nationwide. This study builds a data-driven layer of accountability: it predicts what a road project *should* cost given its geometry and context, flags projects that deviate abnormally, and renders the road parametrically in 3D so that findings are legible to ordinary citizens — not just engineers and auditors.

**Alignment:** SDG 16.6 (accountable institutions), SDG 9.1 (resilient infrastructure), NIBRA 2022–2028 (SAKLAW cluster, Risk Communication for DRR).

## 2. Methodology Overview

| Component | Role |
|---|---|
| **XGBoost** (Chen & Guestrin, 2016) | Predicts expected project cost from geometric and contextual features |
| **Isolation Forest** (Liu, Ting & Zhou, 2008) | Flags statistical outliers (anomalous projects), unsupervised |
| **Multiple Linear Regression** | Baseline comparator (note: Antoniou & Aretoulis, 2025, found MLR can outperform XGBoost on small samples — acknowledged in the thesis) |
| **Plotly + Dash** | Parametric 3D visualization of road geometry (length, width, surface thickness) |

**Research design:** applied, quantitative, developmental, cross-sectional.

## 3. Dataset

- **Source:** PhilGEPS (`notices.philgeps.gov.ph`) — manual extraction via the Detailed Search form (Category + Area of Delivery = "Albay"). No public API exists.
- **Scope:** ~110 candidate DPWH road projects in Albay Province, publish dates 2019–2024; expected 70–90 usable rows after quality screening.
- **Grain:** one row per whole project (not per pavement layer). Surface/wearing course thickness only.
- **Primary geometry feature:** Total Paving Area from the Bill of Quantities (BOQ); Length × Width from Plans as fallback.
- **Quality tiers:** `COMPLETE` / `USABLE_PARTIAL` / `USABLE_SPARSE` / `UNUSABLE` (see `src/dpwh_schema_qc.py`).
- **Known limitation:** Total contract cost includes all scope (earthworks, bridges, drainage), not just road surface. A scope-flag feature is required to avoid false anomaly flags on contracts with heavy ancillary works. See `docs/DATA_DICTIONARY.md`.

Small-sample justification: published ML literature includes comparable highway cost-estimation studies built on as few as 19 projects.

## 4. Repository Structure

```
dpwh-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/                  # PhilGEPS extractions (NOT committed — see .gitignore)
│   ├── processed/            # QC-passed dataset exports
│   └── Albay_DPWH_Road_Projects_Dataset.xlsx  (local only)
├── docs/
│   ├── DATA_DICTIONARY.md    # Column definitions, units, provenance
│   ├── DATA_COLLECTION_GUIDE.md  # PhilGEPS extraction protocol + price/equipment resources
│   └── METHODOLOGY.md        # Pipeline, models, metrics, evaluation plan
├── src/
│   ├── dpwh_schema_qc.py     # ProjectRecord dataclass, geometry fallback, quality tiers
│   ├── features.py           # Feature engineering (incl. scope flag)
│   ├── train_xgboost.py      # Cost prediction model
│   ├── train_mlr_baseline.py # MLR comparator
│   ├── anomaly_iforest.py    # Isolation Forest scoring
│   └── viz_dash_app.py       # Plotly Dash 3D parametric viewer
└── notebooks/
    └── exploratory/          # EDA, sanity checks
```

## 5. Setup

```bash
git clone <your-repo-url>
cd dpwh-anomaly-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the QC pipeline on a raw extraction:

```bash
python src/dpwh_schema_qc.py --input data/raw/philgeps_export.csv --output data/processed/
```

Launch the 3D visualization app:

```bash
python src/viz_dash_app.py
# open http://127.0.0.1:8050
```

## 6. Evaluation

- **Regression:** MAE, RMSE, R² (scikit-learn), XGBoost vs. MLR baseline
- **Anomaly detection:** Isolation Forest anomaly scores; flagged projects cross-checked against COA audit findings and news reports where available
- **Validation caveat:** ground-truth "anomalous" labels do not exist; anomaly flags indicate *statistical* deviation, not proof of irregularity. All outputs are framed as screening signals for further human review.

## 7. Data Ethics & Disclaimer

All data used is publicly available procurement information published by the Government of the Philippines through PhilGEPS and DPWH. This tool does **not** accuse any contractor or official of wrongdoing. An anomaly flag means only that a project's cost deviates from the statistical pattern of comparable projects; legitimate explanations (terrain, right-of-way costs, ancillary structures) are common. Findings are intended to support transparency and citizen oversight, consistent with the Freedom of Information program (EO No. 2, s. 2016) and the Government Procurement Reform Act (RA 9184) / New Government Procurement Act (RA 12009).

## 8. Key References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16.*
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM '08.*
- DPWH (2013). *Standard Specifications for Highways, Bridges and Airports* ("Blue Book"), Items 105, 200, 201, 202.
- Full APA 7th master source list maintained separately in the thesis manuscript.

## 9. Author

Enrique — Legazpi City Science High School, Mathematical and Computational Science strand.
