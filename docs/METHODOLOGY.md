# Methodology & Pipeline

## Pipeline

```
PhilGEPS manual extraction
        │
        ▼
Excel workbook (raw sheet + QC log)
        │
        ▼
src/dpwh_schema_qc.py  →  quality tiers, geometry fallback, RA 9184 ABC check
        │
        ▼
src/features.py  →  price-index deflation, encodings, cost/m², abc_ratio
        │
        ├──► src/train_mlr_baseline.py   (MLR + log-cost MLR, 5-fold CV)
        ├──► src/train_xgboost.py        (XGBoost + native TreeSHAP)
        └──► src/anomaly_iforest.py      (Isolation Forest + LOF + auditor baseline)
                       │
                       ▼
        src/viz_dash_app.py  (interactive website: editable data, live
        parameter sliders, 3D parametric road, SHAP explanations, what-if form)

Standalone: src/tune_xgboost.py (grid search — run once on real data)
```

## Cost-Prediction Models (all reported side by side — no cherry-picking)

**1. MLR baseline** — scikit-learn `LinearRegression`. Included because Antoniou & Aretoulis (2025) found MLR outperforming XGBoost on comparable highway cost data; the thesis reports the comparison honestly either way.

**2. MLR on log(cost)** — construction costs are right-skewed, so a log-target linear model is also fitted. Predictions are back-transformed with exp() and all metrics computed on the original peso scale, so the three models are directly comparable.

**3. XGBoost regressor** — target: deflated contract cost. Features: geometry (m²), surface thickness, surface type (PCCP indicator), duration, publish year, and the ancillary-scope flag. Small-n discipline: shallow trees, strong L2 regularization, 5-fold cross-validation instead of a single holdout. Final hyperparameters come from an exhaustive grid search (`src/tune_xgboost.py`, ~144 combinations × 5 folds) run on the real dataset; the winning parameters and the per-fold RMSE standard deviation (± uncertainty) are reported in the thesis rather than defended as arbitrary choices.

**Explainability (SHAP).** Per-project TreeSHAP contributions are computed via XGBoost's native `pred_contribs` (no external library). For every project, each feature's push on the predicted cost is expressed in pesos; the website surfaces the top drivers in plain language. This grounds the "why was this flagged" answer mathematically instead of heuristically.

**Deliberate exclusion — ABC ratio and target leakage.** The ratio contract amount / ABC is *excluded* from the regression features because its numerator contains the target: including it would let the model trivially "predict" cost and would blind residual-based anomaly detection (an overpriced contract with a matching inflated ABC would look normal). It is used only in the anomaly stage, where it serves as a bid-behavior signal (~1.0 is typical; >1.0 is prohibited under RA 9184).

## Anomaly Detection (three independent lenses)

**1. Isolation Forest (primary)** — unsupervised, run on the standardized model features plus `abc_ratio` plus the *percentage* cost residual (actual − predicted, as % of predicted). The percentage form is used instead of raw pesos so that a 60% overprice on a small barangay road counts as much as on a large project (scale invariance). Flags are reported at two contamination settings — HIGH (default 5%) and MODERATE (default 10%) — as a built-in sensitivity analysis, so conclusions never hinge on one arbitrary threshold.

**2. Local Outlier Factor (second opinion)** — LOF runs independently on the same feature matrix. Projects flagged by *both* Isolation Forest and LOF are reported as the strongest signals in the dataset; method agreement partially compensates for the absence of ground-truth labels.

**3. Auditor's univariate baseline** — the simple spreadsheet rule an auditor could apply without ML: flag any project whose cost per m² exceeds the dataset median + 1.5 standard deviations. Reported alongside the ML flags to answer the value-added question directly: how many projects does ML find that the simple rule misses (typically projects whose cost/m² looks normal but is wrong for their specific size, thickness, or year), and vice versa.

**Known limitation — signal dilution.** Isolation Forest spreads isolation across all dimensions, so a project extreme in only one feature (e.g., +136% residual but otherwise typical) can score lower than intuition suggests. This is a documented characteristic of the algorithm, mitigated here by (a) the LOF second opinion, (b) the univariate baseline, and (c) threshold sensitivity analysis. It is discussed openly in Chapter 5 rather than hidden.

## Evaluation

- **Regression:** MAE, RMSE, R² from 5-fold CV for all three models, plus the per-fold standard deviation of each metric (small-n honesty: report the ± range, not just the mean).
- **Anomaly detection:** no ground-truth labels exist, so quantitative precision/recall is impossible on real data. Validation is therefore (a) method agreement (IF ∩ LOF), (b) comparison against the auditor's baseline, and (c) **manual cross-referencing of HIGH-flagged projects against COA Annual Audit Reports and credible news coverage** — the only path from "the algorithm flagged X projects" to "the algorithm flagged projects with documented issues."
- On the synthetic development dataset only, injected anomalies provide known labels; catch-rate and false-flag counts are shown on the website for demonstration, clearly marked as synthetic.

## 3D Visualization & Interactivity

Plotly + Dash parametric road model: length × width × surface thickness rendered as a 3D solid, color-coded by flag level, with plain-language SHAP-grounded tooltips. The website additionally supports live editing of project values, live retuning of contamination/depth/learning-rate parameters, and a what-if form that prices a hypothetical project and reports whether it would be flagged (including an RA 9184 warning when a proposed cost exceeds the ABC). Design goal: legibility for non-technical citizens, per the study's transparency framing.

## Threats to Validity

1. **Scope contamination** — total cost includes non-road items; mitigated by `scope_flag_ancillary`, but residual noise remains. Full BOQ line-item decomposition is future work.
2. **Small n** — 70–90 rows; mitigated by cross-validation, shallow regularized models, grid-searched parameters, fold-variance reporting, and citation of comparable small-n studies.
3. **No ground truth for anomalies** — flags are statistical signals; validated only via method agreement and COA cross-referencing.
4. **Price index proxy** — PSA CMWPI is NCR-based; Bicol prices may diverge. Stated openly.
5. **Manual extraction error** — mitigated by the QC log and quality tiers.
6. **Isolation Forest dilution** — single-feature extremes can under-score; mitigated by LOF and the univariate baseline (see above).