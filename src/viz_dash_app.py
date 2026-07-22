"""
viz_dash_app.py  (interactive version)
Public transparency website + interactive model lab for Albay DPWH road projects.

What's interactive:
  1. EDIT THE DATA  - key columns in the project table are editable; press
                      "Recompute" and the whole pipeline (QC -> MLR -> XGBoost
                      -> Isolation Forest) retrains live on your edited values.
  2. TUNE THE MODEL - sliders for Isolation Forest contamination (HIGH and
                      MODERATE thresholds) and XGBoost depth / learning rate.
  3. WHAT-IF FORM   - type in a hypothetical project (area, thickness, cost...)
                      and see the predicted fair cost and whether it would be
                      flagged.
  4. PLAIN-LANGUAGE - every flagged project gets a "why" sentence; model
                      accuracy (MAE / RMSE / R2) is shown up front, and on
                      synthetic data a recall check against injected anomalies.

Run (no pipeline pre-run needed - the app computes everything itself):
  python src/generate_synthetic_data.py          # if you have no real CSV yet
  python src/viz_dash_app.py                     # -> http://127.0.0.1:8050
  python src/viz_dash_app.py --input data/raw/albay_projects.csv   # real data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dpwh_schema_qc import ProjectRecord, REQUIRED_COLUMNS          # noqa: E402
from features import (build_features, FEATURE_COLUMNS, TARGET,      # noqa: E402
                      ANOMALY_EXTRA_COLUMNS)
from train_mlr_baseline import cross_validate_mlr, cross_validate_mlr_log  # noqa: E402
from train_xgboost import cross_validate_xgb, shap_contributions    # noqa: E402
from anomaly_iforest import score_anomalies                         # noqa: E402

# ---- design tokens: "asphalt & road paint" -------------------------------
INK = "#22252A"
PAPER = "#F5F4F0"
CARD = "#FFFFFF"
LINE_YELLOW = "#E8B10C"
FLAG_RED = "#C4402F"
FLAG_AMBER = "#D98E04"
OK_GRAY = "#8A8F98"
FONT = "'Arial', 'Helvetica Neue', sans-serif"
FLAG_COLORS = {"HIGH": FLAG_RED, "MODERATE": FLAG_AMBER, "NONE": OK_GRAY}

EDITABLE_COLS = {"contract_amount_php", "total_paving_area_sqm",
                 "surface_thickness_mm", "publish_year", "duration_days",
                 "surface_type", "scope_flag_ancillary"}

# Module-level cache of fitted objects for the what-if form.
# Fine for a local, single-user thesis app; not for multi-user deployment.
CACHE: dict = {}


# ==========================================================================
# Pipeline (runs live inside callbacks)
# ==========================================================================
def qc_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    from dataclasses import asdict
    records = [ProjectRecord.from_row(row) for _, row in raw_df.iterrows()]
    out = pd.DataFrame([asdict(r) for r in records])
    out["qc_notes"] = out["qc_notes"].apply("; ".join)
    return out


def run_full_pipeline(raw_df: pd.DataFrame, cont_high: float, cont_mod: float,
                      max_depth: int, learning_rate: float):
    """raw rows -> (results_df, metrics_dict). Raises ValueError on bad data."""
    qc = qc_dataframe(raw_df)
    usable = qc[qc["quality_tier"] != "UNUSABLE"]
    dropped = len(qc) - len(usable)
    if len(usable) < 15:
        raise ValueError(f"Only {len(usable)} usable rows after QC - too few to model.")

    feat = build_features(usable)
    mlr = cross_validate_mlr(feat)
    mlr_log = cross_validate_mlr_log(feat)
    xgb = cross_validate_xgb(feat, params={"max_depth": int(max_depth),
                                           "learning_rate": float(learning_rate)})
    results = score_anomalies(feat, xgb["oof_predictions"],
                              contamination_levels=(cont_high, cont_mod))

    # SHAP: per-project drivers of the model's expected cost (native TreeSHAP).
    shap = shap_contributions(xgb["model"],
                              results[FEATURE_COLUMNS].to_numpy(dtype=float))
    for ci, cname in enumerate(FEATURE_COLUMNS):
        results[f"shap_{cname}"] = shap[:, ci]

    CACHE["xgb_model"] = xgb["model"]
    CACHE["iso_model"] = results.attrs["iso_model"]
    CACHE["scaler"] = results.attrs["scaler"]
    CACHE["anomaly_cols"] = results.attrs["anomaly_cols"]
    CACHE["median_abc_ratio"] = float(results["abc_ratio"].median())
    CACHE["results"] = results
    # Score thresholds equivalent to the table's HIGH/MODERATE flags, so the
    # what-if form gives verdicts consistent with the project table.
    scores = results["anomaly_score"].to_numpy()
    CACHE["thr_high"] = float(np.quantile(scores, 1.0 - min(cont_high, cont_mod)))
    CACHE["thr_mod"] = float(np.quantile(scores, 1.0 - max(cont_high, cont_mod)))

    ml_flagged = (results["flag_level"] != "NONE")
    metrics = {
        "n": len(results), "dropped": dropped,
        "flags_high": int((results["flag_level"] == "HIGH").sum()),
        "flags_moderate": int((results["flag_level"] == "MODERATE").sum()),
        "mlr": {k: mlr[k] for k in ("cv_mae_mean", "cv_rmse_mean", "cv_r2_mean")},
        "mlr_log": {k: mlr_log[k] for k in ("cv_mae_mean", "cv_rmse_mean",
                                            "cv_r2_mean")},
        "xgb": {k: xgb[k] for k in ("cv_mae_mean", "cv_rmse_mean", "cv_r2_mean")},
        "both_methods": int(results["flagged_by_both"].sum()),
        "baseline_flags": int(results["baseline_flag"].sum()),
        "baseline_overlap": int((results["baseline_flag"] & ml_flagged).sum()),
        "ml_only": int((ml_flagged & ~results["baseline_flag"]).sum()),
    }
    # Demo-only recall check: synthetic data carries its own ground truth.
    if "_synthetic_injected_anomaly" in raw_df.columns:
        truth = raw_df.set_index("ref_no")["_synthetic_injected_anomaly"]
        truth = truth.reindex(results["ref_no"]).fillna(False).astype(bool).to_numpy()
        flagged = (results["flag_level"] != "NONE").to_numpy()
        tp = int((truth & flagged).sum())
        metrics["synthetic_check"] = {
            "injected": int(truth.sum()), "caught": tp,
            "false_flags": int((~truth & flagged).sum()),
        }
    return results, metrics


SHAP_LABELS = {"geometry_sqm": "its paving area",
               "surface_thickness_mm": "its surface thickness",
               "duration_days": "its contract duration",
               "publish_year": "its project year",
               "scope_flag_ancillary": "its ancillary-works scope",
               "surface_is_pccp": "its surface type"}


def explain_row(row: pd.Series, df: pd.DataFrame) -> str:
    """Plain-language 'why was this flagged' sentence, grounded in SHAP."""
    parts = []
    rp = row["residual_pct"]
    if rp >= 15:
        parts.append(f"costs {rp:.0f}% more than the model expects for a road "
                     f"of this size, thickness, and year")
    elif rp <= -15:
        parts.append(f"costs {abs(rp):.0f}% less than expected, which is also "
                     f"unusual")
    cps = row["cost_per_sqm_php"]
    med = df["cost_per_sqm_php"].median()
    if med > 0 and cps > 1.5 * med:
        parts.append(f"its cost per square meter (PHP {cps:,.0f}) is over 1.5x "
                     f"the dataset median (PHP {med:,.0f})")
    if row["scope_flag_ancillary"]:
        parts.append("note: the contract includes major non-road works, which "
                     "can legitimately raise the total")
    if not parts:
        parts.append("its combination of size, duration, and cost is rare in "
                     "the dataset, even though no single value is extreme")
    txt = "This project " + "; ".join(parts) + "."

    # SHAP: what drove the model's expectation for this specific project.
    shap_vals = {c: row.get(f"shap_{c}") for c in FEATURE_COLUMNS
                 if pd.notna(row.get(f"shap_{c}"))}
    if shap_vals:
        top = max(shap_vals, key=lambda c: abs(shap_vals[c]))
        v = shap_vals[top]
        txt += (f" The model's expected cost was pushed "
                f"{'up' if v > 0 else 'down'} mostly by {SHAP_LABELS[top]} "
                f"(PHP {abs(v)/1e6:.1f}M effect, SHAP).")
    if bool(row.get("lof_flag")):
        txt += " A second method (LOF) independently agrees this is an outlier."
    return txt


# ==========================================================================
# Figures
# ==========================================================================
def road_mesh(length_m, width_m, thickness_mm, color):
    L, W, T = float(length_m), float(width_m), float(thickness_mm) / 1000.0
    x = [0, L, L, 0, 0, L, L, 0]
    y = [0, 0, W, W, 0, 0, W, W]
    z = [0, 0, 0, 0, T, T, T, T]
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    fig = go.Figure(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                              color=color, opacity=0.95, flatshading=True))
    fig.add_trace(go.Scatter3d(
        x=[0, L], y=[W / 2, W / 2], z=[T * 1.02, T * 1.02],
        mode="lines", line=dict(color=LINE_YELLOW, width=8),
        hoverinfo="skip", showlegend=False))
    fig.update_layout(
        scene=dict(xaxis_title="Length (m)", yaxis_title="Width (m)",
                   zaxis_title="Thickness (m)", aspectmode="manual",
                   aspectratio=dict(x=3, y=0.9, z=0.5)),
        margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=CARD, height=360)
    return fig


def cost_scatter(df):
    fig = go.Figure()
    for level in ["NONE", "MODERATE", "HIGH"]:
        sub = df[df["flag_level"] == level]
        fig.add_trace(go.Scatter(
            x=sub["predicted_cost_php"] / 1e6, y=sub["cost_deflated_php"] / 1e6,
            mode="markers",
            name={"NONE": "Within pattern", "MODERATE": "Moderate flag",
                  "HIGH": "High flag"}[level],
            marker=dict(color=FLAG_COLORS[level], size=9 if level != "NONE" else 7,
                        line=dict(color=INK, width=0.5)),
            customdata=sub[["ref_no", "municipality"]],
            hovertemplate="<b>%{customdata[0]}</b> - %{customdata[1]}<br>"
                          "Expected: PHP %{x:.1f}M<br>Actual: PHP %{y:.1f}M"
                          "<extra></extra>"))
    lim = max(df["cost_deflated_php"].max(), df["predicted_cost_php"].max()) / 1e6
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                             line=dict(color=OK_GRAY, dash="dash", width=1),
                             name="Cost matches prediction", hoverinfo="skip"))
    fig.update_layout(xaxis_title="Expected cost (PHP millions)",
                      yaxis_title="Actual cost, deflated (PHP millions)",
                      paper_bgcolor=CARD, plot_bgcolor=CARD,
                      font=dict(family=FONT, color=INK),
                      legend=dict(orientation="h", y=1.1), height=360,
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ==========================================================================
# Layout helpers
# ==========================================================================
def kpi(label, value, accent=INK):
    return html.Div(style={"background": CARD, "padding": "12px 16px",
                           "borderRadius": "6px", "borderTop": f"4px solid {accent}",
                           "flex": "1", "minWidth": "140px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"},
                    children=[
        html.Div(label, style={"fontSize": "11px", "color": OK_GRAY,
                               "textTransform": "uppercase",
                               "letterSpacing": "0.06em"}),
        html.Div(value, style={"fontSize": "24px", "fontWeight": "700"})])


def card(children, **style):
    base = {"background": CARD, "padding": "18px", "borderRadius": "6px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.08)", "marginTop": "18px"}
    base.update(style)
    return html.Div(children, style=base)


def metrics_card_children(metrics):
    rows = [("Multiple Linear Regression", metrics["mlr"]),
            ("MLR (log cost)", metrics["mlr_log"]),
            ("XGBoost", metrics["xgb"])]
    best_name = min(rows, key=lambda r: r[1]["cv_rmse_mean"])[0]
    best = dict(rows)[best_name]
    table = html.Table(style={"borderCollapse": "collapse", "fontSize": "14px"},
                       children=[
        html.Tr([html.Th(h, style={"textAlign": "left",
                                   "padding": "6px 22px 6px 0",
                                   "borderBottom": f"2px solid {INK}"})
                 for h in ["Model", "MAE (PHP)", "RMSE (PHP)", "R2"]]),
        *[html.Tr([html.Td(n, style={"padding": "6px 22px 6px 0",
                                     "fontWeight": "700" if n == best_name
                                     else "400"}),
                   html.Td(f"{mm['cv_mae_mean']:,.0f}",
                           style={"padding": "6px 22px 6px 0"}),
                   html.Td(f"{mm['cv_rmse_mean']:,.0f}",
                           style={"padding": "6px 22px 6px 0"}),
                   html.Td(f"{mm['cv_r2_mean']:.3f}",
                           style={"padding": "6px 22px 6px 0"})])
          for n, mm in rows]])
    kids = [
        html.H3("How accurate is the model? (5-fold cross-validation)",
                style={"marginTop": 0}),
        table,
        html.P([
            "In plain terms: the best model (", html.B(best_name), ") misses "
            f"the true cost by about PHP {best['cv_mae_mean']/1e6:.1f} million "
            f"per project on average (MAE) and explains "
            f"{best['cv_r2_mean']*100:.0f}% of cost variation between projects "
            f"(R2). Anomaly flags use the XGBoost prediction gap; all models "
            f"are always reported side by side."],
            style={"fontSize": "13px", "color": "#4A4E55"}),
        html.P([
            html.B("Does ML add value over a simple audit rule? "),
            f"A spreadsheet rule (cost/m2 above median + 1.5 SD) flags "
            f"{metrics['baseline_flags']} projects. The ML pipeline flags "
            f"{metrics['flags_high'] + metrics['flags_moderate']}, of which "
            f"{metrics['baseline_overlap']} overlap with the simple rule and "
            f"{metrics['ml_only']} are found ONLY by ML - typically projects "
            f"whose cost/m2 looks normal but is wrong for their specific size, "
            f"thickness, or year. {metrics['both_methods']} projects are "
            f"flagged by two independent ML methods (Isolation Forest + LOF), "
            f"the strongest signals in the dataset."],
            style={"fontSize": "13px", "color": "#4A4E55"}),
    ]
    sc = metrics.get("synthetic_check")
    if sc:
        kids.append(html.P(
            f"Synthetic-data check: {sc['caught']} of {sc['injected']} "
            f"deliberately overpriced demo projects were flagged, with "
            f"{sc['false_flags']} flags on clean projects. Detection is "
            f"imperfect by design honesty - flags are screening signals, "
            f"not verdicts. Raise/lower the contamination sliders to trade "
            f"catches against false flags.",
            style={"fontSize": "13px", "background": "#FFF6DC",
                   "border": f"1px solid {LINE_YELLOW}", "borderRadius": "6px",
                   "padding": "8px 12px"}))
    return kids


# ==========================================================================
# App
# ==========================================================================
def load_raw(input_path: Path | None) -> pd.DataFrame:
    candidates = ([input_path] if input_path else []) + [
        ROOT / "data" / "raw" / "albay_projects.csv",
        ROOT / "data" / "sample" / "sample_projects_SYNTHETIC.csv",
    ]
    for p in candidates:
        if p and Path(p).exists():
            print(f"Loading {p}")
            return pd.read_csv(p)
    raise SystemExit("No input CSV found. Run: python src/generate_synthetic_data.py")


ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, default=None)
ARGS, _ = ap.parse_known_args()
RAW = load_raw(ARGS.input)
IS_SYNTHETIC = RAW["ref_no"].astype(str).str.startswith("SYN-").any()

DEFAULTS = dict(cont_high=0.05, cont_mod=0.10, max_depth=3, learning_rate=0.05)
RESULTS0, METRICS0 = run_full_pipeline(RAW, **DEFAULTS)

TABLE_COLS = [
    {"name": "Ref No.", "id": "ref_no", "editable": False},
    {"name": "Municipality", "id": "municipality", "editable": False},
    {"name": "Year", "id": "publish_year", "editable": True, "type": "numeric"},
    {"name": "Surface", "id": "surface_type", "editable": True},
    {"name": "Thickness (mm)", "id": "surface_thickness_mm", "editable": True,
     "type": "numeric"},
    {"name": "Paving area (m2)", "id": "total_paving_area_sqm", "editable": True,
     "type": "numeric"},
    {"name": "Duration (days)", "id": "duration_days", "editable": True,
     "type": "numeric"},
    {"name": "Ancillary works", "id": "scope_flag_ancillary", "editable": True},
    {"name": "Cost (PHP)", "id": "contract_amount_php", "editable": True,
     "type": "numeric"},
    {"name": "Expected (PHP)", "id": "predicted_cost_php", "editable": False},
    {"name": "Over/Under %", "id": "residual_pct", "editable": False},
    {"name": "Flag", "id": "flag_level", "editable": False},
    {"name": "LOF agrees", "id": "lof_flag", "editable": False},
]


def results_to_table(results: pd.DataFrame, raw: pd.DataFrame) -> list[dict]:
    """Merge model outputs back onto raw rows for the editable table."""
    keep_raw = raw.set_index("ref_no")
    t = results[["ref_no", "municipality", "publish_year", "surface_type",
                 "surface_thickness_mm", "total_paving_area_sqm",
                 "duration_days", "scope_flag_ancillary",
                 "contract_amount_php", "predicted_cost_php", "residual_pct",
                 "flag_level", "lof_flag", "baseline_flag", "abc_ratio",
                 "length_m", "width_m", "geometry_sqm",
                 "abc_php", "project_title", "procuring_entity", "awardee",
                 "cost_per_sqm_php"]
                + [f"shap_{c}" for c in FEATURE_COLUMNS]].copy()
    t["lof_flag"] = t["lof_flag"].map({True: "YES", False: ""})
    t["predicted_cost_php"] = t["predicted_cost_php"].round(0)
    t["residual_pct"] = t["residual_pct"].round(1)
    t["scope_flag_ancillary"] = t["scope_flag_ancillary"].map(
        {1: "TRUE", 0: "FALSE", True: "TRUE", False: "FALSE"})
    t = t.sort_values("residual_pct", ascending=False)
    if "_synthetic_injected_anomaly" in keep_raw.columns:
        t["_synthetic_injected_anomaly"] = t["ref_no"].map(
            keep_raw["_synthetic_injected_anomaly"]).astype(str)
    return t.to_dict("records")


app = Dash(__name__, title="Albay Road Watch")
server = app.server

slider_label = {"fontSize": "12px", "color": "#4A4E55", "marginTop": "8px"}

app.layout = html.Div(style={"background": PAPER, "minHeight": "100vh",
                             "fontFamily": FONT, "color": INK,
                             "padding": "0 0 40px 0"}, children=[
    html.Div(style={"height": "8px", "background":
                    f"repeating-linear-gradient(45deg, {INK}, {INK} 16px, "
                    f"{LINE_YELLOW} 16px, {LINE_YELLOW} 32px)"}),
    html.Div(style={"maxWidth": "1180px", "margin": "0 auto",
                    "padding": "26px 20px 0"}, children=[

        html.H1("Albay Road Watch",
                style={"margin": 0, "fontSize": "34px", "letterSpacing": "-0.5px"}),
        html.P("Every road is a promise. Compare what DPWH road projects in "
               "Albay actually cost against what similar roads should cost - "
               "and experiment: edit any value, tune the model, or test a "
               "hypothetical project.",
               style={"maxWidth": "780px", "color": "#4A4E55", "marginTop": "6px"}),
        html.Div(style={"background": "#FFF6DC", "border": f"1px solid {LINE_YELLOW}",
                        "borderRadius": "6px", "padding": "10px 14px",
                        "fontSize": "13px", "marginTop": "8px"}, children=[
            html.B("Read this first: "),
            "a flag means a project's cost deviates from the statistical "
            "pattern of comparable projects. It is not an accusation - "
            "terrain, right-of-way, drainage, and other legitimate factors "
            "can explain higher costs."]),
        html.Div(style={"background": "#FDEBE7", "border": f"1px solid {FLAG_RED}",
                        "borderRadius": "6px", "padding": "8px 14px",
                        "fontSize": "13px", "marginTop": "8px",
                        "display": "block" if IS_SYNTHETIC else "none"},
                 children=[html.B("Demo mode: "),
                           "all projects shown are SYNTHETIC (fabricated) "
                           "sample data. No real contracts are displayed."]),

        html.Div(id="kpi-row", style={"display": "flex", "gap": "14px",
                                      "marginTop": "18px", "flexWrap": "wrap"}),

        # ---- controls ----------------------------------------------------
        card([
            html.H3("Model controls", style={"marginTop": 0}),
            html.Div(style={"display": "flex", "gap": "30px", "flexWrap": "wrap"},
                     children=[
                html.Div(style={"flex": 1, "minWidth": "240px"}, children=[
                    html.Div("HIGH flag threshold (contamination) - what share "
                             "of projects the forest treats as clear outliers",
                             style=slider_label),
                    dcc.Slider(0.02, 0.15, 0.01, value=DEFAULTS["cont_high"],
                               id="cont-high", marks={0.02: "2%", 0.05: "5%",
                                                      0.10: "10%", 0.15: "15%"}),
                    html.Div("MODERATE flag threshold", style=slider_label),
                    dcc.Slider(0.05, 0.25, 0.01, value=DEFAULTS["cont_mod"],
                               id="cont-mod", marks={0.05: "5%", 0.10: "10%",
                                                     0.20: "20%", 0.25: "25%"}),
                ]),
                html.Div(style={"flex": 1, "minWidth": "240px"}, children=[
                    html.Div("XGBoost tree depth - deeper trees fit harder but "
                             "risk memorizing a small dataset", style=slider_label),
                    dcc.Slider(2, 6, 1, value=DEFAULTS["max_depth"],
                               id="max-depth"),
                    html.Div("XGBoost learning rate", style=slider_label),
                    dcc.Slider(0.01, 0.30, 0.01, value=DEFAULTS["learning_rate"],
                               id="learning-rate",
                               marks={0.01: "0.01", 0.05: "0.05", 0.10: "0.10",
                                      0.30: "0.30"}),
                ]),
            ]),
            html.Button("Recompute models", id="recompute-btn", n_clicks=0,
                        style={"marginTop": "12px", "background": INK,
                               "color": PAPER, "border": "none",
                               "padding": "10px 22px", "borderRadius": "6px",
                               "fontSize": "14px", "fontWeight": "600",
                               "cursor": "pointer"}),
            html.Span(id="recompute-status",
                      style={"marginLeft": "12px", "fontSize": "13px",
                             "color": OK_GRAY}),
        ]),

        html.Div(id="metrics-card-wrap"),

        # ---- editable table ---------------------------------------------
        card([
            html.H3("Projects - edit any white cell, then press Recompute",
                    style={"marginTop": 0}),
            html.P("Editable: year, surface (PCCP/ASPHALT), thickness, area, "
                   "duration, ancillary works (TRUE/FALSE), and cost. Try "
                   "doubling a project's cost and watch it get flagged.",
                   style={"fontSize": "13px", "color": "#4A4E55"}),
            dash_table.DataTable(
                id="project-table",
                columns=TABLE_COLS,
                data=results_to_table(RESULTS0, RAW),
                editable=True, page_size=10, sort_action="native",
                row_selectable="single", selected_rows=[0],
                style_cell={"fontFamily": FONT, "fontSize": "13px",
                            "padding": "7px", "textAlign": "left",
                            "maxWidth": "180px", "overflow": "hidden",
                            "textOverflow": "ellipsis"},
                style_header={"background": INK, "color": PAPER,
                              "fontWeight": "600"},
                style_data_conditional=[
                    {"if": {"column_editable": False},
                     "backgroundColor": "#F0EFEA"},
                    {"if": {"filter_query": "{flag_level} = 'HIGH'",
                            "column_id": "flag_level"},
                     "backgroundColor": FLAG_RED, "color": "white"},
                    {"if": {"filter_query": "{flag_level} = 'MODERATE'",
                            "column_id": "flag_level"},
                     "backgroundColor": FLAG_AMBER, "color": "white"},
                ]),
        ]),

        # ---- detail row --------------------------------------------------
        html.Div(style={"display": "flex", "gap": "18px", "flexWrap": "wrap"},
                 children=[
            card([html.H3(id="road-title", style={"marginTop": 0}),
                  html.Div(id="road-explain",
                           style={"fontSize": "13px", "color": "#4A4E55",
                                  "marginBottom": "6px"}),
                  dcc.Graph(id="road-3d", config={"displayModeBar": False})],
                 flex="1", minWidth="440px"),
            card([html.H3("Actual vs expected cost", style={"marginTop": 0}),
                  dcc.Graph(id="scatter", config={"displayModeBar": False})],
                 flex="1", minWidth="440px"),
        ]),

        # ---- what-if form ------------------------------------------------
        card([
            html.H3("Test a hypothetical project", style={"marginTop": 0}),
            html.P("Enter a road's specs and a proposed cost. The trained "
                   "model predicts a fair cost and checks whether the "
                   "proposal would be flagged.",
                   style={"fontSize": "13px", "color": "#4A4E55"}),
            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                            "alignItems": "flex-end"}, children=[
                *[html.Div([html.Div(lbl, style=slider_label),
                            dcc.Input(id=iid, type="number", value=val,
                                      style={"width": "130px", "padding": "6px"})])
                  for lbl, iid, val in [
                      ("Paving area (m2)", "wi-area", 9000),
                      ("Thickness (mm)", "wi-thick", 280),
                      ("Duration (days)", "wi-days", 180),
                      ("Year", "wi-year", 2024),
                      ("Proposed cost (PHP)", "wi-cost", 30_000_000),
                      ("ABC (PHP, optional)", "wi-abc", None)]],
                html.Div([html.Div("Surface", style=slider_label),
                          dcc.Dropdown(["PCCP", "ASPHALT"], "PCCP", id="wi-surface",
                                       clearable=False,
                                       style={"width": "130px"})]),
                html.Div([html.Div("Ancillary works?", style=slider_label),
                          dcc.Dropdown(["No", "Yes"], "No", id="wi-anc",
                                       clearable=False,
                                       style={"width": "130px"})]),
                html.Button("Check this project", id="wi-btn", n_clicks=0,
                            style={"background": LINE_YELLOW, "color": INK,
                                   "border": "none", "padding": "10px 18px",
                                   "borderRadius": "6px", "fontWeight": "700",
                                   "cursor": "pointer"}),
            ]),
            html.Div(id="wi-result", style={"marginTop": "14px"}),
        ]),

        html.P("Data: PhilGEPS public procurement notices. A thesis project of "
               "Legazpi City Science High School. This site does not accuse "
               "any contractor or official of wrongdoing.",
               style={"fontSize": "12px", "color": OK_GRAY, "marginTop": "26px"}),
    ]),
])


# ==========================================================================
# Callbacks
# ==========================================================================
@app.callback(
    Output("project-table", "data"),
    Output("kpi-row", "children"),
    Output("metrics-card-wrap", "children"),
    Output("scatter", "figure"),
    Output("recompute-status", "children"),
    Input("recompute-btn", "n_clicks"),
    State("project-table", "data"),
    State("cont-high", "value"), State("cont-mod", "value"),
    State("max-depth", "value"), State("learning-rate", "value"),
)
def recompute(n_clicks, table_data, cont_high, cont_mod, max_depth, lr):
    if n_clicks == 0:
        results, metrics = RESULTS0, METRICS0
        raw = RAW
        status = ""
    else:
        # Rebuild a raw-style frame from the (possibly edited) table.
        raw = pd.DataFrame(table_data)
        for col in REQUIRED_COLUMNS:
            if col not in raw.columns:
                raw[col] = np.nan
        try:
            results, metrics = run_full_pipeline(
                raw, min(cont_high, cont_mod), max(cont_high, cont_mod),
                max_depth, lr)
            status = f"Recomputed on {metrics['n']} rows."
        except (ValueError, Exception) as e:  # surface data errors, don't crash
            return (no_update, no_update, no_update, no_update,
                    f"Error: {e}")

    kpis = [
        kpi("Projects analyzed", f"{metrics['n']}"),
        kpi("High flags", f"{metrics['flags_high']}", FLAG_RED),
        kpi("Moderate flags", f"{metrics['flags_moderate']}", FLAG_AMBER),
        kpi("Best model R2",
            f"{max(metrics['mlr']['cv_r2_mean'], metrics['xgb']['cv_r2_mean']):.2f}",
            LINE_YELLOW),
    ]
    return (results_to_table(results, raw), kpis,
            card(metrics_card_children(metrics)), cost_scatter(results), status)


@app.callback(
    Output("road-3d", "figure"),
    Output("road-title", "children"),
    Output("road-explain", "children"),
    Input("project-table", "selected_rows"),
    Input("project-table", "data"),
)
def update_road(selected_rows, table_data):
    if not table_data:
        return no_update, no_update, no_update
    idx = (selected_rows or [0])[0]
    idx = min(idx, len(table_data) - 1)
    row = pd.Series(table_data[idx])
    results = CACHE.get("results", RESULTS0)
    full = results[results["ref_no"] == row["ref_no"]]
    src = full.iloc[0] if len(full) else row

    L = src.get("length_m")
    W = src.get("width_m")
    if not (pd.notna(L) and L and pd.notna(W) and W):
        W = 6.1
        L = float(src.get("geometry_sqm", 6100)) / W
    T = float(src.get("surface_thickness_mm") or 250)

    flag = row.get("flag_level", "NONE")
    color = "#6E7178" if flag == "NONE" else FLAG_COLORS[flag]
    fig = road_mesh(L, W, T, color)
    explain = explain_row(src, results) if flag != "NONE" else (
        "This project sits within the normal cost pattern for its size, "
        "thickness, and year.")
    subtitle = (f"{src.get('municipality','')} - {int(src.get('publish_year',0))} - "
                f"{src.get('surface_type','')} {T:.0f} mm - "
                f"{L:,.0f} m x {W:.1f} m. {explain}")
    return fig, f"3D road view - {row['ref_no']} (flag: {flag})", subtitle


@app.callback(
    Output("wi-result", "children"),
    Input("wi-btn", "n_clicks"),
    State("wi-area", "value"), State("wi-thick", "value"),
    State("wi-days", "value"), State("wi-year", "value"),
    State("wi-cost", "value"), State("wi-abc", "value"),
    State("wi-surface", "value"), State("wi-anc", "value"),
)
def what_if(n_clicks, area, thick, days, year, cost, abc, surface, anc):
    if not n_clicks:
        return ""
    try:
        from features import YEAR_PRICE_INDEX
        from train_xgboost import shap_contributions
        if None in (area, thick, days, year, cost):
            return html.Div("Please fill in every field (ABC is optional).",
                            style={"color": FLAG_RED})
        if int(year) not in YEAR_PRICE_INDEX:
            return html.Div(f"Year must be one of {sorted(YEAR_PRICE_INDEX)} "
                            f"(price index coverage).", style={"color": FLAG_RED})

        xgb = CACHE["xgb_model"]
        iso = CACHE["iso_model"]
        scaler = CACHE["scaler"]

        feat = {"geometry_sqm": float(area),
                "surface_thickness_mm": float(thick),
                "duration_days": float(days),
                "publish_year": int(year),
                "scope_flag_ancillary": 1 if anc == "Yes" else 0,
                "surface_is_pccp": 1 if surface == "PCCP" else 0}
        Xrow = np.array([[feat[c] for c in FEATURE_COLUMNS]], dtype=float)
        predicted = float(xgb.predict(Xrow)[0])

        deflated = float(cost) * 100.0 / YEAR_PRICE_INDEX[int(year)]
        residual_pct = 100.0 * (deflated - predicted) / predicted
        abc_ratio = (float(cost) / float(abc) if abc and float(abc) > 0
                     else CACHE["median_abc_ratio"])
        feat_full = {**feat, "abc_ratio": abc_ratio,
                     "residual_pct": residual_pct}
        Xiso = scaler.transform(np.array(
            [[feat_full[c] for c in CACHE["anomaly_cols"]]], dtype=float))
        score = float(-iso.score_samples(Xiso)[0])
        if score >= CACHE["thr_high"]:
            color = FLAG_RED
            verdict = "WOULD BE FLAGGED (HIGH) - a clear outlier vs the dataset"
        elif score >= CACHE["thr_mod"]:
            color = FLAG_AMBER
            verdict = "WOULD BE FLAGGED (MODERATE) - unusual enough to review"
        else:
            color = "#2E7D32"
            verdict = "would NOT be flagged - within the normal pattern"

        # SHAP: which inputs drove the predicted fair cost.
        contribs = shap_contributions(xgb, Xrow)[0]
        pairs = sorted(zip(FEATURE_COLUMNS, contribs[:-1]),
                       key=lambda p: -abs(p[1]))[:2]
        drivers = "; ".join(
            f"{SHAP_LABELS[c]} pushed it "
            f"{'up' if v > 0 else 'down'} PHP {abs(v)/1e6:.1f}M"
            for c, v in pairs)

        if abc and abc_ratio > 1.001:
            verdict += (". NOTE: proposed cost exceeds the ABC, which is "
                        "prohibited under RA 9184")

        return html.Div(style={"border": f"2px solid {color}",
                               "borderRadius": "6px", "padding": "12px 16px"},
                        children=[
            html.Div([html.B("Predicted fair cost: "),
                      f"PHP {predicted:,.0f} (in base-year pesos). "
                      f"Your proposal, deflated: PHP {deflated:,.0f} "
                      f"({residual_pct:+.0f}% vs prediction)."]),
            html.Div([html.B("What set the prediction (SHAP): "),
                      drivers + "."],
                     style={"fontSize": "13px", "marginTop": "4px"}),
            html.Div([html.B("Result: "), verdict,
                      f" (anomaly score {score:.2f})."],
                     style={"color": color, "marginTop": "6px"}),
            html.Div("Reminder: a flag is a statistical signal for closer "
                     "review, not proof of anything.",
                     style={"fontSize": "12px", "color": OK_GRAY,
                            "marginTop": "6px"}),
        ])
    except Exception as e:
        return html.Div(f"Error: {e}", style={"color": FLAG_RED})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)