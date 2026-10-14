"""
build_site.py
Builds the static sample website (site/index.html) from pipeline output.

  python src/run_pipeline.py --input data/sample/sample_projects_SYNTHETIC.csv
  python site/build_site.py --raw data/sample/sample_projects_SYNTHETIC.csv

Reads data/processed/results.csv + metrics.json, embeds them as JSON into
site/template.html, and writes site/index.html (one self-contained file,
no server needed). If the raw CSV has the `_synthetic_injected_anomaly`
column, the page also reports how many injected anomalies were caught.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PROCESSED = ROOT / "data" / "processed"


def explain(row: pd.Series, median_cps: float) -> str:
    """Plain-language reason, same rules as viz_dash_app.explain_row (no SHAP)."""
    parts = []
    rp = row["residual_pct"]
    if rp >= 15:
        parts.append(f"costs {rp:.0f}% more than the model expects for a road "
                     f"of this size, thickness, and year")
    elif rp <= -15:
        parts.append(f"costs {abs(rp):.0f}% less than expected, which is also unusual")
    cps = row["cost_per_sqm_php"]
    if median_cps > 0 and cps > 1.5 * median_cps:
        parts.append(f"its cost per square meter (PHP {cps:,.0f}) is over 1.5x "
                     f"the dataset median (PHP {median_cps:,.0f})")
    if row["scope_flag_ancillary"]:
        parts.append("note: the contract includes major non-road works, which "
                     "can legitimately raise the total")
    if not parts:
        parts.append("its combination of size, duration, and cost is rare in "
                     "the dataset, even though no single value is extreme")
    txt = "This project " + "; ".join(parts) + "."
    if bool(row.get("lof_flag")) and row["flag_level"] != "NONE":
        txt += " A second method (LOF) independently agrees this is an outlier."
    return txt


def num(v, nd=0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), nd)


def main(raw_csv: Path | None) -> None:
    res = pd.read_csv(PROCESSED / "results.csv")
    metrics = json.loads((PROCESSED / "metrics.json").read_text())
    med_cps = float(res["cost_per_sqm_php"].median())

    truth = None
    if raw_csv and raw_csv.exists():
        raw = pd.read_csv(raw_csv)
        if "_synthetic_injected_anomaly" in raw.columns:
            truth = raw.set_index("ref_no")["_synthetic_injected_anomaly"].astype(bool)

    projects = []
    for _, r in res.iterrows():
        width = num(r["width_m"], 1)
        length = num(r["length_m"], 0)
        if length is None and width:
            length = round(r["geometry_sqm"] / width)
        projects.append({
            "ref": r["ref_no"],
            "title": r["project_title"].replace("[SYNTHETIC] ", ""),
            "muni": r["municipality"],
            "year": int(r["publish_year"]),
            "awardee": r["awardee"],
            "surface": r["surface_type"],
            "thick": num(r["surface_thickness_mm"]),
            "length": length,
            "width": width,
            "area": num(r["geometry_sqm"]),
            "geomSrc": r["geometry_source"],
            "tier": r["quality_tier"],
            "days": num(r["duration_days"]),
            "contract": num(r["contract_amount_php"]),
            "actual": num(r["cost_deflated_php"]),
            "predicted": num(r["predicted_cost_php"]),
            "residPct": num(r["residual_pct"], 1),
            "cps": num(r["cost_per_sqm_php"]),
            "score": num(r["anomaly_score"], 3),
            "flag": r["flag_level"],
            "lof": bool(r["lof_flag"]),
            "baseline": bool(r["baseline_flag"]),
            "ancillary": bool(r["scope_flag_ancillary"]),
            "injected": bool(truth.get(r["ref_no"], False)) if truth is not None else None,
            "why": explain(r, med_cps),
        })

    if truth is not None:
        flagged = res["flag_level"] != "NONE"
        t = res["ref_no"].map(truth).fillna(False).astype(bool)
        metrics["synthetic_check"] = {
            "injected": int(truth.sum()),
            "caught": int((t & flagged).sum()),
            "false_flags": int((~t & flagged).sum()),
        }
    metrics["median_cps"] = med_cps
    metrics["n_lof"] = int(res["lof_flag"].sum())
    metrics["n_baseline"] = int(res["baseline_flag"].sum())

    payload = json.dumps({"projects": projects, "metrics": metrics},
                         separators=(",", ":"))
    html = (SITE / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {SITE / 'index.html'} ({len(projects)} projects)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None,
                    help="Raw input CSV (optional; enables the synthetic check)")
    main(ap.parse_args().raw)
