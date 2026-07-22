"""
dpwh_schema_qc.py
Schema validation and quality control for the Albay DPWH road projects dataset.

Grain: one row = one whole project (contract).
Geometry policy: BOQ Total Paving Area is primary; Length x Width from Plans is fallback.

Quality tiers:
  COMPLETE       - cost + geometry (BOQ area) + thickness + surface type present
  USABLE_PARTIAL - cost + geometry (any source) present, thickness or surface type missing
  USABLE_SPARSE  - cost present, geometry recoverable only via fallback with assumptions
  UNUSABLE       - missing cost or no geometry recoverable

Usage:
  python src/dpwh_schema_qc.py --input data/raw/projects.csv --output data/processed/
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "ref_no", "project_title", "procuring_entity", "municipality",
    "publish_year", "abc_php", "contract_amount_php", "awardee",
    "duration_days", "surface_type", "surface_thickness_mm",
    "total_paving_area_sqm", "length_m", "width_m", "scope_flag_ancillary",
]

VALID_SURFACE_TYPES = {"PCCP", "ASPHALT", "GRAVEL", "OTHER"}


def _num(value) -> Optional[float]:
    """Coerce to float; treat blanks/NaN/non-numeric as None."""
    if value is None:
        return None
    try:
        f = float(str(value).replace(",", "").replace("\u20b1", "").strip())
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


@dataclass
class ProjectRecord:
    ref_no: str
    project_title: str = ""
    procuring_entity: str = ""
    municipality: str = ""
    publish_year: Optional[int] = None
    abc_php: Optional[float] = None
    contract_amount_php: Optional[float] = None
    awardee: str = ""
    duration_days: Optional[float] = None
    surface_type: str = ""
    surface_thickness_mm: Optional[float] = None
    total_paving_area_sqm: Optional[float] = None
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    scope_flag_ancillary: bool = False

    # Derived by QC
    geometry_sqm: Optional[float] = None
    geometry_source: str = "MISSING"
    quality_tier: str = "UNUSABLE"
    qc_notes: list = field(default_factory=list)

    # ---- geometry fallback -------------------------------------------------
    def resolve_geometry(self) -> None:
        if self.total_paving_area_sqm and self.total_paving_area_sqm > 0:
            self.geometry_sqm = self.total_paving_area_sqm
            self.geometry_source = "BOQ_AREA"
        elif (self.length_m and self.width_m
              and self.length_m > 0 and self.width_m > 0):
            self.geometry_sqm = self.length_m * self.width_m
            self.geometry_source = "PLANS_LXW"
            self.qc_notes.append("Geometry from Plans L x W fallback, not BOQ.")
        else:
            self.geometry_sqm = None
            self.geometry_source = "MISSING"
            self.qc_notes.append("No recoverable geometry.")

    # ---- validation checks -------------------------------------------------
    def run_checks(self) -> None:
        if self.contract_amount_php is None or self.contract_amount_php <= 0:
            self.qc_notes.append("Missing/invalid contract amount.")
        if (self.abc_php and self.contract_amount_php
                and self.contract_amount_php > self.abc_php * 1.001):
            self.qc_notes.append(
                "Contract amount exceeds ABC - verify (normally disallowed under RA 9184).")
        st = (self.surface_type or "").strip().upper()
        if st and st not in VALID_SURFACE_TYPES:
            self.qc_notes.append(f"Unrecognized surface type '{self.surface_type}'.")
        self.surface_type = st
        if self.surface_thickness_mm is not None and not (50 <= self.surface_thickness_mm <= 400):
            self.qc_notes.append(
                f"Surface thickness {self.surface_thickness_mm} mm outside plausible 50-400 mm range.")
        if self.publish_year is not None and not (2019 <= int(self.publish_year) <= 2024):
            self.qc_notes.append("Publish year outside 2019-2024 study window.")

    # ---- tier assignment ---------------------------------------------------
    def assign_tier(self) -> None:
        has_cost = self.contract_amount_php is not None and self.contract_amount_php > 0
        if not has_cost or self.geometry_sqm is None:
            self.quality_tier = "UNUSABLE"
            return
        has_thickness = self.surface_thickness_mm is not None
        has_surface = bool(self.surface_type)
        if self.geometry_source == "BOQ_AREA" and has_thickness and has_surface:
            self.quality_tier = "COMPLETE"
        elif has_thickness or has_surface:
            self.quality_tier = "USABLE_PARTIAL"
        else:
            self.quality_tier = "USABLE_SPARSE"

    @classmethod
    def from_row(cls, row: pd.Series) -> "ProjectRecord":
        rec = cls(
            ref_no=str(row.get("ref_no", "")).strip(),
            project_title=str(row.get("project_title", "")).strip(),
            procuring_entity=str(row.get("procuring_entity", "")).strip(),
            municipality=str(row.get("municipality", "")).strip(),
            publish_year=int(_num(row.get("publish_year")) or 0) or None,
            abc_php=_num(row.get("abc_php")),
            contract_amount_php=_num(row.get("contract_amount_php")),
            awardee=str(row.get("awardee", "")).strip(),
            duration_days=_num(row.get("duration_days")),
            surface_type=str(row.get("surface_type", "")).strip(),
            surface_thickness_mm=_num(row.get("surface_thickness_mm")),
            total_paving_area_sqm=_num(row.get("total_paving_area_sqm")),
            length_m=_num(row.get("length_m")),
            width_m=_num(row.get("width_m")),
            scope_flag_ancillary=str(row.get("scope_flag_ancillary", "")).strip().upper()
            in {"TRUE", "1", "YES", "Y"},
        )
        rec.resolve_geometry()
        rec.run_checks()
        rec.assign_tier()
        return rec


def run_qc(input_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")

    records = [ProjectRecord.from_row(row) for _, row in df.iterrows()]
    out = pd.DataFrame([asdict(r) for r in records])
    out["qc_notes"] = out["qc_notes"].apply("; ".join)

    output_dir.mkdir(parents=True, exist_ok=True)
    usable = out[out["quality_tier"] != "UNUSABLE"].copy()
    usable.to_csv(output_dir / "projects_qc_passed.csv", index=False)
    out.to_csv(output_dir / "projects_qc_full.csv", index=False)

    print("QC summary:")
    print(out["quality_tier"].value_counts().to_string())
    print(f"\nUsable rows: {len(usable)} / {len(out)}")
    print(f"Wrote {output_dir/'projects_qc_passed.csv'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QC the DPWH projects dataset.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    run_qc(args.input, args.output)
