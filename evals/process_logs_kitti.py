"""Aggregate kitti_results.csv + *_depth.json into a summary table.

Writes docs/eval/kitti/summary_table.md with one row per (Method, Drive)
pair and an overall-average row per method at the bottom.

CSV input columns:
    Method, Drive, ATE_RMSE, ScaleFactor, LC_Count, TUM_Poses, Depth_Frames, Status

Per-run depth JSON at:
    evals/logs/kitti/<method>_<drive>_depth.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHOD_LABEL = {
    "vggt_slam": "VGGT-SLAM",
    "da3_baseline": "DA3 baseline",
    "da3_refine": "DA3 refine",
    "da3_refine_inplace": "DA3 refine (inplace)",
    "da3_refine_none": "DA3 refine (none)",
    "da3_large_refine": "DA3 large-refine",
    "da3_baseline_scale_ransac":    "DA3 baseline [scale=ransac]",
    "da3_baseline_scale_weighted":  "DA3 baseline [scale=weighted]",
    "da3_baseline_chunk32_ov4":     "DA3 baseline [chunk=32 ov=4]",
    "da3_baseline_chunk60_ov8":     "DA3 baseline [chunk=60 ov=8]",
    "da3_baseline_mindisp10":       "DA3 baseline [min_disp=10]",
    "da3_baseline_mindisp25":       "DA3 baseline [min_disp=25]",
    "da3_baseline_giant":           "DA3 baseline (giant)",
    "da3_refine_none_giant":        "DA3 refine [none] (giant)",
    "da3_baseline_nested":          "DA3 baseline (nested)",
    "da3_refine_none_nested":       "DA3 refine [none] (nested)",
    "da3_refine_inplace_nested":    "DA3 refine [inplace] (nested)",
}


def _load_depth_json(method: str, drive: str, log_dir: Path) -> dict | None:
    p = log_dir / f"{method}_{drive}_depth.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    try:
        return f"{float(v):.{nd}f}"
    except (ValueError, TypeError):
        return "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(PROJECT_ROOT / "evals/logs/kitti_results.csv"))
    ap.add_argument("--log_dir", default=str(PROJECT_ROOT / "evals/logs/kitti"))
    ap.add_argument("--output", default=str(PROJECT_ROOT / "docs/eval/kitti/summary_table.md"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    log_dir = Path(args.log_dir)
    out_path = Path(args.output)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    # Keep only most recent row per (Method, Drive)
    df = df.drop_duplicates(subset=["Method", "Drive"], keep="last")

    rows = []
    for _, r in df.iterrows():
        method = r["Method"]
        drive = r["Drive"]
        djson = _load_depth_json(method, drive, log_dir)
        dm = (djson or {}).get("metrics") or {}

        rows.append({
            "Method": METHOD_LABEL.get(method, method),
            "method_key": method,
            "Drive": drive,
            "Keyframes": r.get("TUM_Poses"),
            "ATE_RMSE": r.get("ATE_RMSE"),
            "Scale": r.get("ScaleFactor"),
            "LC": r.get("LC_Count"),
            "AbsRel": dm.get("abs_rel"),
            "RMSE_d": dm.get("rmse"),
            "Delta125": dm.get("delta_1_25"),
            "Status": r.get("Status"),
        })

    # Preserve deterministic method + drive ordering
    method_order = [
        "vggt_slam",
        "da3_baseline",
        "da3_refine",
        "da3_refine_inplace",
        "da3_refine_none",
        "da3_large_refine",
        "da3_baseline_scale_ransac",
        "da3_baseline_scale_weighted",
        "da3_baseline_chunk32_ov4",
        "da3_baseline_chunk60_ov8",
        "da3_baseline_mindisp10",
        "da3_baseline_mindisp25",
        "da3_baseline_giant",
        "da3_refine_none_giant",
        "da3_baseline_nested",
        "da3_refine_none_nested",
        "da3_refine_inplace_nested",
    ]
    rows.sort(key=lambda x: (method_order.index(x["method_key"]) if x["method_key"] in method_order else 999, x["Drive"]))

    # Build markdown (Scale column dropped -- evo_ape without --save_results
    # doesn't print the Sim(3) scale factor, so we never populated it).
    header = (
        "| Method | Drive | Keyframes | ATE_RMSE [m] | LC | AbsRel | RMSE_d [m] | δ<1.25 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"| {r['Method']} | {r['Drive']} | "
            f"{_fmt(r['Keyframes'], nd=0)} | "
            f"{_fmt(r['ATE_RMSE'])} | "
            f"{_fmt(r['LC'], nd=0)} | "
            f"{_fmt(r['AbsRel'])} | {_fmt(r['RMSE_d'])} | {_fmt(r['Delta125'])} |"
        )

    # Per-method averages
    summary_lines = ["", "### Averages", "", header.rstrip("\n")]
    for m in method_order:
        grp = [r for r in rows if r["method_key"] == m]
        if not grp:
            continue
        def _avg(k):
            vals = [r[k] for r in grp if r[k] is not None and not (isinstance(r[k], float) and not np.isfinite(r[k]))]
            vals = [float(v) for v in vals if str(v).strip() not in ("", "nan")]
            return np.mean(vals) if vals else None
        summary_lines.append(
            f"| **{METHOD_LABEL.get(m, m)} (avg)** | -- | "
            f"{_fmt(_avg('Keyframes'), nd=1)} | "
            f"{_fmt(_avg('ATE_RMSE'))} | "
            f"{_fmt(_avg('LC'), nd=1)} | "
            f"{_fmt(_avg('AbsRel'))} | {_fmt(_avg('RMSE_d'))} | {_fmt(_avg('Delta125'))} |"
        )

    md = "# KITTI Validation-Set Evaluation\n\n"
    md += "Per-run metrics (scroll below for averages).\n\n"
    md += header + "\n".join(body_lines) + "\n"
    md += "\n".join(summary_lines) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote summary → {out_path}")

    # Stdout brief
    print("\n=== Per-method averages ===")
    for m in method_order:
        grp = [r for r in rows if r["method_key"] == m]
        if not grp:
            continue
        ate = [float(r["ATE_RMSE"]) for r in grp if r["ATE_RMSE"] not in (None, "") and str(r["ATE_RMSE"]).strip() != "nan"]
        abs_rel = [float(r["AbsRel"]) for r in grp if r["AbsRel"] is not None and np.isfinite(float(r["AbsRel"] or np.nan))]
        ate_mean = np.mean(ate) if ate else float("nan")
        abs_mean = np.mean(abs_rel) if abs_rel else float("nan")
        print(f"  {METHOD_LABEL.get(m, m):<18}  ATE_RMSE={ate_mean:.4f}  AbsRel={abs_mean:.4f}  (n={len(grp)})")


if __name__ == "__main__":
    main()
