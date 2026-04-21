"""Diagnose pass-1 instability with large chunk_size on KITTI.

Observation: DA3-SLAM with chunk_size=40 (large-refine preset used by the
KITTI eval) produces a 47 m pass-1 ATE on drive_0013, vs 2.57 m at
chunk_size=17 (baseline preset). Pass-2 reopt is NOT the cause.
This script sweeps chunk_size over one drive and reports:

  - per-chunk ATE RMSE (pass 1 only)
  - submap count, keyframe count
  - per-submap pass-1 scale factors (median inter-submap scale)
  - per-submap predicted camera-distance statistics
  - per-submap z-depth statistics
  - trajectory top-down viz saved to docs/eval/kitti/_chunk_sweep/<drive>/

Usage:
  XFORMERS_DISABLED=1 python evals/diagnose_chunk_size.py \\
      --drive 2011_09_26_drive_0013_sync \\
      --chunk_sizes 17,24,32,40,60,80 \\
      --min_disparity 15
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import time
import contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evals.kitti_drives import image_folder_for, KITTI_ROOT
from evals.diagnose_refine_regression import (
    umeyama_alignment,
    ate_rmse,
    load_gt_positions,
    pair_and_ate,
    extract_positions,
)


SCALE_FACTOR_RE = re.compile(r"scale factor \(\w+\) \(([-0-9.eE+]+),")


def run_pass1(drive, chunk_size, overlap, model_name, min_disparity, model=None):
    """Run DA3-SLAM pass 1 with explicit chunk_size / overlap.

    Captures stdout to extract the printed scale factors from solver.add_edge.
    Returns dict with ATE, submap stats, and captured scales.
    """
    from da3_slam.solver import Solver
    from da3_slam.da3_wrapper import DA3Wrapper
    import da3_slam.slam_utils as utils

    img_folder = image_folder_for(drive, KITTI_ROOT)
    image_names = [f for f in glob.glob(os.path.join(img_folder, "*"))
                   if "depth" not in os.path.basename(f).lower()
                   and "txt" not in os.path.basename(f).lower()
                   and "db" not in os.path.basename(f).lower()]
    image_names = utils.sort_images_by_number(image_names)

    submap_size = chunk_size - overlap

    solver = Solver(init_conf_threshold=25.0, lc_thres=0.95,
                    scale_method="median", overlap=overlap)
    if model is None:
        model = DA3Wrapper(model_name=model_name, device="cuda")

    image_names_subset = []
    image_count = 0

    stdout_capture = io.StringIO()
    t0 = time.time()

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                if hasattr(s, "flush"):
                    s.flush()

    # Tee stdout so we see progress AND capture scale lines
    with contextlib.redirect_stdout(Tee(sys.__stdout__, stdout_capture)):
        for image_name in image_names:
            img = cv2.imread(image_name)
            enough = solver.flow_tracker.compute_disparity(img, min_disparity, False)
            if enough:
                image_names_subset.append(image_name)
                image_count += 1
            if len(image_names_subset) == submap_size + overlap or image_name == image_names[-1]:
                if len(image_names_subset) < 2:
                    image_names_subset = []
                    continue
                preds = solver.run_predictions(image_names_subset, model, 1)
                solver.add_points(preds)
                solver.graph.optimize()
                image_names_subset = image_names_subset[-overlap:]

    captured = stdout_capture.getvalue()
    scale_factors = [float(m) for m in SCALE_FACTOR_RE.findall(captured)]

    pass1_time = time.time() - t0

    # ---- per-submap stats ----
    submap_info = []
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        pc = submap.pointclouds  # (N, H, W, 3)
        z = pc[..., 2].reshape(-1)
        z_good = z[np.isfinite(z) & (z > 0)]
        if z_good.size == 0:
            z_med = z_p10 = z_p90 = 0.0
        else:
            z_med = float(np.median(z_good))
            z_p10 = float(np.percentile(z_good, 10))
            z_p90 = float(np.percentile(z_good, 90))

        poses = submap.poses  # (N, 4, 4) w2c
        c2w = np.linalg.inv(poses)
        trans = c2w[:, :3, 3]
        dists = np.linalg.norm(trans, axis=1)
        dist_med = float(np.median(dists))
        dist_max = float(np.max(dists))

        submap_info.append(dict(
            id=int(submap.get_id()),
            n_frames=int(pc.shape[0]),
            z_median=z_med, z_p10=z_p10, z_p90=z_p90,
            cam_dist_median=dist_med, cam_dist_max=dist_max,
        ))

    # ---- ATE vs GT ----
    gt_by_idx = load_gt_positions(str(PROJECT_ROOT / "evals" / "logs" / "kitti_gt" / f"{drive}.tum"))
    est_positions, kidx_list = extract_positions(solver, drive)
    rmse, est_aligned, gt_mat = pair_and_ate(est_positions, kidx_list, gt_by_idx,
                                              f"chunk={chunk_size}")

    return dict(
        chunk_size=chunk_size,
        overlap=overlap,
        submap_size=submap_size,
        n_keyframes=image_count,
        n_submaps=len(submap_info),
        pass1_time_s=pass1_time,
        ate_rmse_m=rmse,
        scale_factors=scale_factors,
        submaps=submap_info,
        traj_est=est_aligned.tolist() if est_aligned is not None else None,
        traj_gt=gt_mat.tolist() if gt_mat is not None else None,
    )


def save_viz(results, drive, out_dir):
    """Save top-down trajectory overlay for all chunk sizes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    gt = None
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(results)))
    for (res, c) in zip(results, colors):
        if res.get("traj_est") is None:
            continue
        traj = np.array(res["traj_est"])
        ax.plot(traj[:, 0], traj[:, 2],
                color=c, linewidth=1.3, alpha=0.85,
                label=f"chunk={res['chunk_size']} ({res['ate_rmse_m']:.2f}m)")
        if gt is None and res.get("traj_gt") is not None:
            gt = np.array(res["traj_gt"])
    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 2], color="black", linewidth=2.0,
                label="GT", alpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"{drive} -- pass-1 trajectory vs chunk_size")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    outfile = out_dir / "trajectory_vs_chunk.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"[viz] trajectory sweep → {outfile}")

    # ATE curve
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    xs = [r["chunk_size"] for r in results]
    ys = [r["ate_rmse_m"] for r in results]
    ax.plot(xs, ys, marker="o", color="tab:red")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("chunk_size"); ax.set_ylabel("Pass-1 ATE RMSE [m]")
    ax.set_title(f"{drive} -- pass-1 ATE vs chunk_size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ate_vs_chunk.png", dpi=120)
    plt.close(fig)
    print(f"[viz] ate curve → {out_dir / 'ate_vs_chunk.png'}")

    # Cam-distance boxplot per submap, per chunk
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    positions = []
    data = []
    labels = []
    for i, res in enumerate(results):
        d_meds = [s["cam_dist_median"] for s in res["submaps"]]
        if not d_meds:
            continue
        data.append(d_meds)
        positions.append(i)
        labels.append(f"chunk={res['chunk_size']}\nn={res['n_submaps']}")
    if data:
        ax.boxplot(data, positions=positions, labels=labels, widths=0.7)
    ax.set_ylabel("per-submap median cam distance (DA3 local)")
    ax.set_title(f"{drive} -- DA3 per-submap cam-distance medians vs chunk_size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cam_dist_boxplot.png", dpi=120)
    plt.close(fig)
    print(f"[viz] cam-dist boxplot → {out_dir / 'cam_dist_boxplot.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default="2011_09_26_drive_0013_sync")
    ap.add_argument("--chunk_sizes", default="17,24,32,40,60,80")
    ap.add_argument("--overlap", type=int, default=1,
                    help="keyframe overlap between chunks (1 = baseline default)")
    ap.add_argument("--model_name", default="da3-small")
    ap.add_argument("--min_disparity", type=int, default=15)
    args = ap.parse_args()

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    print(f"Sweeping chunk_sizes={chunk_sizes} on {args.drive}")

    out_dir = PROJECT_ROOT / "docs" / "eval" / "kitti" / "_chunk_sweep" / args.drive
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model once and reuse (much faster than reloading per chunk size)
    from da3_slam.da3_wrapper import DA3Wrapper
    model = DA3Wrapper(model_name=args.model_name, device="cuda")

    results = []
    for cs in chunk_sizes:
        print("\n" + "=" * 72)
        print(f"CHUNK SIZE = {cs}  (overlap={args.overlap})")
        print("=" * 72)
        # Large-refine uses overlap=15 for chunk=80 in chunk_strategy.py, but
        # for a fair sweep we keep overlap=1 (baseline) unless overridden.
        res = run_pass1(args.drive, cs, args.overlap, args.model_name,
                        args.min_disparity, model=model)
        results.append(res)
        print(f"  chunk={cs}: n_sub={res['n_submaps']} n_kf={res['n_keyframes']} "
              f"ATE={res['ate_rmse_m']:.3f}m time={res['pass1_time_s']:.1f}s")
        print(f"  scale factors ({len(res['scale_factors'])}): "
              f"{[f'{s:.3f}' for s in res['scale_factors']]}")
        dist_meds = [f"{s['cam_dist_median']:.3f}" for s in res['submaps'][:6]]
        print(f"  per-submap cam dist median (first 6): {dist_meds}")

    # Persist results
    json_out = out_dir / "chunk_sweep_results.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[out] json → {json_out}")

    save_viz(results, args.drive, out_dir)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'chunk':>6} {'sub':>4} {'kf':>4} {'ATE[m]':>8} {'scale_std':>10}")
    for res in results:
        s = res["scale_factors"]
        sstd = float(np.std(s)) if s else 0.0
        print(f"{res['chunk_size']:>6d} {res['n_submaps']:>4d} "
              f"{res['n_keyframes']:>4d} {res['ate_rmse_m']:>8.3f} {sstd:>10.4f}")


if __name__ == "__main__":
    main()
