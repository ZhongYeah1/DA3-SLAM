"""Diagnose why DA3 refine regresses vs baseline on KITTI.

Hypothesis: pass-2 `forward_with_poses` produces depth in a per-submap
normalized scale that differs from pass-1's scale. Since
`run_refinement_pass` only updates `submap.pointclouds` (not `proj_mats`
/ `poses`), the subsequent `reoptimize_after_refinement` mixes pass-2
points with pass-1 proj_mats in the scale-factor estimator, corrupting
inter-submap alignment.

This diagnostic runs on one short drive and captures:
  1. Pass-1 submap pointcloud z stats (before pass 2).
  2. Pass-2 submap pointcloud z stats (after pass 2, before reopt).
  3. Pass-1 scale factors (from add_edge's green print, captured).
  4. Pass-2 scale factors (from reoptimize's green print, captured).
  5. ATE vs GT at 3 stages:
       A. after pass-1 optimization
       B. after pass 2 but before reoptimize  (graph unchanged → same as A
          in theory, but we dump it to be sure)
       C. after reoptimize
  6. Also reports whether pass-2 depths differ from pass-1 depths
     meaningfully (to confirm the "identical depth metrics" observation
     is a collection bug, not a no-op refinement).

Usage:
  cd /nfs/turbo/coe-jungaocv/siyuanb/classes/project/DA3-SLAM
  XFORMERS_DISABLED=1 python evals/diagnose_refine_regression.py \\
      --drive 2011_09_29_drive_0026_sync
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
import cv2

from evals.kitti_drives import image_folder_for, KITTI_ROOT


# -------------------------------------------------------------------------
# Umeyama Sim(3) alignment + ATE RMSE (minimal evo-style)
# -------------------------------------------------------------------------
def umeyama_alignment(est: np.ndarray, gt: np.ndarray, with_scale: bool = True):
    """Fit Sim(3) transform (R, t, s) mapping est → gt by Umeyama's method.

    est, gt: (N, 3) arrays of 3-D points (camera centers).
    Returns (R, t, s).
    """
    n = est.shape[0]
    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_c = est - mu_est
    gt_c = gt - mu_gt
    cov = (gt_c.T @ est_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_est = (est_c ** 2).sum() / n
        s = np.trace(np.diag(D) @ S) / var_est if var_est > 0 else 1.0
    else:
        s = 1.0
    t = mu_gt - s * R @ mu_est
    return R, t, s


def ate_rmse(est_positions: np.ndarray, gt_positions: np.ndarray) -> tuple:
    """Compute ATE RMSE after Sim(3) alignment. Both arrays must be (N, 3)
    and pre-matched index-by-index. Returns (rmse_m, scale_factor).
    """
    assert est_positions.shape == gt_positions.shape, \
        f"Shape mismatch: est {est_positions.shape} vs gt {gt_positions.shape}"
    R, t, s = umeyama_alignment(est_positions, gt_positions, with_scale=True)
    est_aligned = (s * (R @ est_positions.T)).T + t
    errors = np.linalg.norm(est_aligned - gt_positions, axis=1)
    return float(np.sqrt((errors ** 2).mean())), float(s)


# -------------------------------------------------------------------------
# Trajectory extraction helpers
# -------------------------------------------------------------------------
def extract_positions(solver, drive: str) -> tuple:
    """Return (est_positions, kitti_indices) from the current solver state.

    est_positions: (M, 3) camera centers in world frame (c2w translation).
    kitti_indices: list of M KITTI frame indices, parsed via submap.get_frame_ids().
    """
    positions = []
    kidx_list = []
    seen = set()
    for sm in solver.map.ordered_submaps_by_key():
        if sm.get_lc_status():
            continue
        poses = sm.get_all_poses_world(solver.graph)  # (N, 4, 4) c2w
        frame_ids = sm.get_frame_ids()
        for fid, p in zip(frame_ids, poses):
            kidx = int(fid)
            if kidx in seen:
                continue
            seen.add(kidx)
            positions.append(p[:3, 3])
            kidx_list.append(kidx)
    return np.array(positions), kidx_list


def load_gt_positions(gt_tum_path: str) -> dict:
    """Load GT TUM → {kitti_idx: (x, y, z)} using timestamp_mode=frame_index
    (ts = 0.1 * kitti_idx, so kitti_idx = round(ts * 10)).
    """
    gt_by_idx = {}
    with open(gt_tum_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            kidx = int(round(ts * 10))
            gt_by_idx[kidx] = np.array([x, y, z])
    return gt_by_idx


def pair_and_ate(est_positions: np.ndarray, kidx_list: list, gt_by_idx: dict, tag: str) -> tuple:
    """Match est positions to GT by kitti index, compute ATE, print.
    Returns (rmse, est_aligned (M,3), gt (M,3)) for viz.
    """
    matched_est = []
    matched_gt = []
    for p, kidx in zip(est_positions, kidx_list):
        if kidx in gt_by_idx:
            matched_est.append(p)
            matched_gt.append(gt_by_idx[kidx])
    if len(matched_est) < 3:
        print(f"  [{tag}] only {len(matched_est)} matches -- skipping ATE")
        return float("nan"), None, None
    est = np.array(matched_est)
    gt = np.array(matched_gt)
    rmse, s = ate_rmse(est, gt)
    R, t, sc = umeyama_alignment(est, gt, with_scale=True)
    est_aligned = (sc * (R @ est.T)).T + t
    print(f"  [{tag}] n={len(est):4d}  ATE_RMSE = {rmse:7.4f} m   (Sim3 scale = {sc:.4f})")
    return rmse, est_aligned, gt


# -------------------------------------------------------------------------
# Visualization helpers (required per docs/memory -- save viz for every KITTI run)
# -------------------------------------------------------------------------
def save_trajectory_plot(stages: list, out_path: str, drive: str):
    """Overlay trajectories from multiple stages on one top-down XZ plot.

    stages: list of (tag, est_aligned (M,3), gt (M,3)). Only first stage's GT is drawn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    drawn_gt = False
    colors = {"STAGE A": "#1f77b4", "STAGE B": "#2ca02c", "STAGE C": "#d62728"}
    for tag, est, gt in stages:
        if est is None or gt is None:
            continue
        if not drawn_gt:
            ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2.5, label="GT", alpha=0.8)
            ax.scatter(gt[0, 0], gt[0, 2], marker="o", c="k", s=60, zorder=5, label="GT start")
            drawn_gt = True
        ax.plot(est[:, 0], est[:, 2], "-", color=colors.get(tag, "gray"), lw=1.5,
                label=f"{tag} aligned", alpha=0.9)
    ax.set_title(f"Trajectory comparison (top-down XZ) -- {drive}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[viz] trajectory → {out_path}")


def save_depth_viz(solver, out_dir: str, tag: str):
    """Save a grid of per-keyframe depth colormaps from current submap state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for sm in solver.map.ordered_submaps_by_key():
        if sm.get_lc_status():
            continue
        pc = sm.pointclouds
        if pc is None:
            continue
        for i in range(pc.shape[0]):
            frames.append((sm.get_id(), i, pc[i, :, :, 2]))
    if not frames:
        return
    n = min(len(frames), 12)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    for ax, idx in zip(axes, idxs):
        sid, fi, z = frames[idx]
        z_fin = z[np.isfinite(z)]
        vmax = float(np.percentile(z_fin, 99)) if z_fin.size else 1.0
        im = ax.imshow(z, cmap="turbo", vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"sm={sid} f={fi}  z[{z_fin.min():.2f},{z_fin.max():.2f}]",
                     fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.suptitle(f"Depth maps -- {tag}")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"depth_grid_{tag.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"[viz] depth grid → {out_path}")


def save_pointmap_topdown(solver, out_path: str, tag: str, drive: str):
    """Save top-down (X-Z) scatter of the aggregated world-space point cloud."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    all_pts = []
    all_col = []
    for sm in solver.map.ordered_submaps_by_key():
        if sm.get_lc_status():
            continue
        # Transform camera-space points to world via the graph homography + pose
        poses = sm.get_all_poses_world(solver.graph)  # (N, 4, 4) c2w
        pc = sm.pointclouds
        cols = sm.colors
        conf = sm.conf
        thr = sm.get_conf_threshold() if sm.conf_threshold is not None else 0.0
        if pc is None or cols is None:
            continue
        for i in range(pc.shape[0]):
            mask = conf[i] > thr if conf is not None else np.ones(pc[i].shape[:2], dtype=bool)
            pts = pc[i][mask]  # camera-space (K, 3)
            c = cols[i][mask]
            if pts.size == 0:
                continue
            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            pts_h = np.concatenate([pts, ones], axis=1)  # (K, 4)
            world_pts = (poses[i] @ pts_h.T).T[:, :3]
            # subsample to keep memory bounded
            if world_pts.shape[0] > 4000:
                sel = np.random.choice(world_pts.shape[0], 4000, replace=False)
                world_pts = world_pts[sel]
                c = c[sel]
            all_pts.append(world_pts)
            all_col.append(c)
    if not all_pts:
        return
    all_pts = np.concatenate(all_pts, axis=0)
    all_col = np.concatenate(all_col, axis=0) / 255.0
    # Clip outliers for view
    for axis in range(3):
        lo, hi = np.percentile(all_pts[:, axis], [1, 99])
        m = (all_pts[:, axis] > lo) & (all_pts[:, axis] < hi)
        all_pts = all_pts[m]
        all_col = all_col[m]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(all_pts[:, 0], all_pts[:, 2], c=all_col, s=0.3, alpha=0.5)
    ax.set_title(f"Pointmap top-down (X-Z) -- {tag} -- {drive}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[viz] pointmap → {out_path}")


# -------------------------------------------------------------------------
# Submap snapshot helpers
# -------------------------------------------------------------------------
def dump_pointcloud_stats(solver, tag: str):
    print(f"\n=== {tag} -- submap.pointclouds z-stats ===")
    for sm in solver.map.ordered_submaps_by_key():
        if sm.get_lc_status():
            continue
        pc = sm.pointclouds  # (N, H, W, 3)
        if pc is None:
            print(f"  submap {sm.get_id():4d}: pointclouds=None")
            continue
        z = pc[..., 2].ravel()
        z = z[np.isfinite(z)]
        if z.size == 0:
            print(f"  submap {sm.get_id():4d}: all non-finite")
            continue
        pm = sm.proj_mats
        pm_t_norms = np.linalg.norm(pm[:, :3, 3], axis=1) if pm is not None else None
        pm_t_med = float(np.median(pm_t_norms)) if pm_t_norms is not None else float("nan")
        print(f"  submap {sm.get_id():4d} N={pc.shape[0]:2d}: "
              f"z mean={z.mean():7.3f}  median={np.median(z):7.3f}  "
              f"p10={np.percentile(z,10):7.3f}  p90={np.percentile(z,90):7.3f}  "
              f"proj_mat t-norm median={pm_t_med:.3f}")


def snapshot_submap_pointclouds(solver) -> dict:
    """Return {submap_id: np.copy(pointclouds)} for non-LC submaps."""
    snap = {}
    for sm in solver.map.ordered_submaps_by_key():
        if sm.get_lc_status():
            continue
        if sm.pointclouds is not None:
            snap[sm.get_id()] = sm.pointclouds.copy()
    return snap


def compare_pointcloud_scale(pre: dict, post: dict, tag: str):
    print(f"\n=== {tag} -- per-submap pass1→pass2 z-scale ratio ===")
    for sid in sorted(pre.keys()):
        if sid not in post:
            continue
        z_pre = pre[sid][..., 2].ravel()
        z_post = post[sid][..., 2].ravel()
        m1 = np.isfinite(z_pre); m2 = np.isfinite(z_post)
        z_pre = z_pre[m1]; z_post = z_post[m2]
        if z_pre.size == 0 or z_post.size == 0:
            continue
        # Same-pixel median ratio if shapes match
        if pre[sid].shape == post[sid].shape:
            z_pre_2d = pre[sid][..., 2].ravel()
            z_post_2d = post[sid][..., 2].ravel()
            both = (z_pre_2d > 1e-3) & (z_post_2d > 1e-3) & np.isfinite(z_pre_2d) & np.isfinite(z_post_2d)
            if both.sum() > 100:
                ratio = np.median(z_post_2d[both] / z_pre_2d[both])
                print(f"  submap {sid:4d}: median(post/pre) = {ratio:.4f}  "
                      f"(pre z median={np.median(z_pre):.3f}, post z median={np.median(z_post):.3f})")
                continue
        # Fallback: compare medians
        print(f"  submap {sid:4d}: pre median={np.median(z_pre):.3f}, post median={np.median(z_post):.3f}")


# -------------------------------------------------------------------------
# Main diagnostic flow (mirrors visualize_all.run_da3_slam but breaks after
# pass 1 so we can snapshot intermediate state)
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default="2011_09_29_drive_0026_sync",
                    help="Short drive recommended (drive_0026 = 67 frames, ~2 min)")
    ap.add_argument("--model_name", default="da3-small")
    ap.add_argument("--min_disparity", type=int, default=15)
    ap.add_argument("--strategy", default="refine", choices=["refine", "large-refine"])
    ap.add_argument("--kitti_root", default=KITTI_ROOT)
    args = ap.parse_args()

    # --- Resolve paths / GT ---
    img_folder = image_folder_for(args.drive, args.kitti_root)
    gt_tum = str(PROJECT_ROOT / "evals" / "logs" / "kitti_gt" / f"{args.drive}.tum")
    if not os.path.isfile(gt_tum):
        print(f"ERROR: missing GT TUM {gt_tum}", file=sys.stderr)
        sys.exit(1)
    gt_by_idx = load_gt_positions(gt_tum)
    print(f"Loaded {len(gt_by_idx)} GT poses from {gt_tum}")

    # --- Resolve strategy ---
    from da3_slam.chunk_strategy import STRATEGY_CONFIGS, safe_chunk_size
    from da3_slam.scale_estimator import STRATEGY_SCALE_DEFAULTS
    from da3_slam.solver import Solver
    from da3_slam.da3_wrapper import DA3Wrapper
    import da3_slam.slam_utils as utils

    cfg = STRATEGY_CONFIGS[args.strategy]
    chunk_size = cfg.chunk_size
    overlap = cfg.overlap
    chunk_size = safe_chunk_size(chunk_size, args.model_name, cfg.max_vram_gb)
    scale_method = STRATEGY_SCALE_DEFAULTS.get(args.strategy, "median")
    submap_size = chunk_size - overlap
    print(f"Strategy={args.strategy}: chunk_size={chunk_size} overlap={overlap} "
          f"scale={scale_method}")

    # --- SLAM pass 1 ---
    solver = Solver(init_conf_threshold=25.0, lc_thres=0.95,
                    scale_method=scale_method, overlap=overlap)
    model = DA3Wrapper(model_name=args.model_name, device="cuda")

    image_names = [f for f in glob.glob(os.path.join(img_folder, "*"))
                   if "depth" not in os.path.basename(f).lower()
                   and "txt" not in os.path.basename(f).lower()
                   and "db" not in os.path.basename(f).lower()]
    image_names = utils.sort_images_by_number(image_names)
    print(f"Found {len(image_names)} images at {img_folder}")

    image_names_subset = []
    image_count = 0
    pass1_predictions = []
    t0 = time.time()
    for image_name in image_names:
        img = cv2.imread(image_name)
        enough = solver.flow_tracker.compute_disparity(img, args.min_disparity, False)
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
            pass1_predictions.append({
                "depth": preds["depth"].squeeze(-1).copy(),
                "image_names": list(image_names_subset),
            })
            image_names_subset = image_names_subset[-overlap:]
    print(f"Pass 1 done ({image_count} keyframes, {time.time()-t0:.1f}s).")

    # --- viz output directory ---
    viz_dir = PROJECT_ROOT / "docs" / "eval" / "kitti" / "_diagnose_refine" / args.drive
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Viz output → {viz_dir}")

    # --- Stage A: ATE after pass 1 ---
    print("\n" + "=" * 72)
    print("STAGE A: after pass-1 optimization (baseline-equivalent)")
    print("=" * 72)
    dump_pointcloud_stats(solver, "STAGE A (pass-1 points)")
    est_A, kidx_A = extract_positions(solver, args.drive)
    ate_A, est_A_aln, gt_A = pair_and_ate(est_A, kidx_A, gt_by_idx, "STAGE A")
    save_depth_viz(solver, str(viz_dir), "STAGE_A_pass1")
    save_pointmap_topdown(solver, str(viz_dir / "pointmap_STAGE_A.png"), "STAGE A", args.drive)

    # Snapshot pass-1 pointclouds
    pass1_snap = snapshot_submap_pointclouds(solver)

    # --- Pass 2: refinement ---
    print("\n" + "=" * 72)
    print("PASS 2 refinement")
    print("=" * 72)
    from da3_slam.refinement import run_refinement_pass, reoptimize_after_refinement
    run_refinement_pass(model=model, graph_map=solver.map, pose_graph=solver.graph)

    # --- Stage B: ATE after refinement but BEFORE reopt ---
    # Graph unchanged, but pointclouds are now pass-2. Trajectory should be identical to A.
    print("\n" + "=" * 72)
    print("STAGE B: after pass-2 refinement, BEFORE reoptimize")
    print("=" * 72)
    dump_pointcloud_stats(solver, "STAGE B (pass-2 points)")
    pass2_snap = snapshot_submap_pointclouds(solver)
    compare_pointcloud_scale(pass1_snap, pass2_snap, "pass1 vs pass2")
    est_B, kidx_B = extract_positions(solver, args.drive)
    ate_B, est_B_aln, gt_B = pair_and_ate(est_B, kidx_B, gt_by_idx, "STAGE B")
    save_depth_viz(solver, str(viz_dir), "STAGE_B_pass2")
    save_pointmap_topdown(solver, str(viz_dir / "pointmap_STAGE_B.png"), "STAGE B", args.drive)

    # --- Stage C: reoptimize ---
    reoptimize_after_refinement(solver)
    print("\n" + "=" * 72)
    print("STAGE C: after reoptimize_after_refinement")
    print("=" * 72)
    est_C, kidx_C = extract_positions(solver, args.drive)
    ate_C, est_C_aln, gt_C = pair_and_ate(est_C, kidx_C, gt_by_idx, "STAGE C")
    save_pointmap_topdown(solver, str(viz_dir / "pointmap_STAGE_C.png"), "STAGE C", args.drive)

    # Overlay trajectory comparison
    save_trajectory_plot(
        [("STAGE A", est_A_aln, gt_A),
         ("STAGE B", est_B_aln, gt_B),
         ("STAGE C", est_C_aln, gt_C)],
        str(viz_dir / "trajectory_stages_overlay.png"),
        args.drive,
    )

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Stage A (pass-1 optimized / baseline-equiv): ATE_RMSE = {ate_A:.4f} m")
    print(f"  Stage B (pass-2 done, graph unchanged):      ATE_RMSE = {ate_B:.4f} m")
    print(f"  Stage C (after reoptimize):                  ATE_RMSE = {ate_C:.4f} m")
    if np.isfinite(ate_A) and np.isfinite(ate_C):
        delta = ate_C - ate_A
        print(f"  refine - baseline delta = {delta:+.4f} m "
              f"({'REGRESSION' if delta > 0 else 'improvement'})")
    print(f"  Viz dir: {viz_dir}")


if __name__ == "__main__":
    main()
