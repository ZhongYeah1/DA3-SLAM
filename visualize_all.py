"""Comprehensive visualization: run all SLAM variants and visualize ALL frames + full trajectories.

Produces per-method visualizations in docs/eval/:
  docs/eval/vggt_slam/           -- VGGT-SLAM results
  docs/eval/da3_baseline/        -- DA3-SLAM baseline (17f, 1 pass)
  docs/eval/da3_refine/          -- DA3-SLAM refine (17f, 2 passes)
  docs/eval/da3_large_refine/    -- DA3-SLAM large-refine (40f, 2 passes)
  docs/eval/comparison/          -- side-by-side comparisons

Each method folder contains:
  depth_frame_XXXX.png           -- depth map for each keyframe
  trajectory_3d.png              -- full 3D camera trajectory
  trajectory_topdown.png         -- XZ top-down trajectory
  pointcloud_topdown.png         -- top-down point cloud (subsampled)
  pointcloud_side.png            -- side-view point cloud
  summary.txt                    -- numeric summary

Usage:
  XFORMERS_DISABLED=1 python visualize_all.py
"""
import os
os.environ.setdefault("XFORMERS_DISABLED", "1")

import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- paths ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VGGT_SLAM_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "VGGT-SLAM")
OFFICE_LOOP = os.path.join(PROJECT_ROOT, "office_loop")
EVAL_DIR = os.path.join(PROJECT_ROOT, "docs", "eval")


# ===========================================================================
# DA3-SLAM runner
# ===========================================================================
def run_da3_slam(strategy, chunk_size=None, model_name="da3-small", image_folder=None,
                 min_disparity=50, reopt_method="bounded_clip",
                 scale_method=None, overlap=None):
    """Run DA3-SLAM and return per-submap predictions.

    image_folder: directory of RGB frames. Defaults to OFFICE_LOOP.
    min_disparity: LK-flow threshold for keyframe selection. Default 50 (office_loop
        setting); KITTI driving needs ~15 for reasonable density.
    reopt_method: pass-2 reopt strategy.
        'bounded_clip' (default) -- refinement.reoptimize_after_refinement,
            rebuilds graph with bounded-clip per-edge scale.
        'inplace' -- refinement_inplace.reoptimize_after_refinement_inplace,
            replaces inter-submap factors on existing graph and warm-starts
            LM from pass-1 Values.
        'none' -- skip graph reopt entirely; pass-2 depth/conf are still
            substituted, but the pass-1 GTSAM-optimised poses are kept as-is.
    scale_method: override for STRATEGY_SCALE_DEFAULTS (median / depth-ransac /
        depth-weighted). None → use strategy default.
    overlap: override for ChunkConfig.overlap. None → scale with chunk_size as
        before; integer → use as-is.
    """
    from da3_slam.da3_wrapper import DA3Wrapper, unproject_depth_to_points
    from da3_slam.solver import Solver
    from da3_slam.chunk_strategy import STRATEGY_CONFIGS, safe_chunk_size
    from da3_slam.scale_estimator import STRATEGY_SCALE_DEFAULTS
    import da3_slam.slam_utils as utils
    import cv2, time

    img_root = image_folder if image_folder is not None else OFFICE_LOOP

    cfg = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["baseline"])
    cs = chunk_size if chunk_size else cfg.chunk_size
    if overlap is None:
        overlap = cfg.overlap if chunk_size is None else max(1, int(cs * cfg.overlap / cfg.chunk_size))
    else:
        overlap = max(1, int(overlap))
    cs = safe_chunk_size(cs, model_name, cfg.max_vram_gb)
    num_passes = 2 if strategy != "baseline" else 1
    if scale_method is None:
        scale_method = STRATEGY_SCALE_DEFAULTS.get(strategy, "median")
    submap_size = cs - overlap

    print(f"\n{'='*60}")
    print(f"Running DA3-SLAM: strategy={strategy}, chunk={cs}, overlap={overlap}, passes={num_passes}, scale={scale_method}")
    print(f"{'='*60}")

    solver = Solver(init_conf_threshold=25.0, lc_thres=0.95, scale_method=scale_method, overlap=overlap)
    model = DA3Wrapper(model_name=model_name, device="cuda")

    image_names = [f for f in glob.glob(os.path.join(img_root, "*"))
                   if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower()
                   and "db" not in os.path.basename(f).lower()]
    image_names = utils.sort_images_by_number(image_names)

    all_depths = []
    all_confs = []
    all_extrinsics = []
    all_intrinsics = []
    all_frame_names = []

    image_names_subset = []
    image_count = 0
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
            predictions = solver.run_predictions(image_names_subset, model, 1)
            solver.add_points(predictions)
            solver.graph.optimize()

            all_depths.append(predictions["depth"].squeeze(-1))     # (N, H, W)
            all_confs.append(predictions["depth_conf"])             # (N, H, W)
            all_extrinsics.append(predictions["extrinsic"])         # (N, 3, 4)
            all_intrinsics.append(predictions["intrinsic"])         # (N, 3, 3)
            all_frame_names.extend(image_names_subset)

            image_names_subset = image_names_subset[-overlap:]

    if num_passes >= 2:
        from da3_slam.refinement import run_refinement_pass, reoptimize_after_refinement
        refined = run_refinement_pass(
            model=model, graph_map=solver.map, pose_graph=solver.graph,
        )
        if reopt_method == "inplace":
            from da3_slam.refinement_inplace import reoptimize_after_refinement_inplace
            reoptimize_after_refinement_inplace(solver)
        elif reopt_method == "none":
            pass  # depth/conf swap only, no graph reopt
        else:
            reoptimize_after_refinement(solver)

        # Replace per-submap depth/conf entries with pass-2 outputs.
        # The inference loop appends one entry per submap in temporal order;
        # ordered_submaps_by_key iterates the same submaps in the same order
        # (LC submaps don't appear in either sequence at this point).
        idx = 0
        for submap in solver.map.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            sid = submap.get_id()
            if sid in refined and idx < len(all_depths):
                all_depths[idx] = refined[sid]["depth"]
                all_confs[idx] = refined[sid]["conf"]
            idx += 1

    # Collect world-space trajectories from graph (dedup overlap frames)
    all_positions = []
    seen_frame_names = set()
    submap_boundaries = []  # dedup'd pose indices where a new submap starts (first unique frame)
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        poses = submap.get_all_poses_world(solver.graph)  # (N, 4, 4) c2w
        frame_ids = submap.get_frame_ids()
        first_unique_for_this_submap = True
        for frame_id, p in zip(frame_ids, poses):
            if frame_id in seen_frame_names:
                continue
            seen_frame_names.add(frame_id)
            if first_unique_for_this_submap and len(all_positions) > 0:
                submap_boundaries.append(len(all_positions))
            first_unique_for_this_submap = False
            all_positions.append(p[:3, 3])

    return {
        "depths": all_depths,
        "confs": all_confs,
        "extrinsics": all_extrinsics,
        "intrinsics": all_intrinsics,
        "frame_names": all_frame_names,
        "positions": np.array(all_positions),
        "submap_boundaries": np.array(submap_boundaries, dtype=int),
        "solver": solver,
        "image_count": image_count,
    }


# ===========================================================================
# VGGT-SLAM runner
# ===========================================================================
def run_vggt_slam(image_folder=None, min_disparity=50):
    """Run VGGT-SLAM and return predictions.

    image_folder: directory of RGB frames. Defaults to VGGT_SLAM_ROOT/office_loop.
    min_disparity: LK-flow threshold for keyframe selection.
    """
    print(f"\n{'='*60}")
    print(f"Running VGGT-SLAM")
    print(f"{'='*60}")

    # Temporarily add VGGT-SLAM to path
    sys.path.insert(0, VGGT_SLAM_ROOT)
    sys.path.insert(0, os.path.join(VGGT_SLAM_ROOT, "third_party", "vggt"))

    import vggt_slam.slam_utils as vutils
    from vggt_slam.solver import Solver as VSolver
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    import cv2

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
    vggt_model.eval().cuda()

    img_root = image_folder if image_folder is not None else os.path.join(VGGT_SLAM_ROOT, "office_loop")
    image_names = [f for f in glob.glob(os.path.join(img_root, "*"))
                   if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower()
                   and "db" not in os.path.basename(f).lower()]
    image_names = vutils.sort_images_by_number(image_names)

    solver = VSolver(init_conf_threshold=25.0, lc_thres=0.95)

    all_depths = []
    all_confs = []
    all_extrinsics = []
    all_intrinsics = []
    all_frame_names = []

    image_names_subset = []
    image_count = 0
    for image_name in image_names:
        img = cv2.imread(image_name)
        enough = solver.flow_tracker.compute_disparity(img, min_disparity, False)
        if enough:
            image_names_subset.append(image_name)
            image_count += 1

        if len(image_names_subset) == 17 or image_name == image_names[-1]:
            if len(image_names_subset) < 2:
                image_names_subset = []
                continue
            predictions = solver.run_predictions(image_names_subset, vggt_model, 1, None, None)
            solver.add_points(predictions)
            solver.graph.optimize()

            all_depths.append(predictions["depth"].squeeze(-1))
            all_confs.append(predictions["depth_conf"])
            all_extrinsics.append(predictions["extrinsic"])
            all_intrinsics.append(predictions["intrinsic"])
            all_frame_names.extend(image_names_subset)

            image_names_subset = image_names_subset[-1:]

    # Collect world-space trajectories from graph (dedup overlap frames)
    all_positions = []
    seen_frame_names = set()
    submap_boundaries = []
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        poses = submap.get_all_poses_world(solver.graph)
        frame_ids = submap.get_frame_ids()
        first_unique_for_this_submap = True
        for frame_id, p in zip(frame_ids, poses):
            if frame_id in seen_frame_names:
                continue
            seen_frame_names.add(frame_id)
            if first_unique_for_this_submap and len(all_positions) > 0:
                submap_boundaries.append(len(all_positions))
            first_unique_for_this_submap = False
            all_positions.append(p[:3, 3])

    # Remove VGGT paths
    sys.path.remove(VGGT_SLAM_ROOT)
    sys.path.remove(os.path.join(VGGT_SLAM_ROOT, "third_party", "vggt"))

    return {
        "depths": all_depths,
        "confs": all_confs,
        "extrinsics": all_extrinsics,
        "intrinsics": all_intrinsics,
        "frame_names": all_frame_names,
        "positions": np.array(all_positions),
        "submap_boundaries": np.array(submap_boundaries, dtype=int),
        "solver": solver,
        "image_count": image_count,
    }


# ===========================================================================
# Visualization
# ===========================================================================
def save_all_depth_frames(result, out_dir, label):
    """Save depth visualization for ALL keyframes."""
    os.makedirs(out_dir, exist_ok=True)

    frame_idx = 0
    for submap_i, depth_batch in enumerate(result["depths"]):
        n_frames = depth_batch.shape[0]
        for f in range(n_frames):
            d = depth_batch[f]  # (H, W)
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            im = ax.imshow(d, cmap="turbo", vmin=0, vmax=6)
            ax.set_title(f"{label} -- frame {frame_idx:04d} [{d.min():.2f}, {d.max():.2f}]", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.03)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"depth_frame_{frame_idx:04d}.png"), dpi=120)
            plt.close()
            frame_idx += 1

    print(f"  Saved {frame_idx} depth frames to {out_dir}")


def save_trajectory(result, out_dir, label, color="blue"):
    """Save 3D and top-down trajectory plots."""
    os.makedirs(out_dir, exist_ok=True)
    pos = result["positions"]  # (M, 3)
    boundaries = result.get("submap_boundaries", np.array([], dtype=int))

    # Persist raw positions + boundary indices for off-line analysis
    np.save(os.path.join(out_dir, "positions.npy"), pos)
    np.save(os.path.join(out_dir, "submap_boundaries.npy"), boundaries)

    # 3D trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-o", color=color, markersize=2, linewidth=1)
    ax.scatter(*pos[0], color="green", s=100, zorder=5, label="Start")
    ax.scatter(*pos[-1], color="red", s=100, zorder=5, label="End")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"{label} -- 3D Trajectory ({len(pos)} poses)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_3d.png"), dpi=150)
    plt.close()

    # Top-down XZ with submap boundary markers
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.plot(pos[:, 0], pos[:, 2], "-o", color=color, markersize=3, linewidth=1)
    ax.plot(pos[0, 0], pos[0, 2], "g*", markersize=15, label="Start")
    ax.plot(pos[-1, 0], pos[-1, 2], "r^", markersize=12, label="End")
    for i, b in enumerate(boundaries):
        if b < len(pos):
            ax.plot(pos[b, 0], pos[b, 2], "kx", markersize=10, mew=2,
                    label="Submap boundary" if i == 0 else None)
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title(f"{label} -- Top-Down (XZ) [{len(boundaries)} submap boundaries]")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_topdown.png"), dpi=150)
    plt.close()
    print(f"  Saved trajectory plots to {out_dir}")


def save_pointcloud(result, out_dir, label, color="steelblue"):
    """Save top-down and side-view point cloud plots."""
    os.makedirs(out_dir, exist_ok=True)
    solver = result["solver"]

    # Collect all world-space points
    all_pts = []
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        pts = submap.get_points_in_world_frame(solver.graph)
        if pts is not None and len(pts) > 0:
            all_pts.append(pts)
    if not all_pts:
        print(f"  No points to visualize for {label}")
        return
    all_pts = np.concatenate(all_pts, axis=0)

    # Subsample to 200k
    if len(all_pts) > 200000:
        idx = np.random.choice(len(all_pts), 200000, replace=False)
        all_pts = all_pts[idx]

    # Top-down XZ
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_pts[:, 0], all_pts[:, 2], s=0.05, c=color, alpha=0.3)
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title(f"{label} -- Point Cloud Top-Down ({len(all_pts):,} pts)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pointcloud_topdown.png"), dpi=150)
    plt.close()

    # Side XY
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_pts[:, 0], all_pts[:, 1], s=0.05, c=color, alpha=0.3)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"{label} -- Point Cloud Side View ({len(all_pts):,} pts)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pointcloud_side.png"), dpi=150)
    plt.close()
    print(f"  Saved point cloud plots to {out_dir}")


def save_summary(result, out_dir, label):
    """Save numeric summary."""
    os.makedirs(out_dir, exist_ok=True)
    pos = result["positions"]
    all_d = np.concatenate([d.flatten() for d in result["depths"]])
    total_frames = sum(d.shape[0] for d in result["depths"])

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"{label}\n{'='*50}\n")
        f.write(f"Total keyframes: {total_frames}\n")
        f.write(f"Total submaps: {len(result['depths'])}\n")
        f.write(f"Trajectory poses: {len(pos)}\n")
        f.write(f"Depth range: [{all_d.min():.4f}, {all_d.max():.4f}]\n")
        f.write(f"Depth mean: {all_d.mean():.4f}\n")
        f.write(f"Depth std: {all_d.std():.4f}\n")
        if len(pos) > 0:
            traj_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
            f.write(f"Trajectory length: {traj_len:.4f}\n")
            f.write(f"Trajectory extent X: [{pos[:,0].min():.4f}, {pos[:,0].max():.4f}]\n")
            f.write(f"Trajectory extent Y: [{pos[:,1].min():.4f}, {pos[:,1].max():.4f}]\n")
            f.write(f"Trajectory extent Z: [{pos[:,2].min():.4f}, {pos[:,2].max():.4f}]\n")
    print(f"  Saved summary to {out_dir}/summary.txt")


def save_comparison(results_dict, out_dir):
    """Save side-by-side comparison plots across all methods."""
    os.makedirs(out_dir, exist_ok=True)

    # Trajectory overlay
    fig, ax = plt.subplots(figsize=(8, 12))
    colors = {"VGGT-SLAM": "blue", "DA3 baseline": "orange", "DA3 refine": "green", "DA3 large-refine": "red"}
    markers = {"VGGT-SLAM": "o", "DA3 baseline": "s", "DA3 refine": "^", "DA3 large-refine": "D"}
    for name, res in results_dict.items():
        pos = res["positions"]
        if len(pos) == 0:
            continue
        ax.plot(pos[:, 0], pos[:, 2], "-", color=colors.get(name, "gray"),
                markersize=2, linewidth=1.5, label=name, alpha=0.8)
    ax.set_xlabel("X (scene units)"); ax.set_ylabel("Z (scene units)")
    ax.set_title("All Trajectories -- Top-Down (XZ)")
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_trajectories_topdown.png"), dpi=150)
    plt.close()

    # Depth comparison for frame 0 (first submap, first frame of each method)
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5 * len(results_dict), 4))
    if len(results_dict) == 1:
        axes = [axes]
    for i, (name, res) in enumerate(results_dict.items()):
        d = res["depths"][0][0]  # first submap, first frame
        im = axes[i].imshow(d, cmap="turbo", vmin=0, vmax=6)
        axes[i].set_title(f"{name}\n[{d.min():.2f}, {d.max():.2f}]", fontsize=9)
        axes[i].axis("off")
    plt.colorbar(im, ax=axes, fraction=0.02)
    plt.suptitle("Depth Comparison -- Frame 0", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "depth_comparison_frame0.png"), dpi=150)
    plt.close()

    # Depth comparison for a mid-sequence frame
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5 * len(results_dict), 4))
    if len(results_dict) == 1:
        axes = [axes]
    for i, (name, res) in enumerate(results_dict.items()):
        # Use 4th submap, frame 8 (or last available)
        si = min(3, len(res["depths"]) - 1)
        fi = min(8, res["depths"][si].shape[0] - 1)
        d = res["depths"][si][fi]
        im = axes[i].imshow(d, cmap="turbo", vmin=0, vmax=6)
        axes[i].set_title(f"{name}\n[{d.min():.2f}, {d.max():.2f}]", fontsize=9)
        axes[i].axis("off")
    plt.colorbar(im, ax=axes, fraction=0.02)
    plt.suptitle("Depth Comparison -- Mid-Sequence", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "depth_comparison_mid.png"), dpi=150)
    plt.close()

    # Summary table
    with open(os.path.join(out_dir, "comparison_summary.txt"), "w") as f:
        f.write("Method Comparison Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Method':<22} {'Frames':>7} {'Submaps':>8} {'Depth Range':>16} {'Traj Length':>12}\n")
        f.write("-" * 70 + "\n")
        for name, res in results_dict.items():
            nf = sum(d.shape[0] for d in res["depths"])
            ns = len(res["depths"])
            all_d = np.concatenate([d.flatten() for d in res["depths"]])
            pos = res["positions"]
            tl = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)) if len(pos) > 1 else 0
            f.write(f"{name:<22} {nf:>7} {ns:>8} [{all_d.min():.2f}, {all_d.max():.2f}]{'':<4} {tl:>12.4f}\n")

    print(f"  Saved comparison to {out_dir}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    results = {}

    # ---- 1. VGGT-SLAM ----
    try:
        vggt_result = run_vggt_slam()
        results["VGGT-SLAM"] = vggt_result
        vggt_dir = os.path.join(EVAL_DIR, "vggt_slam")
        save_all_depth_frames(vggt_result, vggt_dir, "VGGT-SLAM")
        save_trajectory(vggt_result, vggt_dir, "VGGT-SLAM", color="blue")
        save_pointcloud(vggt_result, vggt_dir, "VGGT-SLAM", color="steelblue")
        save_summary(vggt_result, vggt_dir, "VGGT-SLAM")
        # Free GPU memory
        del vggt_result["solver"]
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"VGGT-SLAM failed: {e}")
        import traceback; traceback.print_exc()

    # ---- 2. DA3 baseline ----
    try:
        da3_base = run_da3_slam("baseline")
        results["DA3 baseline"] = da3_base
        base_dir = os.path.join(EVAL_DIR, "da3_baseline")
        save_all_depth_frames(da3_base, base_dir, "DA3 baseline")
        save_trajectory(da3_base, base_dir, "DA3 baseline", color="orange")
        save_pointcloud(da3_base, base_dir, "DA3 baseline", color="darkorange")
        save_summary(da3_base, base_dir, "DA3 baseline")
    except Exception as e:
        print(f"DA3 baseline failed: {e}")
        import traceback; traceback.print_exc()

    # ---- 3. DA3 refine ----
    try:
        da3_ref = run_da3_slam("refine")
        results["DA3 refine"] = da3_ref
        ref_dir = os.path.join(EVAL_DIR, "da3_refine")
        save_all_depth_frames(da3_ref, ref_dir, "DA3 refine")
        save_trajectory(da3_ref, ref_dir, "DA3 refine", color="green")
        save_pointcloud(da3_ref, ref_dir, "DA3 refine", color="forestgreen")
        save_summary(da3_ref, ref_dir, "DA3 refine")
    except Exception as e:
        print(f"DA3 refine failed: {e}")
        import traceback; traceback.print_exc()

    # ---- 4. DA3 large-refine ----
    try:
        da3_lr = run_da3_slam("large-refine", chunk_size=40)
        results["DA3 large-refine"] = da3_lr
        lr_dir = os.path.join(EVAL_DIR, "da3_large_refine")
        save_all_depth_frames(da3_lr, lr_dir, "DA3 large-refine")
        save_trajectory(da3_lr, lr_dir, "DA3 large-refine", color="red")
        save_pointcloud(da3_lr, lr_dir, "DA3 large-refine", color="firebrick")
        save_summary(da3_lr, lr_dir, "DA3 large-refine")
    except Exception as e:
        print(f"DA3 large-refine failed: {e}")
        import traceback; traceback.print_exc()

    # ---- 5. Cross-method comparison ----
    if results:
        save_comparison(results, os.path.join(EVAL_DIR, "comparison"))

    print(f"\n{'='*60}")
    print(f"All results saved to: {EVAL_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
