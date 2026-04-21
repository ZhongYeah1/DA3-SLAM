"""Refinement pass: re-run DA3 with GTSAM-optimized poses for better depth."""

import numpy as np
import torch
from termcolor import colored

from da3_slam.da3_wrapper import DA3Wrapper, unproject_depth_to_points
from da3_slam.map import GraphMap
from da3_slam.graph import PoseGraph
from da3_slam.slam_utils import decompose_camera
from da3_slam.scale_estimator import (
    estimate_scale_median,
    estimate_scale_depth_ransac,
    estimate_scale_depth_weighted,
)


def extract_poses_from_graph(
    graph_map: GraphMap,
    pose_graph: PoseGraph,
) -> dict:
    """Extract optimized w2c extrinsics and intrinsics per submap from the GTSAM graph.

    Extrinsics are re-anchored to each submap's frame 0 before being returned:
    `extrinsics_4x4[0]` is identity for every submap, and `extrinsics_4x4[i]`
    is the relative w2c from frame 0 to frame i in the GTSAM-optimized
    trajectory. This matches `ref_view_strategy="first"` conventions and is
    what `DA3Wrapper.forward_with_poses` expects as conditioning.

    Returns:
        Dict mapping submap_id -> (extrinsics_4x4, intrinsics, image_names)
        where extrinsics_4x4 is (N, 4, 4) w2c (frame 0 = I) and
        intrinsics is (N, 3, 3).
    """
    result = {}
    for submap in graph_map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue

        submap_id = submap.get_id()
        n_frames = len(submap.poses)
        extrinsics_4x4 = np.empty((n_frames, 4, 4), dtype=np.float32)
        intrinsics_3x3 = np.empty((n_frames, 3, 3), dtype=np.float32)

        for i in range(n_frames):
            node_id = submap_id + i
            H = pose_graph.get_homography(node_id)  # SL(4) 4x4

            # Projection matrix: P = K @ w2c = K @ inv(H)
            K_4x4 = submap.proj_mats[i]  # (4, 4) with K in top-left 3x3
            proj = K_4x4 @ np.linalg.inv(H)
            proj = proj / proj[-1, -1]

            # Decompose P into K, R_c2w, t_c2w
            # decompose_camera returns c2w components (R inverted, t in world)
            K, R_c2w, t_c2w, scale = decompose_camera(proj)

            # Convert c2w to w2c: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ t_c2w
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R_c2w.T
            w2c[:3, 3] = -R_c2w.T @ t_c2w
            extrinsics_4x4[i] = w2c

            # Use stored intrinsics (pass-1 values)
            intrinsics_3x3[i] = K_4x4[:3, :3]

        image_names = submap.img_names
        result[submap_id] = (extrinsics_4x4, intrinsics_3x3, image_names)

    return result


def _compute_pass2_scale_alignment(
    pass1_poses_4x4: np.ndarray,
    extrinsic_pass2: np.ndarray,
) -> float:
    n = pass1_poses_4x4.shape[0]
    if n < 2:
        return 1.0

    identity_4x4 = np.eye(4, dtype=np.float32)
    # assert np.absolute(pass1_poses_4x4[0] - identity_4x4).max() < 1e-1, f"pass1_poses_4x4[0] = {pass1_poses_4x4[0]}"
    # assert np.absolute(extrinsic_pass2[0] - identity_4x4[:3]).max() < 1e-1, f"extrinsic_pass2[0] = {extrinsic_pass2[0]}"

    c2w1 = np.linalg.inv(pass1_poses_4x4)
    p2_4x4 = np.zeros((n, 4, 4), dtype=np.float32)
    p2_4x4[:, :3, :] = extrinsic_pass2
    p2_4x4[:, 3, 3] = 1.0
    c2w2 = np.linalg.inv(p2_4x4)

    t1 = c2w1[1:, :3, 3]
    t2 = c2w2[1:, :3, 3]
    mean1 = float(np.mean(np.linalg.norm(t1, axis=1)))
    mean2 = float(np.mean(np.linalg.norm(t2, axis=1)))

    if mean1 < 1e-6 or mean2 < 1e-6:
        return 1.0

    scale = mean1 / mean2
    if scale < 0.01 or scale > 100.0:
        return 1.0
    return scale


def run_refinement_pass(
    model: DA3Wrapper,
    graph_map: GraphMap,
    pose_graph: PoseGraph,
) -> dict:
    """Re-run DA3 on each submap with GTSAM-optimized poses; align scale; substitute.

    For each non-loop-closure submap:
      1. Extract conditioning poses (graph-world w2c) from the GTSAM graph.
      2. Run DA3 with pose conditioning (pass 2).
      3. Compute per-submap scale alignment scalar (mean-translation ratio
         of pass-2 vs pass-1's stored submap.poses).
      4. Apply scalar to pass-2 depth AND to pass-2 w2c translations so they
         all live in pass-1's internal metric scale.
      5. Unproject the scaled depth to camera-space pointclouds.
      6. Substitute submap.poses, submap.proj_mats, pointclouds, conf, colors
         with the pass-2 (scale-aligned) values.

    Returns:
        Dict[submap_id -> {"depth": (N, H, W) scale-aligned pass-2 depth,
                            "conf":  (N, H, W) pass-2 confidence}].
        Caller can use this to overwrite per-submap depth in evaluation
        outputs so depth metrics reflect pass-2 quality.

    Args:
        model: DA3Wrapper instance.
        graph_map: The map containing all submaps.
        pose_graph: The optimized GTSAM pose graph.
    """
    print(colored("=== Starting refinement pass (pass 2) ===", "cyan"))

    submap_poses = extract_poses_from_graph(graph_map, pose_graph)
    refined_depth_by_submap: dict = {}

    for submap_id, (extrinsics, intrinsics, image_names) in submap_poses.items():
        submap = graph_map.get_submap(submap_id)
        n_frames = extrinsics.shape[0]
        print(f"  Refining submap {submap_id} ({n_frames} frames)...")

        # Snapshot pass-1's submap.poses BEFORE substitution. 
        pass1_w2c_for_scale_ref = submap.poses.copy()
        np.set_printoptions(precision=4, suppress=True, linewidth=160)
        deviation_from_identity = float(np.linalg.norm(pass1_w2c_for_scale_ref[0] - np.eye(4)))

        with torch.no_grad():
            result = model.forward_with_poses(image_names, extrinsics, intrinsics)

        w2c_pass2_3x4 = result["extrinsic"]  # (N, 3, 4) w2c, frame 0 = identity
        depth = result["depth"].squeeze(-1)  # (N, H, W)
        conf = result["depth_conf"]          # (N, H, W)

        scale = _compute_pass2_scale_alignment(pass1_w2c_for_scale_ref, w2c_pass2_3x4)
        if abs(scale - 1.0) < 1e-9:
            print(colored(f"    submap {submap_id}: scale alignment = 1.0 (degenerate or trivial)", "yellow"))
        else:
            print(colored(f"    submap {submap_id}: scale alignment = {scale:.4f}", "green"))

        # Apply scale to DEPTH and to pass-2 w2c TRANSLATIONS so they live in
        # the same metric scale (pass-1 internal scale). Rotations and
        # intrinsics are scale-invariant.
        depth_aligned = depth * scale
        points_cam_pass2 = unproject_depth_to_points(depth_aligned, w2c_pass2_3x4, result["intrinsic"])

        # Build scaled (N, 4, 4) w2c from pass-2's (N, 3, 4) extrinsics.
        n_p2 = w2c_pass2_3x4.shape[0]
        w2c_pass2_4x4 = np.zeros((n_p2, 4, 4), dtype=np.float32)
        w2c_pass2_4x4[:, :3, :] = w2c_pass2_3x4
        w2c_pass2_4x4[:, 3, 3] = 1.0
        w2c_pass2_4x4[:, :3, 3] *= scale  # match depth scaling

        # print('ext_pass2_4x4 - submap.poses', (w2c_pass2_4x4 - submap.poses))

        # Build pass-2 proj_mats (K embedded in 4x4 with identity tail).
        K_pass2 = result["intrinsic"]  # (N, 3, 3) pixel-space K
        proj_mats_pass2 = np.tile(np.eye(4, dtype=np.float32), (n_p2, 1, 1))
        proj_mats_pass2[:, :3, :3] = K_pass2

        images_01 = result["images"]
        colors = (images_01.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

        submap.poses = w2c_pass2_4x4
        submap.proj_mats = proj_mats_pass2
        submap.pointclouds = points_cam_pass2
        submap.colors = colors
        submap.conf = conf
        submap.conf_threshold = np.percentile(conf, 25.0) + 1e-6
        submap.set_conf_masks(conf)

        refined_depth_by_submap[submap_id] = {
            "depth": depth_aligned.astype(np.float32),
            "conf": conf.astype(np.float32),
        }

    print(colored("=== Refinement pass complete ===", "cyan"))
    return refined_depth_by_submap


def _estimate_scale_between_submaps(
    prior_submap,
    current_submap,
    frame_id_prev: int,
    match_idx_curr: int,
    scale_method: str,
) -> float:
    """Estimate scale factor between two submaps using their (possibly refined) pointclouds.

    Mirrors the scale estimation logic in Solver.add_edge() but operates on
    the already-stored submap data (which may have been updated by the
    refinement pass).
    """
    current_conf = current_submap.get_conf_masks_frame(match_idx_curr)
    prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)
    good_mask = (
        (prior_conf > prior_submap.get_conf_threshold())
        * (current_conf > current_submap.get_conf_threshold())
    )
    good_mask = good_mask.reshape(-1)

    if np.sum(good_mask) < 100:
        good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
        if np.sum(good_mask) < 100:
            good_mask = (prior_conf > 0).reshape(-1)

    if scale_method in ("depth-ransac", "depth-weighted"):
        depth_prev = prior_submap.get_frame_pointcloud(frame_id_prev)[:, :, 2]
        depth_curr = current_submap.get_frame_pointcloud(match_idx_curr)[:, :, 2]
        conf_prev_full = prior_submap.get_conf_masks_frame(frame_id_prev)
        conf_curr_full = current_submap.get_conf_masks_frame(match_idx_curr)
        if scale_method == "depth-ransac":
            scale_factor, quality = estimate_scale_depth_ransac(
                depth_prev, conf_prev_full, depth_curr, conf_curr_full
            )
        else:
            scale_factor, quality = estimate_scale_depth_weighted(
                depth_prev, conf_prev_full, depth_curr, conf_curr_full
            )
        if quality < 0.3:
            P_temp = (
                np.linalg.inv(prior_submap.proj_mats[frame_id_prev])
                @ current_submap.proj_mats[match_idx_curr]
            )
            t1 = (
                P_temp[0:3, 0:3]
                @ current_submap.get_frame_pointcloud(match_idx_curr)
                .reshape(-1, 3)[good_mask]
                .T
            ).T
            t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]
            scale_factor, _ = estimate_scale_median(t1, t2)
    else:
        P_temp = (
            np.linalg.inv(prior_submap.proj_mats[frame_id_prev])
            @ current_submap.proj_mats[match_idx_curr]
        )
        t1 = (
            P_temp[0:3, 0:3]
            @ current_submap.get_frame_pointcloud(match_idx_curr)
            .reshape(-1, 3)[good_mask]
            .T
        ).T
        t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]
        scale_factor, _ = estimate_scale_median(t1, t2)

    # Guard against degenerate scale factors
    if scale_factor < 0.01 or scale_factor > 100.0:
        print(colored(f"  WARNING: degenerate scale factor {scale_factor:.6f}, clamping to 1.0", "red"))
        scale_factor = 1.0

    return scale_factor


REOPT_SCALE_CLIP_ALPHA = 0.05


def _bounded_clip_scale(pass1_scale: float, pass2_scale: float,
                        alpha: float = REOPT_SCALE_CLIP_ALPHA) -> float:
    """Clip pass-2 scale factor to pass-1 × [1-α, 1+α].

    Pure function so it can be unit-tested without GTSAM. See
    ``reoptimize_after_refinement`` for the caller context.
    """
    lo = pass1_scale * (1.0 - alpha)
    hi = pass1_scale * (1.0 + alpha)
    return float(np.clip(pass2_scale, lo, hi))


def reoptimize_after_refinement(solver) -> None:
    """Rebuild the GTSAM graph after refinement with bounded-clip scale updates.

    For each inter-submap edge, the new scale factor is computed from
    pass-2 pointclouds and clipped to pass-1 × [1-α, 1+α]. This lets
    refine nudge the trajectory where pass-2 agrees with pass-1 while
    bounding the per-edge drift that caused the attempt-3 regression on
    long no-LC chains.

    The pass-1 scale_factor per edge is stored in solver._edge_log as the
    6th element of each tuple/list (added by Solver.add_edge).

    Args:
        solver: Solver instance with .map, .graph, .overlap, .scale_method,
                and ._edge_log attributes.
    """
    print(colored(
        f"=== Re-optimizing graph after refinement (bounded-clip α={REOPT_SCALE_CLIP_ALPHA}) ===",
        "cyan",
    ))

    graph_map = solver.map
    overlap = solver.overlap
    scale_method = solver.scale_method

    # Build a new PoseGraph from scratch
    new_graph = PoseGraph()

    # Replay all edges from the original construction order.
    # Edge log format: [submap_id_curr, frame_id_curr, submap_id_prev,
    #                   frame_id_prev, is_loop_closure, pass1_scale_factor]
    # (6th element added by Solver.add_edge for attempt-5 bounded clip.)
    for edge_idx, edge_entry in enumerate(solver._edge_log):
        submap_id_curr = edge_entry[0]
        frame_id_curr = edge_entry[1]
        submap_id_prev = edge_entry[2]
        frame_id_prev = edge_entry[3]
        is_loop_closure = edge_entry[4]
        pass1_scale = edge_entry[5] if len(edge_entry) > 5 else 1.0

        current_submap = graph_map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)

        if submap_id_prev is not None:
            prior_submap = graph_map.get_submap(submap_id_prev)
            overlapping_node_id_prev = submap_id_prev + frame_id_prev

            # Determine the correct matching frame in the current submap
            if is_loop_closure:
                match_idx_curr = frame_id_curr
            else:
                match_idx_curr = min(overlap - 1, len(current_submap.pointclouds) - 1)

            scale_pass2 = _estimate_scale_between_submaps(
                prior_submap, current_submap,
                frame_id_prev, match_idx_curr,
                scale_method,
            )
            scale_factor = _bounded_clip_scale(pass1_scale, scale_pass2)
            clipped_flag = " (CLIPPED)" if scale_factor != scale_pass2 else ""
            print(colored(
                f"  edge {edge_idx}: pass1={pass1_scale:.4f} pass2={scale_pass2:.4f} "
                f"→ used={scale_factor:.4f}{clipped_flag}",
                "green",
            ))

            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            H_overlap = (
                np.linalg.inv(prior_submap.proj_mats[frame_id_prev])
                @ current_submap.proj_mats[match_idx_curr]
                @ H_scale
            )

            H_w_match = new_graph.get_homography(overlapping_node_id_prev) @ H_overlap

            # Add the match frame node (skip for loop closure backward edges)
            if not is_loop_closure:
                new_graph.add_homography(submap_id_curr + match_idx_curr, H_w_match)

            # Add between factor: prev_last → curr[match_idx], direct.
            new_graph.add_between_factor(
                overlapping_node_id_prev,
                submap_id_curr + match_idx_curr,
                H_overlap,
                new_graph.intra_submap_noise,
            )

        else:
            # First submap: prior at identity
            assert submap_id_curr == 0 and frame_id_curr == 0
            new_graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            new_graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)

        # Loop closure only gets the between factor, no inner submap edges
        if is_loop_closure:
            continue

        # Determine entry point (same logic as solver.py)
        if submap_id_prev is not None:
            if is_loop_closure:
                entry_idx = frame_id_curr
            else:
                entry_idx = min(overlap - 1, len(current_submap.pointclouds) - 1)
        else:
            entry_idx = 0

        # Inner-submap edges: forward from entry_idx.
        # local_w2c[i] = w2c of frame i in this submap's local frame
        # (frame 0 ≈ identity), NOT in graph-world. The relative
        # transform local_w2c[i-1] @ inv(local_w2c[i]) is frame-pair
        # geometry that holds in any consistent frame.
        local_w2c = current_submap.get_all_poses()
        for index in range(entry_idx + 1, len(local_w2c)):
            H_inner = local_w2c[index - 1] @ np.linalg.inv(local_w2c[index])
            current_node = new_graph.get_homography(submap_id_curr + index - 1) @ H_inner
            new_graph.add_homography(submap_id_curr + index, current_node)
            new_graph.add_between_factor(
                submap_id_curr + index - 1,
                submap_id_curr + index,
                H_inner,
                new_graph.inner_submap_noise,
            )

        # Inner-submap edges: backward from entry_idx to 0
        for index in range(entry_idx - 1, -1, -1):
            H_inner = local_w2c[index + 1] @ np.linalg.inv(local_w2c[index])
            current_node = new_graph.get_homography(submap_id_curr + index + 1) @ H_inner
            new_graph.add_homography(submap_id_curr + index, current_node)
            new_graph.add_between_factor(
                submap_id_curr + index + 1,
                submap_id_curr + index,
                H_inner,
                new_graph.inner_submap_noise,
            )

    # Preserve loop closure count
    new_graph.num_loop_closures = solver.graph.num_loop_closures
    # Preserve auto-calibration homographies if any
    new_graph.auto_cal_H_mats = solver.graph.auto_cal_H_mats.copy()

    # Optimize the new graph
    print(colored("  Running Levenberg-Marquardt on rebuilt graph...", "cyan"))
    new_graph.optimize()

    # Replace solver's graph
    solver.graph = new_graph
    print(colored("=== Re-optimization complete ===", "cyan"))
