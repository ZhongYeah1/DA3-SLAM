# da3_slam/scene_export.py
"""
Export a finished SLAM run to a self-contained directory for post-hoc viewing.

Usage (called from main.py after graph optimisation):
    from da3_slam.scene_export import export_scene
    export_scene(solver, strategy_name="baseline", output_dir="logs/scene_baseline")
"""

import os
import json
import numpy as np
import torch


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]


def _denorm_frames(frames) -> np.ndarray:
    """ImageNet-normalized tensor/array (S,3,H,W) -> uint8 (S,H,W,3)."""
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().float().numpy()
    imgs_01 = np.clip(frames * _IMAGENET_STD + _IMAGENET_MEAN, 0.0, 1.0)
    return (imgs_01.transpose(0, 2, 3, 1) * 255).astype(np.uint8)


def export_scene(solver, strategy_name: str, output_dir: str) -> None:
    """
    Serialise the current SLAM result to *output_dir*.

    Directory layout
    ----------------
    output_dir/
    ├── metadata.json
    ├── poses.npy          float32 (N,4,4)  c2w world poses, one per frame
    ├── intrinsics.npy     float32 (N,3,3)  one K per frame
    ├── submap_info.json   {frame_idx: submap_id, ...}
    └── frames/
        ├── 0000.npz       points(H,W,3) float32, colors(H,W,3) uint8, conf(H,W) float32
        └── ...

    Parameters
    ----------
    solver : da3_slam.solver.Solver
        Finished solver (after graph.optimize() and optional refinement).
    strategy_name : str
        Human-readable label stored in metadata (e.g. "baseline", "refine").
    output_dir : str
        Destination directory (created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    graph = solver.graph
    graph_map = solver.map

    all_poses = []        # c2w (4,4) per frame
    all_intrinsics = []   # K (3,3) per frame
    submap_info = {}      # frame_idx (int) -> submap_id (int)
    num_submaps = 0
    frame_idx = 0         # global counter across all non-LC submaps

    for submap in graph_map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue

        num_submaps += 1
        submap_id = submap.get_id()

        # --- world-space points and per-frame conf masks ---
        # Returns lists of length S: each element is (H,W,3) or (H,W) bool
        point_list, _frame_ids, conf_mask_list = submap.get_points_list_in_world_frame(graph)

        # --- c2w poses (S,4,4) ---
        c2w_poses = submap.get_all_poses_world(graph)   # (S,4,4)

        # --- intrinsics (S,3,3) from proj_mats (S,4,4) ---
        # proj_mats stores K_4x4 (forward intrinsics, NOT inverse).
        # Verified in solver.py add_points: K_4x4[:, :3, :3] = intrinsics_cam,
        # then passed directly to submap.add_all_points as the 'intrinsics_inv'
        # parameter (misleading name) -- actual content is the camera matrix K.
        intrinsics = submap.proj_mats[:, :3, :3]         # (S,3,3)  K upper-left block

        # --- denormalized colors (S,H,W,3) uint8 ---
        frames_raw = submap.get_all_frames()             # tensor or ndarray (S,3,H,W)
        colors = _denorm_frames(frames_raw)              # (S,H,W,3) uint8

        # raw conf values (S,H,W) float32
        conf_raw = submap.conf                           # (S,H,W)

        S = c2w_poses.shape[0]
        for s in range(S):
            np.savez_compressed(
                os.path.join(frames_dir, f"{frame_idx:04d}.npz"),
                points=point_list[s].astype(np.float32),   # (H,W,3) world-space
                colors=colors[s],                          # (H,W,3) uint8
                conf=conf_raw[s].astype(np.float32),       # (H,W)
            )
            all_poses.append(c2w_poses[s].astype(np.float32))
            all_intrinsics.append(intrinsics[s].astype(np.float32))
            submap_info[frame_idx] = submap_id
            frame_idx += 1

    # Guard against empty result (all submaps were LC submaps or none processed)
    if not all_poses:
        print("No non-LC submaps to export")
        return

    # Save aggregated arrays
    np.save(os.path.join(output_dir, "poses.npy"),      np.stack(all_poses,      axis=0))
    np.save(os.path.join(output_dir, "intrinsics.npy"), np.stack(all_intrinsics, axis=0))

    with open(os.path.join(output_dir, "submap_info.json"), "w") as f:
        # JSON keys must be strings
        json.dump({str(k): v for k, v in submap_info.items()}, f, indent=2)

    metadata = {
        "strategy":    strategy_name,
        "num_submaps": num_submaps,
        "num_frames":  frame_idx,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[scene_export] Saved {frame_idx} frames ({num_submaps} submaps) -> {output_dir}")
