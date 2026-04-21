"""Run SLAM on one KITTI drive for one method and persist artifacts.

For (method, drive) this produces:
  evals/logs/kitti/<method>_<drive>.tum         -- TUM-format trajectory
  docs/eval/kitti/<method>/<drive>/
      depth_frame_XXXX.npy                       -- raw metric depth (float32)
      depth_frame_XXXX.png                       -- colormapped viz
      trajectory_topdown.png
      trajectory_3d.png
      summary.txt
      loop_closure_count.txt                     -- diagnostic

Resumable: if the TUM file already exists and is non-empty AND the expected
depth .npy count matches the saved trajectory poses, skip the SLAM run.

Methods:
  vggt_slam          -- VGGT-SLAM baseline
  da3_baseline       -- DA3-SLAM baseline (1 pass)
  da3_refine         -- DA3-SLAM refine (2 passes)
  da3_large_refine   -- DA3-SLAM large-refine (chunk=40, 2 passes)

Usage:
  XFORMERS_DISABLED=1 python evals/kitti_runner.py \\
      --method da3_baseline --drive 2011_09_26_drive_0002_sync \\
      --model_name da3-small
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure visualize_all.py (at project root) is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np

from evals.kitti_drives import image_folder_for, KITTI_ROOT

# Mapping from method name → (runner_name, strategy, chunk_size, reopt_method).
# reopt_method is only consulted for DA3 strategies with num_passes >= 2.
METHOD_CONFIG = {
    "vggt_slam":          ("vggt_slam", None,           None, None),
    "da3_baseline":       ("da3_slam",  "baseline",     None, None),
    "da3_refine":         ("da3_slam",  "refine",       None, "bounded_clip"),
    "da3_refine_inplace": ("da3_slam",  "refine",       None, "inplace"),
    "da3_refine_none":    ("da3_slam",  "refine",       None, "none"),
    "da3_large_refine":   ("da3_slam",  "large-refine", 40,   "bounded_clip"),
}


def _save_depth_artifacts(result, solver, out_dir: str) -> int:
    """Save per-keyframe depth as .npy + .png, keyed by the ORIGINAL KITTI
    frame index (parsed from filenames via submap.get_frame_ids()).

    Overlap frames (same KITTI index in multiple submaps) are written once.
    The iteration mirrors `_save_tum` so the depth manifest, TUM trajectory,
    and LiDAR-projected GT PNGs all align on KITTI frame index.

    Filename convention matches the KITTI GT naming: `depth_frame_NNNNNNNNNN.npy`
    (10-digit zero-padded KITTI frame index). Also writes
    `depth_frame_index.json` listing all saved indices in write order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    seen: set[int] = set()
    saved_indices: list[int] = []

    for submap, depth_batch in zip(solver.map.ordered_submaps_by_key(), result["depths"]):
        if submap.get_lc_status():
            continue
        frame_ids = submap.get_frame_ids()
        for fid, d in zip(frame_ids, depth_batch):
            kidx = int(fid)
            if kidx in seen:
                continue
            seen.add(kidx)
            d = np.asarray(d, dtype=np.float32)
            stem = f"depth_frame_{kidx:010d}"
            np.save(os.path.join(out_dir, f"{stem}.npy"), d)

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            vmax = float(np.percentile(d[np.isfinite(d)], 99)) if np.any(np.isfinite(d)) else 1.0
            im = ax.imshow(d, cmap="turbo", vmin=0, vmax=max(vmax, 1e-6))
            ax.set_title(f"KITTI frame {kidx:010d} depth [{d.min():.2f}, {d.max():.2f}] m", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.03)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=90)
            plt.close()
            saved_indices.append(kidx)

    with open(os.path.join(out_dir, "depth_frame_index.json"), "w") as f:
        json.dump({"kitti_indices": saved_indices}, f)
    return len(saved_indices)


def _save_tum(solver, tum_path: str, frame_names: list[str]) -> int:
    """Write TUM trajectory. Timestamps are frame indices (0, 1, 2, ...) since KITTI
    SLAM runs don't know real timestamps. evo_ape aligns by index for TUM when
    timestamps don't overlap, but here we need timestamps that MATCH the GT TUM
    from oxts. We synthesize 10Hz increments to align with KITTI's ~10 Hz capture.
    """
    os.makedirs(os.path.dirname(tum_path) or ".", exist_ok=True)
    # solver.map.write_poses_to_file uses its own indexing. We fall back to a
    # manual write driven by the graph so we control timestamps.
    import gtsam
    from da3_slam.slam_utils import decompose_camera

    lines = []
    seen = set()
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        poses = submap.get_all_poses_world(solver.graph)  # (N, 4, 4) c2w
        frame_ids = submap.get_frame_ids()
        for fid, T in zip(frame_ids, poses):
            if fid in seen:
                continue
            seen.add(fid)
            t = T[:3, 3]
            R = T[:3, :3]
            q = _rotmat_to_quat_xyzw(R)
            # Align with GT TUM via frame-index-based timestamps (10 Hz)
            ts = 0.1 * float(fid)
            lines.append(
                f"{ts:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )
    with open(tum_path, "w") as f:
        f.writelines(lines)
    return len(lines)


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw])


def _count_existing_npys(dir_path: str) -> int:
    if not os.path.isdir(dir_path):
        return 0
    return sum(1 for n in os.listdir(dir_path) if n.startswith("depth_frame_") and n.endswith(".npy"))


def _has_new_format_depth(dir_path: str) -> bool:
    """A run has the KITTI-idx-named depth artifacts iff the manifest exists."""
    return os.path.isfile(os.path.join(dir_path, "depth_frame_index.json"))


def _count_tum_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for l in f if l.strip())


def run_once(method: str, drive: str, model_name: str, kitti_root: str,
             tum_path: str, out_dir: str, min_disparity: int = 15,
             force: bool = False,
             chunk_size_override: int | None = None,
             overlap_override: int | None = None,
             scale_method_override: str | None = None) -> dict:
    """Run one (method, drive) pair. Returns a summary dict.

    The *_override arguments come from ablation sweeps. They have no effect
    on the stored `method` name; the caller is responsible for using a
    distinct `method` key (e.g. "da3_baseline_scale_ransac") so outputs
    land in a separate directory.
    """
    runner_name, strategy, chunk_size, reopt_method = METHOD_CONFIG[method]
    img_folder = image_folder_for(drive, kitti_root)

    if chunk_size_override is not None:
        chunk_size = chunk_size_override

    # Resumability check: require BOTH TUM and the new-format depth manifest.
    # Runs predating the fixed depth naming (see _save_depth_artifacts) have
    # sequential-indexed .npy files with no manifest; those need to be redone.
    tum_lines = _count_tum_lines(tum_path)
    npy_count = _count_existing_npys(out_dir)
    if not force and tum_lines > 0 and npy_count > 0 and _has_new_format_depth(out_dir):
        print(f"[skip] {method} {drive}: existing {tum_lines} TUM poses + {npy_count} depth .npy")
        return {"method": method, "drive": drive, "status": "skipped",
                "tum_poses": tum_lines, "depth_frames": npy_count}

    # Lazy import to keep --help fast and avoid touching GPU unless needed
    from visualize_all import run_da3_slam, run_vggt_slam, save_trajectory, save_summary

    if runner_name == "vggt_slam":
        result = run_vggt_slam(image_folder=img_folder, min_disparity=min_disparity)
    else:
        da3_kwargs = dict(chunk_size=chunk_size, model_name=model_name,
                          image_folder=img_folder, min_disparity=min_disparity)
        if reopt_method is not None:
            da3_kwargs["reopt_method"] = reopt_method
        if overlap_override is not None:
            da3_kwargs["overlap"] = overlap_override
        if scale_method_override is not None:
            da3_kwargs["scale_method"] = scale_method_override
        result = run_da3_slam(strategy, **da3_kwargs)

    os.makedirs(out_dir, exist_ok=True)

    # 1. TUM trajectory
    n_poses = _save_tum(result["solver"], tum_path, result.get("frame_names", []))
    print(f"  wrote {n_poses} TUM poses → {tum_path}")

    # 2. Depth artifacts (.npy for eval, .png for viz), KITTI-idx-named
    n_depth = _save_depth_artifacts(result, result["solver"], out_dir)
    print(f"  wrote {n_depth} depth frames → {out_dir}")

    # 3. Trajectory plots + summary (reuse visualize_all helpers)
    save_trajectory(result, out_dir, f"{method} {drive}")
    save_summary(result, out_dir, f"{method} {drive}")

    # 4. Loop-closure count diagnostic
    lc_count = result["solver"].graph.get_num_loops()
    with open(os.path.join(out_dir, "loop_closure_count.txt"), "w") as f:
        f.write(f"{lc_count}\n")

    return {
        "method": method, "drive": drive, "status": "ok",
        "tum_poses": n_poses, "depth_frames": n_depth, "loop_closures": lc_count,
        "submaps": len(result["depths"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=list(METHOD_CONFIG.keys()))
    ap.add_argument("--drive", required=True, help="KITTI drive name (e.g. 2011_09_26_drive_0002_sync)")
    ap.add_argument("--model_name", default="da3-small", help="DA3 model variant (ignored for vggt_slam)")
    ap.add_argument("--kitti_root", default=KITTI_ROOT)
    ap.add_argument("--tum_path", default=None, help="Override TUM output path")
    ap.add_argument("--out_dir", default=None, help="Override per-(method, drive) output directory")
    ap.add_argument("--min_disparity", type=int, default=15,
                    help="LK keyframe-selection threshold (KITTI needs ~15; office_loop uses 50)")
    ap.add_argument("--force", action="store_true", help="Rerun even if outputs exist")
    ap.add_argument("--experiment_name", default=None,
                    help="Override output method-name used for TUM/output dir (for ablation sweeps; method dispatch stays unchanged)")
    ap.add_argument("--chunk_size", type=int, default=None,
                    help="Override METHOD_CONFIG chunk_size (e.g. sweep chunk sizes)")
    ap.add_argument("--overlap", type=int, default=None,
                    help="Override chunk overlap (DA3-SLAM only)")
    ap.add_argument("--scale_method", default=None,
                    choices=["median", "depth-ransac", "depth-weighted"],
                    help="Override scale estimator (DA3-SLAM only); omit to use strategy default")
    args = ap.parse_args()

    name = args.experiment_name or args.method
    tum_path = args.tum_path or os.path.join(
        PROJECT_ROOT, "evals", "logs", "kitti", f"{name}_{args.drive}.tum"
    )
    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "docs", "eval", "kitti", name, args.drive
    )

    summary = run_once(
        method=args.method, drive=args.drive, model_name=args.model_name,
        kitti_root=args.kitti_root, tum_path=tum_path, out_dir=out_dir,
        min_disparity=args.min_disparity, force=args.force,
        chunk_size_override=args.chunk_size,
        overlap_override=args.overlap,
        scale_method_override=args.scale_method,
    )
    summary["method"] = name
    print("RESULT:", summary)


if __name__ == "__main__":
    main()
