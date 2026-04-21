"""Headline VGGT-SLAM vs DA3 nested-giant comparison visualization.

Per representative drive, writes to
    docs/eval/kitti/comparison/vggt_vs_nested/<drive>/
        trajectory_xz.png      GT + both methods Umeyama-aligned (top-down XZ)
        depth_grid.png         sample frames x 3 rows: RGB / VGGT / nested
        pointcloud_topdown.png back-projected point clouds in GT-aligned frame
        metrics.txt            headline ATE / depth metrics pulled from logs

Uses cached per-drive artifacts in docs/eval/kitti/{vggt_slam,da3_baseline_nested}/
and evals/logs/kitti*/. No SLAM is re-run.

Usage:
    python evals/compare_vggt_vs_nested.py                # both representatives
    python evals/compare_vggt_vs_nested.py <drive>        # one drive
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image
from scipy.spatial.transform import Rotation as SciR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.kitti_drives import REPRESENTATIVE_DRIVES, KITTI_ROOT, date_from_drive

METHODS = {
    "vggt_slam": {"label": "VGGT-SLAM (baseline)", "color": "#1f77b4"},
    "da3_baseline_nested": {"label": "DA3 nested-giant (ours)", "color": "#ff7f0e"},
}
GT_COLOR = "black"

# KITTI raw image_02 intrinsics (cam_02 typical values across val drives).
KITTI_RAW_W, KITTI_RAW_H = 1242, 375
KITTI_RAW_FX, KITTI_RAW_FY = 721.5377, 721.5377
KITTI_RAW_CX, KITTI_RAW_CY = 609.559, 172.854


def load_tum(path: Path):
    data = np.loadtxt(path)
    ts = data[:, 0]
    t = data[:, 1:4]
    q = data[:, 4:8]  # qx qy qz qw
    Rmats = SciR.from_quat(q).as_matrix()
    return ts, t, Rmats


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Umeyama Sim(3): find s, R, t with dst ≈ s R src + t."""
    N = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    Sigma = (Y.T @ X) / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    Rmat = U @ S @ Vt
    var_x = (X ** 2).sum() / N
    s = np.trace(np.diag(D) @ S) / var_x if var_x > 1e-12 else 1.0
    tvec = mu_d - s * Rmat @ mu_s
    return float(s), Rmat, tvec


def apply_sim3(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * pts @ R.T + t


def kitti_indices_from_depth_files(run_dir: Path) -> np.ndarray:
    files = sorted(glob.glob(str(run_dir / "depth_frame_*.npy")))
    rx = re.compile(r"depth_frame_(\d+)\.npy$")
    idxs = []
    for f in files:
        m = rx.search(f)
        if m:
            idxs.append(int(m.group(1)))
    return np.array(idxs, dtype=int)


def match_pred_to_gt(pred_pos: np.ndarray, gt_pos: np.ndarray,
                     kitti_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_matched, gt_matched) using KITTI index → GT row (10 Hz).

    Works because KITTI GT TUM is one row per KITTI index (0.1 s step).
    Truncates pred if it has more poses than retained depth frames.
    """
    n_pred = len(pred_pos)
    n_idx = len(kitti_idx)
    n = min(n_pred, n_idx)
    pred_matched = pred_pos[:n]
    # Cap GT index to available GT rows.
    gt_ids = np.clip(kitti_idx[:n], 0, len(gt_pos) - 1)
    gt_matched = gt_pos[gt_ids]
    return pred_matched, gt_matched


def read_run(method: str, drive: str):
    run_dir = PROJECT_ROOT / "docs" / "eval" / "kitti" / method / drive
    if not run_dir.is_dir():
        return None
    pos_path = run_dir / "positions.npy"
    if not pos_path.is_file():
        return None
    pos = np.load(pos_path)
    kitti_idx = kitti_indices_from_depth_files(run_dir)
    depth_files = sorted(glob.glob(str(run_dir / "depth_frame_*.npy")))
    return {
        "run_dir": run_dir,
        "positions": pos,
        "kitti_idx": kitti_idx,
        "depth_files": depth_files,
    }


def read_metrics_json(method: str, drive: str) -> dict | None:
    p = PROJECT_ROOT / "evals" / "logs" / "kitti" / f"{method}_{drive}_depth.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def pose_from_tum_row(t: np.ndarray, Rmat: np.ndarray) -> np.ndarray:
    P = np.eye(4)
    P[:3, :3] = Rmat
    P[:3, 3] = t
    return P


def backproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return (points_cam_Nx3, mask_flat_bool) from a HxW depth."""
    H, W = depth.shape
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)
    d = depth[vs[:, None], us[None, :]]
    valid = (d > 0) & np.isfinite(d)
    d = d[valid]
    uu = uu[valid]
    vv = vv[valid]
    x = (uu - cx) * d / fx
    y = (vv - cy) * d / fy
    z = d
    return np.stack([x, y, z], axis=-1), valid


def method_intrinsics(H: int, W: int) -> tuple[float, float, float, float]:
    fx = KITTI_RAW_FX * (W / KITTI_RAW_W)
    fy = KITTI_RAW_FY * (H / KITTI_RAW_H)
    cx = KITTI_RAW_CX * (W / KITTI_RAW_W)
    cy = KITTI_RAW_CY * (H / KITTI_RAW_H)
    return fx, fy, cx, cy


def _data_extent_xz(gt_pos: np.ndarray, runs: dict, pad: float = 20.0):
    xs = [gt_pos[:, 0]]
    zs = [gt_pos[:, 2]]
    for info in runs.values():
        if "pred_aligned" in info:
            xs.append(info["pred_aligned"][:, 0])
            zs.append(info["pred_aligned"][:, 2])
    xs = np.concatenate(xs); zs = np.concatenate(zs)
    xmin, xmax = float(xs.min()) - pad, float(xs.max()) + pad
    zmin, zmax = float(zs.min()) - pad, float(zs.max()) + pad
    return xmin, xmax, zmin, zmax


def _figsize_for_extent(xmin, xmax, zmin, zmax, min_dim=4.0, max_dim=14.0):
    dx = xmax - xmin
    dz = zmax - zmin
    if dx <= 0 or dz <= 0:
        return (8.0, 8.0)
    ratio = dx / dz
    # Pick a figure that keeps aspect equal while fitting in a reasonable box.
    if ratio >= 1:
        w = max_dim
        h = max(min_dim, min(max_dim, max_dim / ratio))
    else:
        h = max_dim
        w = max(min_dim, min(max_dim, max_dim * ratio))
    return (w, h)


def plot_trajectory_xz(drive: str, out_path: Path, runs: dict, gt_pos: np.ndarray,
                       headline: dict):
    xmin, xmax, zmin, zmax = _data_extent_xz(gt_pos, runs)
    fig, ax = plt.subplots(figsize=_figsize_for_extent(xmin, xmax, zmin, zmax))

    ax.plot(gt_pos[:, 0], gt_pos[:, 2], color=GT_COLOR, lw=2.2, label="KITTI GT", zorder=3)

    for method, info in runs.items():
        if info is None:
            continue
        pred = info["pred_aligned"]
        meta = METHODS[method]
        ate = headline.get(method, {}).get("ate_rmse_m")
        lbl = f"{meta['label']}"
        if ate is not None:
            lbl += f"  ATE={ate:.2f} m"
        ax.plot(pred[:, 0], pred[:, 2], color=meta["color"], lw=1.6, label=lbl, zorder=4)
        ax.scatter(pred[0, 0], pred[0, 2], color=meta["color"], marker="o", s=40, zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Trajectory XZ -- {drive}  (Umeyama-aligned to GT)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_depth_grid(drive: str, out_path: Path, runs: dict, kitti_idx_common: np.ndarray,
                    n_samples: int = 6):
    """Rows: RGB, VGGT depth, nested depth. Cols: sample KITTI frames."""
    # Sample kitti indices evenly from the matched set.
    if len(kitti_idx_common) < n_samples:
        n_samples = len(kitti_idx_common)
    sel = np.linspace(0, len(kitti_idx_common) - 1, n_samples, dtype=int)
    sample_idxs = kitti_idx_common[sel]

    rgb_dir = Path(KITTI_ROOT) / date_from_drive(drive) / drive / "image_02" / "data"

    fig, axes = plt.subplots(3, n_samples, figsize=(3.2 * n_samples, 6.4))
    if n_samples == 1:
        axes = axes[:, None]

    # Per-method depth normalization (use global percentiles over sampled frames).
    method_order = ["vggt_slam", "da3_baseline_nested"]
    method_rows = {m: i + 1 for i, m in enumerate(method_order)}
    per_method_depths = {}
    for method in method_order:
        info = runs.get(method)
        if info is None:
            continue
        depths = []
        for kidx in sample_idxs:
            ix = int(np.where(info["kitti_idx"] == kidx)[0][0])
            depths.append(np.load(info["depth_files"][ix]))
        per_method_depths[method] = depths

    per_method_vmin = {}
    per_method_vmax = {}
    for m, ds in per_method_depths.items():
        allvals = np.concatenate([d[np.isfinite(d) & (d > 0)].ravel() for d in ds])
        per_method_vmin[m] = float(np.percentile(allvals, 5))
        per_method_vmax[m] = float(np.percentile(allvals, 95))

    for ci, kidx in enumerate(sample_idxs):
        # RGB row.
        rgb_path = rgb_dir / f"{kidx:010d}.png"
        ax_rgb = axes[0, ci]
        if rgb_path.is_file():
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            ax_rgb.imshow(rgb)
        ax_rgb.set_title(f"frame {kidx:010d}", fontsize=9)
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        if ci == 0:
            ax_rgb.set_ylabel("RGB", fontsize=10)

        for mi, method in enumerate(method_order):
            ax = axes[method_rows[method], ci]
            ds = per_method_depths.get(method)
            if ds is None:
                ax.set_axis_off()
                continue
            d = ds[ci]
            im = ax.imshow(d, cmap="turbo",
                           vmin=per_method_vmin[method],
                           vmax=per_method_vmax[method])
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(METHODS[method]["label"], fontsize=10)
            if ci == n_samples - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
                cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"Depth maps -- {drive}\n"
                 "(VGGT is relative scale; DA3 nested-giant is near-metric)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_pointcloud(info: dict, pred_rot: np.ndarray, pred_trans: np.ndarray,
                     sim3: tuple, max_frames: int = 8, stride: int = 6,
                     max_pts_per_frame: int = 8000) -> np.ndarray:
    s, Rmat, tvec = sim3
    n = min(len(info["kitti_idx"]), len(pred_trans))
    step = max(1, n // max_frames)
    pts_all = []
    for i in range(0, n, step):
        depth = np.load(info["depth_files"][i])
        H, W = depth.shape
        fx, fy, cx, cy = method_intrinsics(H, W)
        pts_cam, _ = backproject(depth, fx, fy, cx, cy, stride=stride)
        if pts_cam.shape[0] == 0:
            continue
        if pts_cam.shape[0] > max_pts_per_frame:
            sel = np.random.default_rng(0).choice(pts_cam.shape[0], max_pts_per_frame, replace=False)
            pts_cam = pts_cam[sel]
        # Camera frame → pred-world via (R_c2w, t_c2w).
        Rc = pred_rot[i]
        tc = pred_trans[i]
        pts_world = pts_cam @ Rc.T + tc
        pts_aligned = apply_sim3(pts_world, s, Rmat, tvec)
        pts_all.append(pts_aligned)
    if not pts_all:
        return np.zeros((0, 3))
    return np.concatenate(pts_all, axis=0)


def plot_pointcloud_topdown(drive: str, out_path: Path, runs: dict,
                            gt_pos: np.ndarray):
    xmin, xmax, zmin, zmax = _data_extent_xz(gt_pos, runs, pad=40.0)
    # Restrict point cloud display to a band ±perp_band m around the trajectory
    # hull so we highlight drivable-area structure.
    fw, fh = _figsize_for_extent(xmin, xmax, zmin, zmax, max_dim=11.0)
    fig, axes = plt.subplots(1, 2, figsize=(fw * 2 + 1, fh + 0.5), sharex=True, sharey=True)
    method_order = ["vggt_slam", "da3_baseline_nested"]
    for ax, method in zip(axes, method_order):
        info = runs.get(method)
        if info is None or "pointcloud" not in info:
            ax.set_axis_off()
            continue
        pts = info["pointcloud"]
        meta = METHODS[method]
        if pts.shape[0] > 0:
            # Keep a horizontal ground-height band + inside the plotting bbox.
            y = pts[:, 1]
            y_ok = (y > np.percentile(y, 2)) & (y < np.percentile(y, 98))
            box_ok = ((pts[:, 0] > xmin) & (pts[:, 0] < xmax) &
                      (pts[:, 2] > zmin) & (pts[:, 2] < zmax))
            pts_v = pts[y_ok & box_ok]
            ax.scatter(pts_v[:, 0], pts_v[:, 2], s=0.5, c=meta["color"], alpha=0.35)
        ax.plot(gt_pos[:, 0], gt_pos[:, 2], color=GT_COLOR, lw=1.8, label="KITTI GT", zorder=3)
        pred = info["pred_aligned"]
        ax.plot(pred[:, 0], pred[:, 2], color=meta["color"], lw=1.5, label=meta["label"], zorder=4)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(meta["label"])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Point cloud top-down -- {drive}  "
                 "(back-projected depth in GT-aligned frame)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_metrics_txt(out_path: Path, drive: str, headline: dict):
    lines = [f"Headline metrics -- {drive}",
             "=" * 60,
             f"{'method':<30}{'ATE RMSE (m)':>15}{'AbsRel':>10}{'RMSE_d':>10}{'δ<1.25':>10}"]
    for method, h in headline.items():
        lbl = METHODS[method]["label"]
        ate = h.get("ate_rmse_m")
        abs_rel = h.get("abs_rel")
        rmse_d = h.get("rmse_d")
        delta = h.get("delta_1_25")
        lines.append(
            f"{lbl:<30}"
            f"{('%.3f' % ate) if ate is not None else '--':>15}"
            f"{('%.4f' % abs_rel) if abs_rel is not None else '--':>10}"
            f"{('%.3f' % rmse_d) if rmse_d is not None else '--':>10}"
            f"{('%.4f' % delta) if delta is not None else '--':>10}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def process_drive(drive: str) -> None:
    out_dir = PROJECT_ROOT / "docs" / "eval" / "kitti" / "comparison" / "vggt_vs_nested" / drive
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_tum = PROJECT_ROOT / "evals" / "logs" / "kitti_gt" / f"{drive}.tum"
    if not gt_tum.is_file():
        print(f"[{drive}] missing GT TUM {gt_tum}")
        return
    _, gt_t, _ = load_tum(gt_tum)

    runs: dict = {}
    headline: dict = {}
    common_kidx: np.ndarray | None = None

    for method in METHODS:
        info = read_run(method, drive)
        if info is None:
            print(f"[{drive}] no cached run for {method}")
            continue
        tum_path = PROJECT_ROOT / "evals" / "logs" / "kitti" / f"{method}_{drive}.tum"
        if not tum_path.is_file():
            print(f"[{drive}] missing TUM {tum_path}")
            continue
        _, pred_t, pred_R = load_tum(tum_path)

        # Prefer positions.npy as the pose list (matches depth_frame ordering).
        # Fall back to TUM translation if counts differ by > 2.
        if abs(len(info["positions"]) - len(pred_t)) <= 2:
            pred_pos = info["positions"][:min(len(info["positions"]), len(pred_t))]
            pred_R = pred_R[:len(pred_pos)]
            pred_t = pred_t[:len(pred_pos)]
        else:
            pred_pos = pred_t

        pred_matched, gt_matched = match_pred_to_gt(pred_pos, gt_t, info["kitti_idx"])
        if len(pred_matched) < 5:
            print(f"[{drive}] too few matches for {method}")
            continue

        s, Rmat, tvec = umeyama(pred_matched, gt_matched)
        pred_aligned = apply_sim3(pred_matched, s, Rmat, tvec)
        ate_rmse = float(np.sqrt(((pred_aligned - gt_matched) ** 2).sum(axis=1).mean()))

        info["pred_aligned"] = pred_aligned
        info["sim3"] = (s, Rmat, tvec)
        info["pred_rot"] = pred_R[:len(pred_matched)]
        info["pred_trans"] = pred_matched
        info["pointcloud"] = build_pointcloud(info, info["pred_rot"], info["pred_trans"], info["sim3"])
        runs[method] = info

        m = read_metrics_json(method, drive) or {}
        metrics = m.get("metrics", {})
        headline[method] = {
            "ate_rmse_m": ate_rmse,
            "abs_rel": metrics.get("abs_rel"),
            "rmse_d": metrics.get("rmse"),
            "delta_1_25": metrics.get("delta_1_25"),
        }

        # Track union of kitti indices.
        if common_kidx is None:
            common_kidx = info["kitti_idx"]
        else:
            common_kidx = np.intersect1d(common_kidx, info["kitti_idx"])

    if not runs:
        print(f"[{drive}] no methods loaded -- skipping")
        return

    plot_trajectory_xz(drive, out_dir / "trajectory_xz.png", runs, gt_t, headline)
    if common_kidx is not None and len(common_kidx) > 0:
        plot_depth_grid(drive, out_dir / "depth_grid.png", runs, common_kidx)
    plot_pointcloud_topdown(drive, out_dir / "pointcloud_topdown.png", runs, gt_t)
    write_metrics_txt(out_dir / "metrics.txt", drive, headline)
    print(f"[{drive}] wrote outputs to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", nargs="?", default=None,
                    help="Optional drive name (default: both representatives)")
    args = ap.parse_args()
    drives = [args.drive] if args.drive else REPRESENTATIVE_DRIVES
    for d in drives:
        process_drive(d)


if __name__ == "__main__":
    main()
