"""Per-drive sparse depth evaluation against KITTI GT depth.

For a given (method, drive) pair we load predicted depth maps from
    docs/eval/kitti/<method>/<drive>/depth_frame_NNNNNNNNNN.npy

where NNNNNNNNNN is the 10-digit ORIGINAL KITTI frame index (matching the
GT PNG naming under <KITTI_ROOT>/val/<drive>/proj_depth/groundtruth/image_02/).

Pairing: pred[kidx] is matched to GT[kidx] by KITTI frame index. Keyframes
that have no corresponding GT (no LiDAR projection available at that frame)
are skipped rather than mispaired.

Legacy fallback: older runs used sequential naming (`depth_frame_XXXX.npy`,
4-digit counter) with no KITTI-index mapping. For those, we fall back to
sorted-index pairing and warn that results are approximate.

Each frame is scale-aligned by per-frame median(GT/pred) over the valid
mask (Eigen crop + median align, standard KITTI depth convention).

Writes one JSON summarizing AbsRel, RMSE, RMSE_log, and δ<1.25^k (k=1..3).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals.kitti_drives import gt_depth_folder_for


MIN_DEPTH_M = 1e-3
MAX_DEPTH_M = 80.0  # KITTI depth benchmark cap


def _eigen_crop_mask(h: int, w: int) -> np.ndarray:
    """Eigen crop (Eigen et al., 2014) applied to all KITTI depth benchmarks."""
    mask = np.zeros((h, w), dtype=bool)
    y0 = int(0.40810811 * h)
    y1 = int(0.99189189 * h)
    x0 = int(0.03594771 * w)
    x1 = int(0.96405229 * w)
    mask[y0:y1, x0:x1] = True
    return mask


def _resize_to(pred: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a 2D float array to (H, W) using PIL."""
    th, tw = target_hw
    if pred.shape == (th, tw):
        return pred
    img = Image.fromarray(pred.astype(np.float32), mode="F").resize((tw, th), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def _load_gt_png(path: str) -> np.ndarray:
    """KITTI GT depth PNGs are uint16 / 256 in metres. Zero = invalid."""
    arr = np.array(Image.open(path), dtype=np.float32)
    return arr / 256.0


def _per_frame_metrics(gt: np.ndarray, pred: np.ndarray) -> dict | None:
    """Return a dict of metric contributions (sum over valid pixels)
    so we can aggregate across frames before taking averages.
    Returns None if no valid pixels.
    """
    h, w = gt.shape
    pred_r = _resize_to(pred, (h, w))

    crop = _eigen_crop_mask(h, w)
    valid = (gt > MIN_DEPTH_M) & (gt < MAX_DEPTH_M) & crop & np.isfinite(pred_r) & (pred_r > 0)
    n = int(valid.sum())
    if n == 0:
        return None

    g = gt[valid]
    p = pred_r[valid]

    # Per-frame median scale alignment
    scale = float(np.median(g / p))
    p = p * scale
    p = np.clip(p, MIN_DEPTH_M, MAX_DEPTH_M)

    # Metric contributions (sums so aggregation is exact)
    abs_rel = np.abs(g - p) / g
    sq_rel = (g - p) ** 2 / g
    rmse = (g - p) ** 2
    rmse_log = (np.log(g) - np.log(p)) ** 2

    ratio = np.maximum(g / p, p / g)
    d1 = (ratio < 1.25).astype(np.float64)
    d2 = (ratio < 1.25 ** 2).astype(np.float64)
    d3 = (ratio < 1.25 ** 3).astype(np.float64)

    return {
        "n": n,
        "scale": scale,
        "abs_rel_sum": float(abs_rel.sum()),
        "sq_rel_sum":  float(sq_rel.sum()),
        "rmse_sqsum":  float(rmse.sum()),
        "rmse_log_sqsum": float(rmse_log.sum()),
        "d1_sum": float(d1.sum()),
        "d2_sum": float(d2.sum()),
        "d3_sum": float(d3.sum()),
    }


def _is_new_format_name(stem: str) -> bool:
    """Return True iff stem is depth_frame_<10-digit KITTI idx>."""
    parts = stem.split("_")
    if len(parts) != 3 or parts[0] != "depth" or parts[1] != "frame":
        return False
    return len(parts[2]) == 10 and parts[2].isdigit()


def _pair_by_kitti_idx(pred_dir: str, gt_pngs: list[str]) -> tuple[list[tuple[str, str]], int, bool]:
    """Pair pred .npy to GT .png by KITTI frame index parsed from filename stem.

    Returns (pairs, unmatched_preds, is_new_format).
    """
    pred_npys = sorted(glob.glob(os.path.join(pred_dir, "depth_frame_*.npy")))
    gt_by_idx = {int(os.path.splitext(os.path.basename(p))[0]): p for p in gt_pngs}

    # Decide format from the first pred file
    if not pred_npys:
        return [], 0, False
    first_stem = os.path.splitext(os.path.basename(pred_npys[0]))[0]
    is_new = _is_new_format_name(first_stem)

    if not is_new:
        # Legacy: pair by sorted index (old, incorrect, but preserved for back-compat)
        n_pairs = min(len(pred_npys), len(gt_pngs))
        return [(gt_pngs[i], pred_npys[i]) for i in range(n_pairs)], 0, False

    pairs: list[tuple[str, str]] = []
    unmatched = 0
    for p in pred_npys:
        stem = os.path.splitext(os.path.basename(p))[0]
        kidx = int(stem.split("_")[-1])
        gt_path = gt_by_idx.get(kidx)
        if gt_path is None:
            unmatched += 1
            continue
        pairs.append((gt_path, p))
    return pairs, unmatched, True


def evaluate(method: str, drive: str, pred_dir: str, kitti_root: str) -> dict:
    gt_folder = gt_depth_folder_for(drive, kitti_root)
    gt_pngs = sorted(glob.glob(os.path.join(gt_folder, "*.png")))
    if not gt_pngs:
        raise FileNotFoundError(f"No GT PNGs under {gt_folder}")

    pred_npys_all = glob.glob(os.path.join(pred_dir, "depth_frame_*.npy"))
    if not pred_npys_all:
        raise FileNotFoundError(f"No depth_frame_*.npy under {pred_dir}")

    pairs, unmatched, is_new_format = _pair_by_kitti_idx(pred_dir, gt_pngs)
    if not is_new_format:
        print(f"  [warn] {pred_dir}: legacy sequential naming -- pairing by sorted index "
              f"is approximate; re-run SLAM for accurate depth metrics.")

    totals = {
        "n": 0,
        "abs_rel_sum": 0.0, "sq_rel_sum": 0.0,
        "rmse_sqsum": 0.0,   "rmse_log_sqsum": 0.0,
        "d1_sum": 0.0, "d2_sum": 0.0, "d3_sum": 0.0,
    }
    scales = []
    frames_used = 0
    frames_empty = 0

    for gt_path, pred_path in pairs:
        gt = _load_gt_png(gt_path)
        pred = np.load(pred_path).astype(np.float32)
        contrib = _per_frame_metrics(gt, pred)
        if contrib is None:
            frames_empty += 1
            continue
        for k in ("n", "abs_rel_sum", "sq_rel_sum", "rmse_sqsum",
                  "rmse_log_sqsum", "d1_sum", "d2_sum", "d3_sum"):
            totals[k] += contrib[k]
        scales.append(contrib["scale"])
        frames_used += 1

    n = totals["n"]
    base_info = {
        "method": method, "drive": drive,
        "pairs": len(pairs), "pred_frames": len(pred_npys_all),
        "gt_frames": len(gt_pngs), "unmatched_preds": unmatched,
        "pairing": "kitti_idx" if is_new_format else "sorted_index_legacy",
    }
    if n == 0:
        return {**base_info, "n_valid_pixels": 0, "frames_used": 0,
                "frames_empty": frames_empty, "metrics": None,
                "note": "no valid pixels"}

    abs_rel = totals["abs_rel_sum"] / n
    sq_rel  = totals["sq_rel_sum"]  / n
    rmse    = float(np.sqrt(totals["rmse_sqsum"] / n))
    rmse_log = float(np.sqrt(totals["rmse_log_sqsum"] / n))
    d1 = totals["d1_sum"] / n
    d2 = totals["d2_sum"] / n
    d3 = totals["d3_sum"] / n

    return {
        **base_info,
        "n_valid_pixels": int(n), "frames_used": frames_used,
        "frames_empty": frames_empty,
        "scales_median": float(np.median(scales)),
        "scales_std": float(np.std(scales)),
        "metrics": {
            "abs_rel": float(abs_rel),
            "sq_rel":  float(sq_rel),
            "rmse":    rmse,
            "rmse_log": rmse_log,
            "delta_1_25":    float(d1),
            "delta_1_25_2":  float(d2),
            "delta_1_25_3":  float(d3),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    res = evaluate(args.method, args.drive, args.pred_dir, args.kitti_root)
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)

    m = res.get("metrics")
    if m:
        print(f"{args.method} {args.drive}: "
              f"AbsRel={m['abs_rel']:.4f}  RMSE={m['rmse']:.4f}  "
              f"δ<1.25={m['delta_1_25']:.4f}  (n={res['n_valid_pixels']})")
    else:
        print(f"{args.method} {args.drive}: NO VALID PIXELS")


if __name__ == "__main__":
    main()
