"""4-way comparison plots for the 2 representative KITTI drives.

Rebuilds a minimal `result` dict per method from the cached artifacts
that `kitti_runner.py` already produced (positions.npy, depth_frame_*.npy),
then calls `visualize_all.save_comparison` to emit the side-by-side plots.

No SLAM is re-run -- we reuse the runner's outputs in
    docs/eval/kitti/<method>/<drive>/
and write comparison plots to
    docs/eval/kitti/comparison/four_way_strategies/<drive>/

Usage:
    python evals/visualize_kitti_sequence.py             # both representative drives
    python evals/visualize_kitti_sequence.py <drive>     # one drive
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from evals.kitti_drives import REPRESENTATIVE_DRIVES, METHODS

METHOD_LABEL = {
    "vggt_slam": "VGGT-SLAM",
    "da3_baseline": "DA3 baseline",
    "da3_refine": "DA3 refine",
    "da3_large_refine": "DA3 large-refine",
}


def _load_method_result(method: str, drive: str) -> dict | None:
    """Reconstruct the minimal result dict needed by save_comparison from disk.

    Splits the flat depth stack into per-submap groups using the cached
    `submap_boundaries.npy` so save_comparison sees the true submap count
    and can index specific (submap, frame) positions.
    """
    per_run_dir = PROJECT_ROOT / "docs" / "eval" / "kitti" / method / drive
    pos_path = per_run_dir / "positions.npy"
    if not pos_path.is_file():
        print(f"[skip] missing {pos_path}")
        return None

    positions = np.load(pos_path)
    depth_files = sorted(glob.glob(str(per_run_dir / "depth_frame_*.npy")))
    if not depth_files:
        print(f"[skip] no depth .npy under {per_run_dir}")
        return None
    depths_stack = np.stack([np.load(p) for p in depth_files], axis=0)  # (N, H, W)

    bnd_path = per_run_dir / "submap_boundaries.npy"
    boundaries = np.load(bnd_path) if bnd_path.is_file() else np.array([], dtype=int)

    # `boundaries` holds index-after-last-frame per submap (from save_trajectory).
    # Split depth_stack into N_submaps chunks. Depth count may exceed position count
    # by one overlap frame per submap boundary (pre-fix runs) -- slice defensively.
    N = depths_stack.shape[0]
    splits = [int(b) for b in boundaries if 0 < int(b) < N]
    depths_by_submap = np.split(depths_stack, splits) if splits else [depths_stack]
    depths_by_submap = [g for g in depths_by_submap if g.shape[0] > 0]

    return {
        "positions": positions,
        "submap_boundaries": boundaries,
        "depths": depths_by_submap,
    }


def build_for_drive(drive: str) -> None:
    results = {}
    for method in METHODS:
        r = _load_method_result(method, drive)
        if r is None:
            continue
        results[METHOD_LABEL[method]] = r

    if not results:
        print(f"[{drive}] no methods with cached outputs -- skipping")
        return

    out_dir = PROJECT_ROOT / "docs" / "eval" / "kitti" / "comparison" / "four_way_strategies" / drive
    out_dir.mkdir(parents=True, exist_ok=True)

    # Delayed import: visualize_all pulls torch/matplotlib (matplotlib is fine,
    # torch is imported but unused when we only call save_comparison).
    from visualize_all import save_comparison
    save_comparison(results, str(out_dir))
    print(f"[{drive}] wrote comparison to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", nargs="?", default=None,
                    help="Optional drive name (default: both representatives)")
    args = ap.parse_args()

    drives = [args.drive] if args.drive else REPRESENTATIVE_DRIVES
    for d in drives:
        build_for_drive(d)


if __name__ == "__main__":
    main()
