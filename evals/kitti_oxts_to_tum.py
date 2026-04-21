"""Convert KITTI OXTS (GPS+IMU) data to a TUM-format trajectory file.

KITTI OXTS records hold lat/lon/alt (WGS84) and roll/pitch/yaw in the IMU
frame. The standard KITTI → pose conversion uses a Mercator projection
anchored at the first frame's latitude, plus the Z-Y-X Euler convention
for rotation.

Output is a TUM file with one row per frame:
    timestamp tx ty tz qx qy qz qw

The first frame is anchored to identity so all trajectories share a
common origin -- matching the convention used by SLAM estimates from
DA3-SLAM / VGGT-SLAM.

CLI:
    python evals/kitti_oxts_to_tum.py <drive_dir> <output.tum>

where <drive_dir> is the directory containing `oxts/data/*.txt` and
`oxts/timestamps.txt` (e.g. .../2011_09_26/2011_09_26_drive_0002_sync).
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np


EARTH_RADIUS = 6378137.0  # WGS84 equatorial radius, metres


def _mercator_scale(lat_deg: float) -> float:
    return np.cos(lat_deg * np.pi / 180.0)


def _latlon_to_xy(lat_deg: float, lon_deg: float, scale: float) -> tuple[float, float]:
    """Spherical Mercator projection centred at (0,0)."""
    x = scale * EARTH_RADIUS * (lon_deg * np.pi / 180.0)
    y = scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat_deg) * np.pi / 360.0))
    return x, y


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """KITTI R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (Z-Y-X intrinsic convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (qx, qy, qz, qw) using shepperd/stable form."""
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


def _parse_timestamp_line(line: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS.ffffff' (9-digit fraction) → seconds since epoch."""
    line = line.strip()
    # Python's datetime supports up to microseconds (6 digits). KITTI has nanoseconds (9).
    # Truncate fractional part to 6 digits.
    if "." in line:
        head, frac = line.split(".", 1)
        frac = frac[:6]
        line = f"{head}.{frac}"
    dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


def _read_oxts_record(path: str) -> tuple[float, float, float, float, float, float]:
    """Return (lat, lon, alt, roll, pitch, yaw) from one oxts .txt file."""
    with open(path, "r") as f:
        vals = f.read().split()
    lat, lon, alt = float(vals[0]), float(vals[1]), float(vals[2])
    roll, pitch, yaw = float(vals[3]), float(vals[4]), float(vals[5])
    return lat, lon, alt, roll, pitch, yaw


def convert(drive_dir: str, output_path: str, timestamp_mode: str = "frame_index") -> int:
    """Convert oxts records under drive_dir into TUM file at output_path.

    timestamp_mode:
        "frame_index" -- timestamp = i * 0.1 (aligns with SLAM keyframes via
                        index-based matching). Default.
        "real_epoch"  -- real KITTI epoch timestamps from timestamps.txt.

    Returns number of poses written.
    """
    oxts_dir = os.path.join(drive_dir, "oxts")
    data_dir = os.path.join(oxts_dir, "data")
    ts_path = os.path.join(oxts_dir, "timestamps.txt")

    oxts_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not oxts_files:
        raise FileNotFoundError(f"No oxts .txt files under {data_dir}")

    if timestamp_mode == "real_epoch":
        with open(ts_path, "r") as f:
            timestamps = [_parse_timestamp_line(l) for l in f.readlines() if l.strip()]
        if len(timestamps) != len(oxts_files):
            raise ValueError(
                f"oxts count ({len(oxts_files)}) != timestamp count ({len(timestamps)}) in {drive_dir}"
            )
    elif timestamp_mode == "frame_index":
        # Extract integer index from filename (e.g. "0000000042.txt" → 42)
        timestamps = [int(os.path.splitext(os.path.basename(p))[0]) * 0.1 for p in oxts_files]
    else:
        raise ValueError(f"Unknown timestamp_mode: {timestamp_mode}")

    # First frame: derive scale + anchor
    lat0, lon0, alt0, roll0, pitch0, yaw0 = _read_oxts_record(oxts_files[0])
    scale = _mercator_scale(lat0)
    x0, y0 = _latlon_to_xy(lat0, lon0, scale)
    R0 = _rpy_to_matrix(roll0, pitch0, yaw0)
    t0 = np.array([x0, y0, alt0])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n_written = 0
    with open(output_path, "w") as fout:
        for i, oxts_path in enumerate(oxts_files):
            lat, lon, alt, roll, pitch, yaw = _read_oxts_record(oxts_path)
            x, y = _latlon_to_xy(lat, lon, scale)
            R_abs = _rpy_to_matrix(roll, pitch, yaw)
            t_abs = np.array([x, y, alt])

            # Anchor so frame 0 is identity: T_rel = T_0^{-1} @ T_abs.
            R_rel = R0.T @ R_abs
            t_rel = R0.T @ (t_abs - t0)
            q = _rotmat_to_quat_xyzw(R_rel)
            ts = timestamps[i]
            fout.write(
                f"{ts:.9f} {t_rel[0]:.9f} {t_rel[1]:.9f} {t_rel[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )
            n_written += 1
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("drive_dir", help="Path containing oxts/ subdir (e.g. .../<drive>)")
    ap.add_argument("output_path", help="Output TUM file")
    ap.add_argument("--timestamp_mode", choices=["frame_index", "real_epoch"],
                    default="frame_index",
                    help="frame_index: ts=i*0.1 (matches SLAM keyframe idx); real_epoch: epoch seconds")
    args = ap.parse_args()
    n = convert(args.drive_dir, args.output_path, timestamp_mode=args.timestamp_mode)
    print(f"Wrote {n} poses to {args.output_path}")


if __name__ == "__main__":
    main()
