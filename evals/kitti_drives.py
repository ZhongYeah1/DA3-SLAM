"""KITTI validation-set drive metadata.

The 13 val drives live under:
    <KITTI_ROOT>/<date>/<drive>/image_02/data/*.png
    <KITTI_ROOT>/<date>/<drive>/oxts/

Sparse LiDAR GT depth for eval lives under:
    <KITTI_ROOT>/val/<drive>/proj_depth/groundtruth/image_02/*.png
"""

from __future__ import annotations

KITTI_ROOT = "/nfs/turbo/coe-jungaocv/siyuanb/github/Depth-Anything-3/data/cut3r_data/kitti_backup/kitti"

VAL_DRIVES = [
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0005_sync",
    "2011_09_26_drive_0013_sync",
    "2011_09_26_drive_0020_sync",
    "2011_09_26_drive_0023_sync",
    "2011_09_26_drive_0036_sync",
    "2011_09_26_drive_0079_sync",
    "2011_09_26_drive_0095_sync",
    "2011_09_26_drive_0113_sync",
    "2011_09_28_drive_0037_sync",
    "2011_09_29_drive_0026_sync",
    "2011_09_30_drive_0016_sync",
    "2011_10_03_drive_0047_sync",
]

REPRESENTATIVE_DRIVES = [
    "2011_09_26_drive_0036_sync",
    "2011_10_03_drive_0047_sync",
]

METHODS = ["vggt_slam", "da3_baseline", "da3_refine", "da3_large_refine"]


def date_from_drive(drive: str) -> str:
    """Return the YYYY_MM_DD prefix (first 10 chars) from a drive name."""
    return drive[:10]


def image_folder_for(drive: str, root: str = KITTI_ROOT) -> str:
    return f"{root}/{date_from_drive(drive)}/{drive}/image_02/data"


def oxts_folder_for(drive: str, root: str = KITTI_ROOT) -> str:
    return f"{root}/{date_from_drive(drive)}/{drive}/oxts"


def gt_depth_folder_for(drive: str, root: str = KITTI_ROOT) -> str:
    return f"{root}/val/{drive}/proj_depth/groundtruth/image_02"
