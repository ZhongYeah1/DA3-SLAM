"""Pluggable scale estimation between consecutive submaps.

Three methods:
  - median:         current approach -- median of pairwise distance ratios
  - depth-ransac:   RANSAC linear fit on overlapping depth pixels
  - depth-weighted: confidence-weighted quantile regression
"""

import numpy as np


def estimate_scale_median(
    points_curr: np.ndarray,
    points_prev: np.ndarray,
) -> tuple[float, float]:
    """Median of pairwise distance ratios (current method).

    Args:
        points_curr: (M, 3) rotated current-submap points.
        points_prev: (M, 3) previous-submap points.

    Returns:
        (scale_factor, quality_score) where quality_score is always 1.0.
    """
    x_dists = np.linalg.norm(points_curr, axis=1)
    y_dists = np.linalg.norm(points_prev, axis=1)
    valid = x_dists > 1e-8
    if valid.sum() < 10:
        return 1.0, 0.0
    scales = y_dists[valid] / x_dists[valid]
    return float(np.median(scales)), 1.0


def estimate_scale_depth_ransac(
    depth_prev: np.ndarray,
    conf_prev: np.ndarray,
    depth_curr: np.ndarray,
    conf_curr: np.ndarray,
    conf_threshold_pct: float = 25.0,
    n_iter: int = 200,
    inlier_threshold: float = 0.1,
) -> tuple[float, float]:
    """RANSAC linear fit: depth_prev = scale * depth_curr.

    Operates on the raw depth maps of the overlapping frame(s),
    using only high-confidence pixels.

    Args:
        depth_prev: (H, W) depth from previous submap's overlap frame.
        conf_prev:  (H, W) confidence from previous submap.
        depth_curr: (H, W) depth from current submap's overlap frame.
        conf_curr:  (H, W) confidence from current submap.
        conf_threshold_pct: percentile threshold for confidence filtering.
        n_iter: number of RANSAC iterations.
        inlier_threshold: relative error threshold for inliers.

    Returns:
        (scale_factor, quality_score) where quality_score is inlier ratio.
    """
    conf_thresh_prev = np.percentile(conf_prev, conf_threshold_pct)
    conf_thresh_curr = np.percentile(conf_curr, conf_threshold_pct)
    mask = (conf_prev >= conf_thresh_prev) & (conf_curr >= conf_thresh_curr)
    d_prev = depth_prev[mask].flatten()
    d_curr = depth_curr[mask].flatten()

    if len(d_prev) < 50:
        return 1.0, 0.0

    # Subsample for speed
    if len(d_prev) > 10000:
        idx = np.random.choice(len(d_prev), 10000, replace=False)
        d_prev = d_prev[idx]
        d_curr = d_curr[idx]

    best_scale = 1.0
    best_inlier_ratio = 0.0

    for _ in range(n_iter):
        # Random sample: pick one point, compute scale
        i = np.random.randint(len(d_prev))
        if abs(d_curr[i]) < 1e-8:
            continue
        s = d_prev[i] / d_curr[i]
        if s < 0.1 or s > 10.0:
            continue

        # Count inliers
        residuals = np.abs(d_prev - s * d_curr) / (d_prev + 1e-8)
        inlier_ratio = float((residuals < inlier_threshold).mean())

        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_scale = s

    # Refine on inliers
    residuals = np.abs(d_prev - best_scale * d_curr) / (d_prev + 1e-8)
    inlier_mask = residuals < inlier_threshold
    if inlier_mask.sum() > 10:
        d_curr_safe = np.clip(d_curr[inlier_mask], 1e-8, None)
        best_scale = float(np.median(d_prev[inlier_mask] / d_curr_safe))

    return best_scale, best_inlier_ratio


def estimate_scale_depth_weighted(
    depth_prev: np.ndarray,
    conf_prev: np.ndarray,
    depth_curr: np.ndarray,
    conf_curr: np.ndarray,
    conf_threshold_pct: float = 25.0,
) -> tuple[float, float]:
    """Confidence-weighted quantile regression.

    Args:
        depth_prev: (H, W) depth from previous submap's overlap frame.
        conf_prev:  (H, W) confidence from previous submap.
        depth_curr: (H, W) depth from current submap's overlap frame.
        conf_curr:  (H, W) confidence from current submap.
        conf_threshold_pct: percentile threshold for confidence filtering.

    Returns:
        (scale_factor, quality_score) based on weighted median.
    """
    conf_thresh_prev = np.percentile(conf_prev, conf_threshold_pct)
    conf_thresh_curr = np.percentile(conf_curr, conf_threshold_pct)
    mask = (conf_prev >= conf_thresh_prev) & (conf_curr >= conf_thresh_curr)
    d_prev = depth_prev[mask].flatten()
    d_curr = depth_curr[mask].flatten()

    if len(d_prev) < 50:
        return 1.0, 0.0

    # Weights: (conf_prev * conf_curr)^2
    w = (conf_prev[mask].flatten() * conf_curr[mask].flatten()) ** 2
    ratios = d_prev / np.clip(d_curr, 1e-8, None)

    # Weighted 90th percentile (robust to outliers at the high end)
    sorted_idx = np.argsort(ratios)
    ratios_sorted = ratios[sorted_idx]
    w_sorted = w[sorted_idx]
    cumw = np.cumsum(w_sorted)
    cumw = cumw / cumw[-1]

    # Weighted median
    median_idx = np.searchsorted(cumw, 0.5)
    scale = float(ratios_sorted[min(median_idx, len(ratios_sorted) - 1)])

    # Quality: entropy-based -- low spread = high quality
    q25_idx = np.searchsorted(cumw, 0.25)
    q75_idx = np.searchsorted(cumw, 0.75)
    iqr = ratios_sorted[min(q75_idx, len(ratios_sorted) - 1)] - ratios_sorted[min(q25_idx, len(ratios_sorted) - 1)]
    quality = max(0.0, 1.0 - iqr / (abs(scale) + 1e-8))

    return scale, quality


# Default scale method per strategy
STRATEGY_SCALE_DEFAULTS = {
    "baseline": "median",
    "refine": "median",
    "large-refine": "median",
}
