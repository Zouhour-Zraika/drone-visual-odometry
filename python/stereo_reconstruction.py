"""Sparse stereo reconstruction for the legacy MH_01_easy export.

Assumes images are rectified, horizontal positive disparity, and a baseline
of 0.110078 *meters*. This baseline/unit assumption must be verified against
the dataset provenance before treating the output as metric ground truth.
No ground-truth poses are used.

Usage:
 python python/stereo_reconstruction.py --frame 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from euroc_stereo_loader import load_stereo_index, read_pair

FX = FY = 435.205
CX, CY = 367.452, 252.201
BASELINE_M = 0.110078


def triangulate_rectified(
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
    *,
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
    baseline_m: float = BASELINE_M,
    min_disparity_px: float = 1.0,
    max_vertical_error_px: float = 2.0,
    min_depth_m: float = 0.3,
    max_depth_m: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 3D points in the left camera frame and retained input indices."""
    left_pixels = np.asarray(left_pixels, dtype=np.float64)
    right_pixels = np.asarray(right_pixels, dtype=np.float64)
    if (left_pixels.ndim != 2 or left_pixels.shape[1] != 2
            or right_pixels.shape != left_pixels.shape):
        raise ValueError("Expected corresponding Nx2 pixel arrays")
    if fx <= 0 or fy <= 0 or baseline_m <= 0:
        raise ValueError("Focal lengths and baseline must be positive")
    if min_disparity_px <= 0 or min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ValueError("Invalid disparity or depth limits")

    disparity = left_pixels[:, 0] - right_pixels[:, 0]
    dy = np.abs(left_pixels[:, 1] - right_pixels[:, 1])
    valid = (
        np.isfinite(left_pixels).all(axis=1)
        & np.isfinite(right_pixels).all(axis=1)
        & (disparity >= min_disparity_px)
        & (dy <= max_vertical_error_px)
    )
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float64), indices

    z = fx * baseline_m / disparity[indices]
    depth_valid = (z >= min_depth_m) & (z <= max_depth_m)
    indices = indices[depth_valid]
    z = z[depth_valid]
    u, v = left_pixels[indices].T
    xyz = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return xyz, indices


def match_stereo(left: np.ndarray, right: np.ndarray, nfeatures: int = 3000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    key_l, des_l = orb.detectAndCompute(left, None)
    key_r, des_r = orb.detectAndCompute(right, None)
    if des_l is None or des_r is None:
        raise ValueError("No ORB descriptors in stereo pair")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_l, des_r, k=2)
    matches = [m for pair in pairs if len(pair) == 2
               for m, n in [pair] if m.distance < 0.75 * n.distance]
    if not matches:
        raise ValueError("No stereo matches after ratio test")
    left_pts = np.float64([key_l[m.queryIdx].pt for m in matches])
    right_pts = np.float64([key_r[m.trainIdx].pt for m in matches])
    return left_pts, right_pts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/euroc/MH_01_easy"))
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/euroc_stereo_points.npz"))
    parser.add_argument("--baseline-m", type=float, default=BASELINE_M,
                        help="Assumed baseline in meters; verify independently")
    args = parser.parse_args()

    frames = load_stereo_index(args.root)
    by_index = {f.index: f for f in frames}
    if args.frame not in by_index:
        parser.error(f"Frame {args.frame} not found")
    frame = by_index[args.frame]
    left, right = read_pair(frame)
    left_pts, right_pts = match_stereo(left, right)
    xyz, retained = triangulate_rectified(
        left_pts, right_pts, baseline_m=args.baseline_m
    )
    if xyz.shape[0] < 10:
        raise RuntimeError(f"Only {xyz.shape[0]} valid 3D points; check calibration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_xyz_left_m=xyz,
        pixels_left=left_pts[retained],
        pixels_right=right_pts[retained],
        frame_index=frame.index,
        timestamp_s=frame.timestamp_s,
        K=np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]),
        baseline_m=args.baseline_m,
    )
    print(f"Frame: {frame.index}")
    print(f"ORB stereo matches: {len(left_pts)}")
    print(f"Valid 3D points: {len(xyz)}")
    print(f"Median depth (assumed meters): {np.median(xyz[:, 2]):.3f}")
    print(f"Depth p10/p90: {np.percentile(xyz[:, 2], [10, 90])}")
    print(f"Saved: {args.output}")
    print("WARNING: metric units depend on unverified baseline interpretation.")


if __name__ == "__main__":
    main()
