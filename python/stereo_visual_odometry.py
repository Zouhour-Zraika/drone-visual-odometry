"""Experimental stereo visual odometry: ORB stereo depth + temporal PnP RANSAC.

Camera poses are reported in the first *left-camera* frame, not drone/body frame.
Metric interpretation requires verified baseline units and camera calibration.
No ground truth is used for motion estimation.

Usage: python python/stereo_visual_odometry.py --max-frames 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from euroc_stereo_loader import load_stereo_index, read_pair
from stereo_reconstruction import FX, FY, CX, CY, BASELINE_M, triangulate_rectified

K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


def ratio_matches(des_a, des_b, ratio=0.75):
    if des_a is None or des_b is None or len(des_b) < 2:
        return []
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    return [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]


def stereo_landmarks(left, right, orb, baseline_m):
    """3D landmarks keyed by descriptor indices of the left image."""
    kp_l, des_l = orb.detectAndCompute(left, None)
    kp_r, des_r = orb.detectAndCompute(right, None)
    if des_l is None or des_r is None:
        return kp_l, des_l, {}
    matches = ratio_matches(des_l, des_r)
    if not matches:
        return kp_l, des_l, {}
    pts_l = np.array([kp_l[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts_r = np.array([kp_r[m.trainIdx].pt for m in matches], dtype=np.float64)
    xyz, kept = triangulate_rectified(pts_l, pts_r, baseline_m=baseline_m)
    landmarks = {matches[i].queryIdx: point for i, point in zip(kept, xyz)}
    return kp_l, des_l, landmarks


def estimate_relative_pose(points_3d, pixels_2d, *, min_inliers=15):
    """Estimate T_current_previous from 3D previous-camera points to current pixels."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    pixels_2d = np.asarray(pixels_2d, dtype=np.float64)
    if len(points_3d) < 6:
        return None, 0
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, pixels_2d, K, None,
        iterationsCount=200, reprojectionError=2.5,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    count = 0 if inliers is None else len(inliers)
    if not success or count < min_inliers:
        return None, count
    # Refine only using RANSAC inliers; do not let outliers bias the final pose.
    chosen = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(
        points_3d[chosen], pixels_2d[chosen], K, None, rvec, tvec
    )
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.ravel()
    return transform, count


def run(root, max_frames, baseline_m, min_inliers):
    frames = load_stereo_index(root)
    if max_frames is not None:
        frames = frames[:max_frames]
    if len(frames) < 2:
        raise ValueError("Need at least two frames")

    orb = cv2.ORB_create(nfeatures=3000)
    trajectory = [np.zeros(3)]
    frame_ids = [frames[0].index]
    timestamps = [frames[0].timestamp_s]
    inlier_counts = [0]
    success_flags = [True]
    T_world_previous = np.eye(4)
    left, right = read_pair(frames[0])
    prev_kp, prev_des, prev_landmarks = stereo_landmarks(left, right, orb, baseline_m)
    failures = 0
    start = perf_counter()

    for frame in frames[1:]:
        left, right = read_pair(frame)
        kp, des, landmarks = stereo_landmarks(left, right, orb, baseline_m)
        temporal = ratio_matches(prev_des, des)
        linked = [(prev_landmarks[m.queryIdx], kp[m.trainIdx].pt)
                  for m in temporal if m.queryIdx in prev_landmarks]
        transform, count = (estimate_relative_pose(
            np.asarray([p for p, _ in linked]),
            np.asarray([uv for _, uv in linked]),
            min_inliers=min_inliers,
        ) if len(linked) >= 6 else (None, 0))

        if transform is None:
            # Explicitly mark failure and keep the last valid pose;
            # never silently report a zero-motion estimate as a success.
            failures += 1
            success = False
        else:
            T_world_previous = T_world_previous @ np.linalg.inv(transform)
            success = True

        trajectory.append(T_world_previous[:3, 3].copy())
        frame_ids.append(frame.index)
        timestamps.append(frame.timestamp_s)
        inlier_counts.append(count)
        success_flags.append(success)
        prev_kp, prev_des, prev_landmarks = kp, des, landmarks

        if len(trajectory) % 20 == 0:
            print(f"Frame {frame.index}: PnP inliers={count}, success={success}")

    elapsed = perf_counter() - start
    return {
        "positions": np.asarray(trajectory),
        "frame_ids": np.asarray(frame_ids),
        "timestamps_s": np.asarray(timestamps),
        "inlier_counts": np.asarray(inlier_counts),
        "success": np.asarray(success_flags),
        "failures": failures,
        "elapsed_s": elapsed,
        "baseline_m": baseline_m,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/euroc/MH_01_easy"))
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--baseline-m", type=float, default=BASELINE_M)
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("results/euroc_stereo_vo.npz"))
    args = parser.parse_args()
    if args.max_frames < 2 or args.baseline_m <= 0 or args.min_inliers < 6:
        parser.error("Invalid frame count, baseline, or minimum inliers")
    result = run(args.root, args.max_frames, args.baseline_m, args.min_inliers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    n = len(result["frame_ids"])
    print(f"Frames processed: {n}")
    print(f"PnP failures: {result['failures']} / {n-1}")
    print(f"Mean successful PnP inliers: "
          f"{np.mean(result['inlier_counts'][1:][result['success'][1:]]):.1f}"
          if np.any(result["success"][1:]) else "No successful PnP estimates")
    print(f"Elapsed: {result['elapsed_s']:.2f} s; "
          f"throughput: {(n-1)/result['elapsed_s']:.2f} frame pairs/s")
    print(f"Final camera position relative to first frame: {result['positions'][-1]}")
    print(f"Saved: {args.output}")
    print("WARNING: not ground-truth aligned; baseline units not independently verified.")


if __name__ == "__main__":
    main()
