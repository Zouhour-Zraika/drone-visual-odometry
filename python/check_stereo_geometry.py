"""Inspect whether legacy EuRoC stereo pairs behave like rectified images.

This is a diagnostic, not proof of rectification or metric calibration.
Run: python python/check_stereo_geometry.py --root data/euroc/MH_01_easy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from euroc_stereo_loader import load_stereo_index, read_pair


def analyze_pair(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    orb = cv2.ORB_create(nfeatures=3000)
    kpl, desl = orb.detectAndCompute(left, None)
    kpr, desr = orb.detectAndCompute(right, None)
    if desl is None or desr is None:
        raise ValueError("No descriptors found in stereo pair")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(desl, desr, k=2)
    good = [m for pair in matches if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        raise ValueError(f"Too few stereo matches: {len(good)}")

    pts_l = np.float32([kpl[m.queryIdx].pt for m in good])
    pts_r = np.float32([kpr[m.trainIdx].pt for m in good])
    disparities = pts_l[:, 0] - pts_r[:, 0]
    vertical = np.abs(pts_l[:, 1] - pts_r[:, 1])

    # Geometric sanity filter only: for a rectified left/right pair,
    # valid points normally have positive disparity and small vertical error.
    plausible = (disparities > 1.0) & (vertical < 3.0)
    return {
        "matches": float(len(good)),
        "positive_disparity_pct": float(np.mean(disparities > 0) * 100),
        "vertical_median_px": float(np.median(vertical)),
        "vertical_p90_px": float(np.percentile(vertical, 90)),
        "plausible_pct": float(np.mean(plausible) * 100),
        "plausible_count": float(np.count_nonzero(plausible)),
        "median_disparity_px": (
            float(np.median(disparities[plausible]))
            if np.any(plausible) else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/euroc/MH_01_easy"))
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    frames = load_stereo_index(args.root)
    indices = sorted(set(np.linspace(0, len(frames) - 1,
                                    num=min(args.samples, len(frames)),
                                    dtype=int).tolist()))
    print("Frame | Matches | Positive d % | Median |dy| px | P90 |dy| px | Plausible % | Median d px")
    for i in indices:
        frame = frames[i]
        left, right = read_pair(frame)
        try:
            stats = analyze_pair(left, right)
        except ValueError as exc:
            print(f"{frame.index:5d} | diagnostic failed: {exc}")
            continue
        print(
            f"{frame.index:5d} | {int(stats['matches']):7d} | "
            f"{stats['positive_disparity_pct']:12.1f} | "
            f"{stats['vertical_median_px']:14.2f} | "
            f"{stats['vertical_p90_px']:11.2f} | "
            f"{stats['plausible_pct']:11.1f} | "
            f"{stats['median_disparity_px']:11.2f}"
        )
    print("Interpretation: low vertical residuals and predominantly positive disparity")
    print("support, but do not prove, a rectified left/right stereo geometry.")
    print("Metric depth requires confirmed baseline units and camera calibration.")


if __name__ == "__main__":
    main()
