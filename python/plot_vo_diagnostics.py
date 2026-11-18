"""Plot VO diagnostics and identify intervals of poor geometric tracking.

Requires an instrumented stereo_visual_odometry.py output file.
No ground truth is needed for these diagnostic plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vo", type=Path, default=Path("results/euroc_stereo_vo_diagnostics.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    with np.load(args.vo) as data:
        required = (
            "timestamps_s", "poses_world_camera", "pnp_inlier_ratios",
            "reprojection_rmse_px", "processing_time_ms", "success",
            "pnp_correspondence_counts",
        )
        missing = [key for key in required if key not in data]
        if missing:
            parser.error(f"Missing diagnostics {missing}; rerun updated VO")
        t = data["timestamps_s"] - data["timestamps_s"][0]
        poses = data["poses_world_camera"]
        ratios = data["pnp_inlier_ratios"]
        reproj = data["reprojection_rmse_px"]
        timing = data["processing_time_ms"]
        success = data["success"]
        counts = data["pnp_correspondence_counts"]

    if poses.shape != (len(t), 4, 4):
        parser.error("Unexpected pose array shape")
    relative_rotations = np.einsum(
        "nij,njk->nik",
        np.transpose(poses[:-1, :3, :3], (0, 2, 1)),
        poses[1:, :3, :3],
    )
    cosine = np.clip(
        (np.trace(relative_rotations, axis1=1, axis2=2) - 1) / 2, -1, 1
    )
    angular_step_deg = np.degrees(np.arccos(cosine))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(t, counts)
    axes[0].set_ylabel("PnP correspondences")
    axes[1].plot(t, ratios)
    axes[1].set_ylabel("Inlier ratio")
    axes[1].set_ylim(0, 1.05)
    axes[2].plot(t, reproj)
    axes[2].set_ylabel("Reprojection RMSE (px)")
    axes[3].plot(t, timing)
    axes[3].axhline(50, color="red", linestyle="--", label="20 Hz budget (50 ms)")
    axes[3].set_ylabel("Frame time (ms)")
    axes[3].set_xlabel("Time from first frame (s)")
    axes[3].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = args.output_dir / "vo_quality_diagnostics.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t[1:], angular_step_deg)
    ax.set_xlabel("Time from first frame (s)")
    ax.set_ylabel("Relative rotation per frame (degrees)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    rotation_path = args.output_dir / "vo_relative_rotations.png"
    fig.savefig(rotation_path, dpi=170)
    plt.close(fig)

    finite = np.isfinite(reproj) & success
    print(f"Frames: {len(t)}; PnP failures: {np.count_nonzero(~success[1:])}")
    if np.any(finite):
        print(f"Reprojection RMSE median/p90: "
              f"{np.median(reproj[finite]):.3f}/"
              f"{np.percentile(reproj[finite], 90):.3f} px")
    print(f"Frame processing median/p90: "
          f"{np.nanmedian(timing):.2f}/"
          f"{np.nanpercentile(timing, 90):.2f} ms")
    print(f"Frames exceeding 50 ms: {np.count_nonzero(timing[1:] > 50)}")
    print(f"Maximum relative rotation step: {np.max(angular_step_deg):.3f} deg")
    print(f"Saved: {path}")
    print(f"Saved: {rotation_path}")
    print("Rotation is camera-relative, not ground-truth rotation error.")


if __name__ == "__main__":
    main()
