"""Evaluate stereo VO positions against EuRoC state ground truth.

Uses timestamp interpolation and rigid SE(3) position alignment, NO scale fitting.
This is a *position-only* proxy evaluation: camera-to-body lever arm and rotation
are not calibrated, so results are not rigorous camera-pose ATE/RPE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_ground_truth(path: Path):
    data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError("Expected EuRoC GT CSV with timestamp, position, quaternion")
    t = data[:, 0] * 1e-9
    p = data[:, 1:4]
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
        raise ValueError("Nonfinite GT timestamps or positions")
    if not np.all(np.diff(t) > 0):
        raise ValueError("GT timestamps must increase strictly")
    return t, p


def interpolate_positions(query_t, gt_t, gt_p):
    query_t = np.asarray(query_t, dtype=np.float64)
    if np.any(query_t < gt_t[0]) or np.any(query_t > gt_t[-1]):
        raise ValueError("Queries must be within ground-truth time interval")
    return np.column_stack([
        np.interp(query_t, gt_t, gt_p[:, axis]) for axis in range(3)
    ])


def align_rigid(source, target):
    """Fit R,t minimizing ||source @ R.T + t - target||, with det(R)=+1."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Expected equal Nx3 position arrays")
    if len(source) < 3:
        raise ValueError("Need at least 3 aligned poses")
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = (u @ correction @ vt).T
    translation = b - rotation @ a
    return rotation, translation


def evaluate(vo_t, vo_p, gt_t, gt_p, success=None):
    vo_t = np.asarray(vo_t, dtype=np.float64)
    vo_p = np.asarray(vo_p, dtype=np.float64)
    if vo_p.shape != (len(vo_t), 3):
        raise ValueError("VO timestamps/positions shape mismatch")
    if not np.all(np.diff(vo_t) > 0):
        raise ValueError("VO timestamps must increase")
    overlap = (vo_t >= gt_t[0]) & (vo_t <= gt_t[-1])
    if success is not None:
        success = np.asarray(success, dtype=bool)
        if success.shape != overlap.shape:
            raise ValueError("Success flags shape mismatch")
        overlap &= success
    if np.count_nonzero(overlap) < 3:
        raise ValueError("Too few overlapping valid VO poses")
    timestamps = vo_t[overlap]
    estimated = vo_p[overlap]
    reference = interpolate_positions(timestamps, gt_t, gt_p)
    rotation, translation = align_rigid(estimated, reference)
    aligned = estimated @ rotation.T + translation
    errors = np.linalg.norm(aligned-reference, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors**2)))
    # One-step relative *translation-vector* residual in a single aligned
    # world frame. Not full SE(3) RPE (requires calibrated orientations).
    delta_est = np.diff(aligned, axis=0)
    delta_ref = np.diff(reference, axis=0)
    rpe = np.linalg.norm(delta_est-delta_ref, axis=1)
    rpe_rmse = float(np.sqrt(np.mean(rpe**2)))
    return {
        "timestamps_s": timestamps,
        "aligned_positions": aligned,
        "reference_positions": reference,
        "ate_errors_m": errors,
        "rpe_translation_errors_m": rpe,
        "ate_rmse_m": ate_rmse,
        "rpe_translation_rmse_m": rpe_rmse,
        "rotation": rotation,
        "translation": translation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vo", type=Path, default=Path("results/euroc_stereo_vo.npz"))
    parser.add_argument("--gt", type=Path, default=Path(
        "data/euroc/MH_01_easy/state_groundtruth_estimate.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/euroc_vo_evaluation.npz"))
    args = parser.parse_args()
    gt_t, gt_p = load_ground_truth(args.gt)
    with np.load(args.vo) as vo:
        timestamps = vo["timestamps_s"]
        positions = vo["positions"]
        success = vo["success"] if "success" in vo else None
    result = evaluate(timestamps, positions, gt_t, gt_p, success)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Matched valid poses: {len(result['timestamps_s'])}")
    print(f"Overlap duration: {np.ptp(result['timestamps_s']):.3f} s")
    print(f"ATE position RMSE (rigid alignment, no scale): {result['ate_rmse_m']:.4f} m")
    print(f"1-step aligned-world translation residual RMSE: "
          f"{result['rpe_translation_rmse_m']:.4f} m")
    print(f"Saved: {args.output}")
    print("LIMITATION: camera-to-body extrinsics not calibrated; "
          "these are position-only proxy metrics, not full SE(3) trajectory errors.")


if __name__ == "__main__":
    main()
