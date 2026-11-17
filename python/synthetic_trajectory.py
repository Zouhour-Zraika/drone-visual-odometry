"""Synthetic monocular VO benchmark.

The ground-truth step length is an *oracle* used only to study pose-estimation
error. Monocular essential-matrix geometry cannot recover metric scale.
Run from the repository root: python python/synthetic_trajectory.py
"""
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
K = np.array([[450., 0., 320.], [0., 450., 240.], [0., 0., 1.]])
N = 40
points_world = np.column_stack((
    rng.uniform(-5, 5, 1000),
    rng.uniform(-3, 3, 1000),
    rng.uniform(6, 18, 1000),
))
# A larger baseline avoids near-zero parallax and unstable cheirality checks.
# This is a synthetic test condition, not a claim of real drone performance.
truth = np.array([[.4*i, .15*np.sin(i/8), .03*i] for i in range(N)])

def project(center):
    camera_points = points_world - center
    uvw = (K @ camera_points.T).T
    return uvw[:, :2] / uvw[:, 2:]

observations = [project(c) + rng.normal(0, .25, (len(points_world), 2))
                for c in truth]

# T_cw maps world coordinates into camera coordinates.
R_cw = np.eye(3)
t_cw = np.zeros((3, 1))
estimated = [np.zeros(3)]
inliers_per_step = []
rotation_errors_deg = []
translation_direction_errors_deg = []

for i in range(1, N):
    p1, p2 = observations[i-1], observations[i]
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC,
                                   prob=.999, threshold=1.)
    if E is None:
        raise RuntimeError(f"Essential matrix estimation failed at frame {i}")
    count, R_rel, t_rel, pose_mask = cv2.recoverPose(E, p1, p2, K, mask=mask)
    if count < 8:
        raise RuntimeError(f"Insufficient pose inliers at frame {i}: {count}")
    inliers_per_step.append(int(count))

    # The cameras have identical true orientation: R_relative_true = I.
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    rotation_errors_deg.append(float(np.degrees(angle)))

    # With identity true rotations, relative camera translation is -delta_C.
    true_step = truth[i] - truth[i-1]
    true_camera_translation_direction = -true_step / np.linalg.norm(true_step)
    estimated_direction = t_rel.ravel() / np.linalg.norm(t_rel)
    direction_angle = np.arccos(np.clip(
        estimated_direction @ true_camera_translation_direction, -1, 1))
    translation_direction_errors_deg.append(float(np.degrees(direction_angle)))

    # ORACLE SCALE: supplied by simulated ground truth, not recovered by VO.
    scale = np.linalg.norm(true_step)
    t_cw = R_rel @ t_cw + scale * t_rel
    R_cw = R_rel @ R_cw
    estimated.append((-R_cw.T @ t_cw).ravel())

estimated = np.asarray(estimated)
errors = np.linalg.norm(estimated - truth, axis=1)
rmse = float(np.sqrt(np.mean(errors**2)))
print("=== SYNTHETIC MONOCULAR VO (ORACLE SCALE) ===")
print(f"Poses: {N}")
print(f"Mean pose inliers: {np.mean(inliers_per_step):.1f}")
print(f"Position RMSE: {rmse:.4f} m")
print(f"Mean relative rotation error: {np.mean(rotation_errors_deg):.3f} deg")
print(f"Mean translation-direction error: {np.mean(translation_direction_errors_deg):.3f} deg")
print("WARNING: metric step lengths come from ground truth; this is not metric monocular VO.")

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(*truth.T, "g-", label="Ground truth", linewidth=2)
ax.plot(*estimated.T, "r--", label="Estimated (oracle scale)", linewidth=2)
ax.scatter(*truth[0], color="blue", s=60, label="Start")
ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
       title="Synthetic monocular VO — oracle-scale evaluation")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "trajectory_3d.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.arange(N), errors, label="Position error")
ax.set(xlabel="Frame", ylabel="Euclidean position error (m)",
       title=f"Accumulated drift (RMSE = {rmse:.3f} m)")
ax.grid(alpha=.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "position_error.png", dpi=180)
plt.close(fig)

np.savez(OUT / "synthetic_trajectory.npz", true_positions=truth,
         estimated_positions=estimated, errors=errors,
         inlier_counts=np.asarray(inliers_per_step),
         rotation_errors_deg=np.asarray(rotation_errors_deg),
         translation_direction_errors_deg=np.asarray(translation_direction_errors_deg))
print(f"Plots: {OUT / 'trajectory_3d.png'} and {OUT / 'position_error.png'}")
