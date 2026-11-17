
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# Calibration de la caméra virtuelle
K = np.array([
    [450, 0, 320],
    [0, 450, 240],
    [0, 0, 1]
], dtype=np.float64)

# Points fixes dans le monde
points_3d = np.column_stack([
    rng.uniform(-5, 5, 1000),
    rng.uniform(-3, 3, 1000),
    rng.uniform(6, 18, 1000)
])

# Trajectoire réelle : positions des caméras
N = 40
true_positions = np.array([
    [0.12 * i,
     0.15 * np.sin(i / 8),
     0.03 * i]
    for i in range(N)
])

# Toutes les caméras ont la même orientation
R_world_cam = np.eye(3)

def project(position):
    # Transformation monde -> caméra
    t = -R_world_cam @ position
    camera_pts = (
        R_world_cam @ points_3d.T
    ).T + t

    pixels = (K @ camera_pts.T).T
    return pixels[:, :2] / pixels[:, 2:3]

# Générer les observations avec du bruit
observations = [
    project(p) + rng.normal(0, 0.25, (1000, 2))
    for p in true_positions
]

# Pose monde -> caméra estimée
R_cw = np.eye(3)
t_cw = np.zeros((3, 1))

estimated_positions = [np.zeros(3)]
inlier_counts = []

for i in range(1, N):
    pts1 = observations[i - 1]
    pts2 = observations[i]

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        raise RuntimeError(f"Echec image {i}")

    inliers, R_rel, t_rel, _ = cv2.recoverPose(
        E, pts1, pts2, K, mask=mask
    )
    inlier_counts.append(inliers)

    # Echelle connue de la simulation
    # (oracle, indisponible en monoculaire réel)
    scale = np.linalg.norm(
        true_positions[i] - true_positions[i - 1]
    )

    # Composition des poses monde -> caméra
    t_cw = R_rel @ t_cw + scale * t_rel
    R_cw = R_rel @ R_cw

    # Centre de caméra exprimé dans le monde
    camera_center = -R_cw.T @ t_cw
    estimated_positions.append(camera_center.ravel())

estimated_positions = np.array(estimated_positions)

errors = np.linalg.norm(
    estimated_positions - true_positions, axis=1
)
rmse = np.sqrt(np.mean(errors ** 2))

print("=== TRAJECTOIRE SYNTHETIQUE ===")
print(f"Nombre de poses : {N}")
print(f"Inliers moyens : {np.mean(inlier_counts):.1f}")
print(f"RMSE position : {rmse:.4f} m")

# Visualisation
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    true_positions[:, 0],
    true_positions[:, 1],
    true_positions[:, 2],
    "g-", label="Trajectoire réelle", linewidth=2
)

ax.plot(
    estimated_positions[:, 0],
    estimated_positions[:, 1],
    estimated_positions[:, 2],
    "r--", label="Trajectoire estimée", linewidth=2
)

ax.scatter(
    *true_positions[0],
    color="blue", s=60, label="Départ"
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Odométrie visuelle - Simulation 3D")
ax.legend()

plt.tight_layout()
plt.savefig("results/trajectory_3d.png", dpi=180)
plt.close()

np.savez(
    "results/synthetic_trajectory.npz",
    true_positions=true_positions,
    estimated_positions=estimated_positions,
    errors=errors
)

print("Graphique : results/trajectory_3d.png")
