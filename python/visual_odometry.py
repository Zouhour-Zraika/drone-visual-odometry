
import cv2
import numpy as np
from pathlib import Path

Path("results").mkdir(exist_ok=True)

# Paramètres d'une caméra virtuelle
K = np.array([
    [450, 0, 320],
    [0, 450, 240],
    [0, 0, 1]
], dtype=np.float64)

# Générer des points 3D devant la caméra
rng = np.random.default_rng(42)
points_3d = np.column_stack([
    rng.uniform(-3, 3, 300),
    rng.uniform(-2, 2, 300),
    rng.uniform(4, 12, 300)
])

# Première caméra : origine
R1 = np.eye(3)
t1 = np.zeros((3, 1))

# Deuxième caméra : mouvement connu
angle = np.deg2rad(5)
R2, _ = cv2.Rodrigues(
    np.array([0.0, angle, 0.0])
)
t2 = np.array([[0.5], [0.0], [0.1]])

def project(points, R, t):
    camera_points = (R @ points.T + t).T
    pixels = (K @ camera_points.T).T
    return pixels[:, :2] / pixels[:, 2:3]

pts1 = project(points_3d, R1, t1)
pts2 = project(points_3d, R2, t2)

# Ajouter un bruit de mesure
pts1 += rng.normal(0, 0.3, pts1.shape)
pts2 += rng.normal(0, 0.3, pts2.shape)

# Estimer la matrice essentielle avec RANSAC
E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None:
    raise RuntimeError("Estimation de E impossible")

# Récupérer la rotation et la translation
inliers, R_est, t_est, pose_mask = cv2.recoverPose(
    E, pts1, pts2, K, mask=mask
)

# Comparer les rotations
R_error = R_est @ R2.T
cos_angle = np.clip(
    (np.trace(R_error) - 1) / 2,
    -1.0, 1.0
)
rotation_error = np.rad2deg(np.arccos(cos_angle))

# Comparer les directions de translation
t_true = t2 / np.linalg.norm(t2)
cos_t = np.clip(
    float(t_est.ravel() @ t_true.ravel()),
    -1.0, 1.0
)
translation_error = np.rad2deg(np.arccos(cos_t))

print("=== VISUAL ODOMETRY 3D ===")
print(f"Points 3D : {len(points_3d)}")
print(f"Inliers pose : {inliers}")
print(f"Erreur rotation : {rotation_error:.3f} deg")
print(f"Erreur direction translation : {translation_error:.3f} deg")
print("Rotation estimée :")
print(R_est)
print("Direction translation estimée :")
print(t_est.ravel())

np.savez(
    "results/pose_estimation.npz",
    R=R_est,
    t=t_est,
    E=E
)

print("Résultat : results/pose_estimation.npz")
