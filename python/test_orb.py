
import cv2
import numpy as np
from pathlib import Path

# Créer les dossiers
Path("results").mkdir(exist_ok=True)

# Image synthétique avec des détails visuels
rng = np.random.default_rng(42)
image1 = np.zeros((480, 640), dtype=np.uint8)

for _ in range(200):
    x = int(rng.integers(20, 620))
    y = int(rng.integers(20, 460))
    radius = int(rng.integers(3, 12))
    color = int(rng.integers(100, 256))
    cv2.circle(image1, (x, y), radius, color, -1)

# Simuler un déplacement de l'image
M = np.float32([[1, 0, 15], [0, 1, 8]])
image2 = cv2.warpAffine(image1, M, (640, 480))

# Détecter les points ORB
orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)

# Associer les points
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(
    matcher.match(des1, des2),
    key=lambda m: m.distance
)


# Estimer le déplacement avec RANSAC
pts1 = np.float32([
    kp1[m.queryIdx].pt for m in matches
])
pts2 = np.float32([
    kp2[m.trainIdx].pt for m in matches
])

M_est, inliers = cv2.estimateAffinePartial2D(
    pts1,
    pts2,
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0
)

if M_est is None:
    raise RuntimeError("RANSAC n'a pas trouvé de transformation.")

# Garder uniquement les correspondances cohérentes
good_matches = [
    m for m, valid in zip(matches, inliers.ravel())
    if valid
]

# Visualisation après RANSAC
result = cv2.drawMatches(
    image1, kp1,
    image2, kp2,
    good_matches[:80], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite("results/orb_ransac.png", result)

print(f"Correspondances initiales : {len(matches)}")
print(f"Correspondances RANSAC : {len(good_matches)}")
print(f"Translation estimée X : {M_est[0, 2]:.2f} pixels")
print(f"Translation estimée Y : {M_est[1, 2]:.2f} pixels")
print("Résultat : results/orb_ransac.png")

