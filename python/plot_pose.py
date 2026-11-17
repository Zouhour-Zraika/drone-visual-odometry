
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)

# Charger la pose estimée
data = np.load("results/pose_estimation.npz")
R_est = data["R"]
t_est = data["t"].ravel()

# Translation réelle de notre simulation
t_true = np.array([0.5, 0.0, 0.1])

# Normalisation : la VO monoculaire
# ne connaît pas l'échelle absolue
t_true = t_true / np.linalg.norm(t_true)
t_est = t_est / np.linalg.norm(t_est)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

origin = np.zeros(3)

ax.quiver(
    *origin, *t_true,
    color="green",
    label="Direction réelle",
    linewidth=3
)

ax.quiver(
    *origin, *t_est,
    color="red",
    label="Direction estimée",
    linewidth=3
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Comparaison des directions de translation")
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.2, 0.5)
ax.set_box_aspect((1.4, 1, 0.7))
ax.legend()

plt.savefig("results/pose_3d.png", dpi=180)
plt.close()

print("Visualisation enregistrée : results/pose_3d.png")
