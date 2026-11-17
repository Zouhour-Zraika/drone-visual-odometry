# Drone visual odometry

Educational visual-odometry experiments using Python, OpenCV and synthetic imagery. The longer-term goal is evaluation on the **EuRoC MAV** visual-inertial dataset.

## Setup (Ubuntu / WSL)

```bash
cd ~/drone-visual-odometry
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Synthetic benchmark

```bash
python python/test_orb.py
python python/visual_odometry.py
python python/plot_pose.py
python python/synthetic_trajectory.py
```

Outputs are written to `results/` (excluded from Git). The trajectory script saves `trajectory_3d.png`, `position_error.png`, and a NumPy archive containing pose and error diagnostics.

**Important limitation:** the synthetic trajectory uses the *ground-truth step length* to scale each translation direction returned by `recoverPose`. This is explicitly an **oracle-scale evaluation**, not a deployable metric monocular odometry system. Relative pose errors compound across frames. For real-world metric scale, use an additional source such as stereo, inertial sensing with an observable scale, or known geometry.

## Next milestones

1. Run and validate the synthetic benchmark; inspect pose and drift metrics.
2. Parse a single EuRoC MAV sequence and its camera calibration.
3. Implement image feature tracking and camera-only relative-pose estimation on real images.
4. Evaluate against timestamp-aligned ground truth, handling coordinate frames and scale explicitly.
5. Add stereo/IMU integration as a separate, validated stage.

EuRoC reference: https://doi.org/10.3929/ethz-b-000690084
