"""Synthetic geometric regression test for the PnP component."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from stereo_visual_odometry import K, estimate_relative_pose


class PnPTests(unittest.TestCase):
    def test_recovers_known_relative_transform(self):
        rng = np.random.default_rng(23)
        points = np.column_stack([
            rng.uniform(-2, 2, 120),
            rng.uniform(-1.5, 1.5, 120),
            rng.uniform(3, 12, 120),
        ]).astype(np.float64)
        rvec = np.array([0.01, -0.015, 0.02])
        tvec = np.array([0.08, -0.03, 0.04])
        pixels, _ = cv2.projectPoints(points, rvec, tvec, K, None)
        pixels = pixels.reshape(-1, 2) + rng.normal(0, 0.2, (120, 2))
        transform, count = estimate_relative_pose(points, pixels, min_inliers=50)
        self.assertIsNotNone(transform)
        self.assertGreater(count, 100)
        np.testing.assert_allclose(transform[:3, 3], tvec, atol=0.025)

    def test_insufficient_correspondences(self):
        transform, count = estimate_relative_pose(
            np.zeros((4, 3)), np.zeros((4, 2))
        )
        self.assertIsNone(transform)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
