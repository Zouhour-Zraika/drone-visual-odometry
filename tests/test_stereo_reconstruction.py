"""Deterministic tests for rectified stereo back-projection."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from stereo_reconstruction import triangulate_rectified


class TriangulationTests(unittest.TestCase):
    def test_known_depth_and_center(self):
        fx, baseline = 400.0, 0.1
        left = np.array([[320.0, 240.0]])
        right = np.array([[300.0, 240.0]])
        xyz, kept = triangulate_rectified(
            left, right, fx=fx, fy=fx, cx=320.0, cy=240.0,
            baseline_m=baseline,
        )
        np.testing.assert_allclose(xyz, [[0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(kept, [0])

    def test_negative_disparity_and_vertical_offset_rejected(self):
        left = np.array([[100., 100.], [100., 100.], [100., 100.]])
        right = np.array([[110., 100.], [90., 105.], [90., 100.]])
        xyz, kept = triangulate_rectified(left, right)
        self.assertEqual(len(xyz), 1)
        np.testing.assert_array_equal(kept, [2])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            triangulate_rectified(np.ones((3, 2)), np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
