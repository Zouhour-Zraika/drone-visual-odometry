"""Regression tests for rigid alignment and timestamp overlap."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from evaluate_euroc_vo import align_rigid, evaluate


class EvaluationTests(unittest.TestCase):
    def test_rigid_alignment_without_scale(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.],
                           [0., 2., 0.], [1., 2., 3.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = source @ rotation.T + np.array([3., -2., 4.])
        r, t = align_rigid(source, target)
        np.testing.assert_allclose(source @ r.T + t, target, atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(r), 1., places=10)

    def test_overlap_and_perfect_positions(self):
        gt_t = np.arange(0., 7., 1.)
        gt_p = np.column_stack([gt_t, 2 * gt_t, np.zeros(len(gt_t))])
        vo_t = np.array([-1., 1., 2., 3., 4., 7.])
        vo_p = np.column_stack([vo_t, 2 * vo_t, np.zeros(len(vo_t))])
        result = evaluate(vo_t, vo_p, gt_t, gt_p)
        self.assertEqual(len(result["timestamps_s"]), 4)
        self.assertLess(result["ate_rmse_m"], 1e-9)
        self.assertLess(result["rpe_translation_rmse_m"], 1e-9)

    def test_no_overlap_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(np.array([20., 21., 22.]), np.zeros((3, 3)),
                     np.arange(5., dtype=float), np.zeros((5, 3)))


if __name__ == "__main__":
    unittest.main()
