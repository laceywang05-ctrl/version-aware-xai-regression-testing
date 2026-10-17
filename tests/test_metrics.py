import unittest

import numpy as np

from xai_regression.metrics import (
    absolute_rank_correlation,
    cosine_similarity,
    direction_flip_rate,
    top_k_overlap,
)


class ExplanationMetricTests(unittest.TestCase):
    def test_identical_vectors_have_perfect_similarity(self):
        values = np.array([1.0, -2.0, 0.5])
        self.assertAlmostEqual(cosine_similarity(values, values), 1.0)
        self.assertAlmostEqual(absolute_rank_correlation(values, values), 1.0)
        self.assertAlmostEqual(top_k_overlap(values, values, 2), 1.0)
        self.assertAlmostEqual(direction_flip_rate(values, values), 0.0)

    def test_zero_vectors_are_treated_as_identical(self):
        zeros = np.zeros(3)
        self.assertEqual(cosine_similarity(zeros, zeros), 1.0)
        self.assertEqual(direction_flip_rate(zeros, zeros), 0.0)

    def test_direction_flip_is_detected(self):
        left = np.array([1.0, -2.0, 3.0])
        right = np.array([-1.0, -2.0, -3.0])
        self.assertAlmostEqual(direction_flip_rate(left, right), 2 / 3)

    def test_top_k_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            top_k_overlap(np.ones(3), np.ones(3), 0)


if __name__ == "__main__":
    unittest.main()

