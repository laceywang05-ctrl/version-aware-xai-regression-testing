import unittest

from xai_regression.metrics import (
    cosine_similarity,
    direction_flip_rate,
    spearman_rank_correlation,
    top_k_overlap,
)


class MetricTests(unittest.TestCase):
    def test_identical_vectors_have_full_similarity(self):
        values = [0.5, -0.2, 0.1]
        self.assertAlmostEqual(cosine_similarity(values, values), 1.0)
        self.assertAlmostEqual(spearman_rank_correlation(values, values), 1.0)
        self.assertAlmostEqual(top_k_overlap(values, values, 2), 1.0)
        self.assertAlmostEqual(direction_flip_rate(values, values), 0.0)

    def test_direction_flip_is_detected(self):
        self.assertAlmostEqual(
            direction_flip_rate([1.0, -2.0], [-1.0, -3.0]),
            0.5,
        )

    def test_top_k_overlap_uses_absolute_importance(self):
        self.assertAlmostEqual(
            top_k_overlap([10.0, -8.0, 1.0], [9.0, 0.5, -7.0], 2),
            0.5,
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            cosine_similarity([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
