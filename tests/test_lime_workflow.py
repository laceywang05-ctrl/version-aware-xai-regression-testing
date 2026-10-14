import unittest

import numpy as np

from xai_regression.lime_workflow import explanation_to_vector


class FakeExplanation:
    def __init__(self):
        self.local_exp = {1: [(0, 0.5), (2, -0.25)]}


class LimeWorkflowTests(unittest.TestCase):
    def test_sparse_explanation_is_mapped_to_feature_order(self):
        vector = explanation_to_vector(
            FakeExplanation(),
            label=1,
            n_features=4,
        )

        np.testing.assert_allclose(vector, [0.5, 0.0, -0.25, 0.0])

    def test_missing_class_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "class label 0"):
            explanation_to_vector(
                FakeExplanation(),
                label=0,
                n_features=4,
            )


if __name__ == "__main__":
    unittest.main()
