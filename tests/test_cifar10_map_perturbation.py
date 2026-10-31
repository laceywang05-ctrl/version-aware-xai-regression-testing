"""Checks for the exploratory CIFAR-10 pixel-occlusion audit."""

import unittest

import numpy as np

from xai_regression.cifar10_map_perturbation import _masked, _selected_indices


class Cifar10MapPerturbationTests(unittest.TestCase):
    def test_mask_replaces_all_rgb_channels_at_only_selected_pixels(self):
        image = np.arange(32 * 32 * 3, dtype=np.float32)
        result = _masked(image, np.array([0, 100, 1023]), np.array([7, 8, 9]))
        expected = image.reshape(-1, 3).copy()
        expected[[0, 100, 1023]] = [7, 8, 9]
        np.testing.assert_array_equal(result, expected.reshape(-1))
        np.testing.assert_array_equal(image[:3], [0, 1, 2])

    def test_selection_requires_both_models_correct_in_each_class(self):
        labels = np.repeat(np.arange(10), 4)
        first, second = labels.copy(), labels.copy()
        first[0] = 1
        second[4] = 0
        chosen = _selected_indices(labels, first, second, per_class=2, seed=11)
        self.assertEqual(len(chosen), 20)
        self.assertNotIn(0, chosen)
        self.assertNotIn(4, chosen)
        self.assertTrue(all(first[index] == second[index] == labels[index] for index in chosen))
        self.assertEqual(chosen, _selected_indices(labels, first, second, per_class=2, seed=11))


if __name__ == "__main__":
    unittest.main()
