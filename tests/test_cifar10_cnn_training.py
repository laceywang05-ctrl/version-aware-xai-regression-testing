"""Checks for the fixed CIFAR-10 training split and train-only preprocessing."""

import unittest

import numpy as np

try:
    import torch
    from xai_regression.cifar10_cnn_training import (
        _augment,
        _normalization_stats,
        build_cnn,
        split_training_indices,
    )
except ImportError:  # optional dependency in minimal CI
    torch = None


@unittest.skipIf(torch is None, "PyTorch neural extra is not installed")
class Cifar10CnnTrainingTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_reproducible(self):
        labels = np.tile(np.arange(10), 20)
        train, validation = split_training_indices(labels, validation_size=40, seed=42)
        again_train, again_validation = split_training_indices(labels, validation_size=40, seed=42)
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(validation, again_validation)
        self.assertEqual(len(set(train) & set(validation)), 0)
        np.testing.assert_array_equal(np.bincount(labels[train]), np.full(10, 16))
        np.testing.assert_array_equal(np.bincount(labels[validation]), np.full(10, 4))

    def test_normalization_uses_only_selected_training_images(self):
        images = np.zeros((3, 2, 2, 3), dtype=np.uint8)
        images[0, :, :, :] = [0, 50, 100]
        images[1, :, :, :] = [100, 150, 200]
        images[2, :, :, :] = [255, 255, 255]
        mean, std = _normalization_stats(images, np.array([0, 1]), batch_size=1)
        np.testing.assert_allclose(mean, np.array([50, 100, 150]) / 255)
        np.testing.assert_allclose(std, np.full(3, 50 / 255))

    def test_network_accepts_image_batch_and_exposes_ten_logits(self):
        network = build_cnn([0.5] * 3, [0.25] * 3)
        images = torch.rand(2, 3, 32, 32)
        augmented = _augment(images)
        self.assertEqual(tuple(augmented.shape), (2, 3, 32, 32))
        self.assertEqual(tuple(network(augmented).shape), (2, 10))

    def test_wider_version_changes_capacity_without_changing_output_contract(self):
        baseline = build_cnn([0.5] * 3, [0.25] * 3)
        candidate = build_cnn([0.5] * 3, [0.25] * 3, widths=(40, 80, 160))
        self.assertGreater(
            sum(parameter.numel() for parameter in candidate.parameters()),
            sum(parameter.numel() for parameter in baseline.parameters()),
        )
        self.assertEqual(tuple(candidate(torch.rand(2, 3, 32, 32)).shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
