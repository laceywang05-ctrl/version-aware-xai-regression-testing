import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from xai_regression.mnist_calibration import run_mnist_explanation_calibration


class MnistCalibrationTests(unittest.TestCase):
    def test_calibration_uses_benign_retraining_pairs(self):
        rng = np.random.default_rng(42)
        x_train = rng.integers(0, 256, size=(120, 4, 4), dtype=np.uint8)
        y_train = np.tile(np.arange(3), 40)
        x_test = rng.integers(0, 256, size=(30, 4, 4), dtype=np.uint8)
        y_test = np.tile(np.arange(3), 10)
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "mnist.npz"
            np.savez(
                source,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            result = run_mnist_explanation_calibration(
                source,
                Path(temporary) / "output",
                training_seeds=(11, 29, 47),
                validation_seeds=(59, 71),
                train_limit=120,
                sample_limit=20,
            )
            self.assertEqual(len(result["pairs"]), 3)
            self.assertGreater(result["same_prediction_comparisons"], 0)
            self.assertGreaterEqual(result["calibrated_threshold"], -1.0)
            self.assertLessEqual(result["calibrated_threshold"], 1.0)
            self.assertEqual(len(result["heldout_validation"]["pairs"]), 1)
            self.assertGreater(
                result["heldout_validation"]["same_prediction_comparisons"], 0
            )

    def test_calibration_rejects_overlapping_seed_sets(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run_mnist_explanation_calibration(
                "unused.npz",
                "unused",
                training_seeds=(11, 29, 47),
                validation_seeds=(47, 59),
            )


if __name__ == "__main__":
    unittest.main()
