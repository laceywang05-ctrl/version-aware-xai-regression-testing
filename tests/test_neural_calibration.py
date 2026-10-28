from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from xai_regression.neural_calibration import run_neural_explanation_calibration

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch neural extra is not installed")
class NeuralCalibrationTests(unittest.TestCase):
    def test_calibrates_both_neural_methods(self):
        rng = np.random.default_rng(4)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mnist.npz"
            np.savez(
                source,
                x_train=rng.integers(0, 256, (100, 4, 4), dtype=np.uint8),
                y_train=np.tile(np.arange(2), 50),
                x_test=rng.integers(0, 256, (20, 4, 4), dtype=np.uint8),
                y_test=np.tile(np.arange(2), 10),
            )
            result = run_neural_explanation_calibration(
                source, root / "output", train_limit=100, sample_limit=4,
                epochs=1, comparison_seeds=(11, 13), calibration_runs=1,
            )
            self.assertEqual(set(result["methods"]), {"integrated_gradients", "grad_cam"})
            self.assertTrue((root / "output" / "summary.json").exists())
            for row in result["methods"].values():
                self.assertGreaterEqual(row["calibration_pairs"], 1)
                self.assertGreaterEqual(row["heldout_pairs"], 1)
                self.assertGreater(row["informative_coverage"], 0)
