import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np

from xai_regression.neural_image_demo import build_neural_image_demo


try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency in minimal CI
    torch = None


@unittest.skipIf(torch is None, "PyTorch neural extra is not installed")
class NeuralImageDemoTests(unittest.TestCase):
    def test_builds_four_upload_ready_models(self):
        rng = np.random.default_rng(8)
        train_labels = np.tile(np.arange(2), 50)
        test_labels = np.tile(np.arange(2), 10)
        train_images = rng.integers(0, 256, size=(100, 4, 4), dtype=np.uint8)
        test_images = rng.integers(0, 256, size=(20, 4, 4), dtype=np.uint8)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mnist.npz"
            output = root / "output"
            np.savez(
                source,
                x_train=train_images,
                y_train=train_labels,
                x_test=test_images,
                y_test=test_labels,
            )

            result = build_neural_image_demo(
                source, output, limit=100, test_limit=20, epochs=1
            )

            self.assertEqual(result["model_count"], 4)
            self.assertTrue((output / "samples.csv").exists())
            self.assertTrue((output / "images.zip").exists())
            summary = json.loads((output / "README.json").read_text())
            self.assertEqual(
                [row["explanation_method"] for row in summary["models"]],
                ["Integrated Gradients", "Integrated Gradients", "Grad-CAM", "Grad-CAM"],
            )
            model = joblib.load(output / "model_3_cnn_shallow.joblib")
            self.assertEqual(model.predict(test_images.reshape(20, -1)).shape, (20,))


if __name__ == "__main__":
    unittest.main()
