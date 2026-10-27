import unittest

import numpy as np

from xai_regression.neural_explanations import TorchImageClassifier


try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in minimal CI
    torch = None


@unittest.skipIf(torch is None, "PyTorch neural extra is not installed")
class TorchImageClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 4 * 4, 2),
        )
        self.classifier = TorchImageClassifier(
            self.network, (4, 4), [0, 1], scale=1.0
        )
        self.features = np.linspace(0.0, 1.0, 32, dtype=np.float32).reshape(2, 16)

    def test_predict_and_gradient_explanations(self):
        predictions = self.classifier.predict(self.features)

        self.assertEqual(self.classifier.predict_proba(self.features).shape, (2, 2))
        self.assertEqual(
            np.asarray(self.classifier.grad_cam(self.features, predictions)).shape,
            (2, 4, 4),
        )
        self.assertEqual(
            np.asarray(
                self.classifier.integrated_gradients(
                    self.features, predictions, steps=4
                )
            ).shape,
            (2, 4, 4),
        )

    def test_grad_cam_rejects_dense_network(self):
        dense = TorchImageClassifier(
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 2)),
            (4, 4),
            [0, 1],
            scale=1.0,
        )
        predictions = dense.predict(self.features)

        with self.assertRaisesRegex(ValueError, "Conv2d"):
            dense.grad_cam(self.features, predictions)


if __name__ == "__main__":
    unittest.main()
