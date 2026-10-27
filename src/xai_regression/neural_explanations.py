"""Optional PyTorch image-classifier adapter and gradient explanations.

PyTorch is deliberately an optional dependency: the existing scikit-learn
upload workflow remains lightweight, while trusted ``.joblib`` files may wrap
a PyTorch network with :class:`TorchImageClassifier` when neural explanations
are required.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "PyTorch is required for Integrated Gradients and Grad-CAM. "
            "Install the project with the neural extra."
        ) from error
    return torch


def _normalize_map(values: Any) -> list[list[float]]:
    array = np.asarray(values, dtype=float)
    maximum = float(np.max(np.abs(array)))
    if maximum > 0:
        array = array / maximum
    else:
        array = np.zeros_like(array)
    return array.round(5).tolist()


class TorchImageClassifier:
    """Expose a trusted PyTorch image network through the upload API.

    The wrapped network must return unnormalised class logits. Input CSV rows
    are flattened HWC images and are converted to NCHW tensors internally.
    """

    def __init__(
        self,
        network: Any,
        image_shape: tuple[int, ...],
        classes: list[Any] | np.ndarray,
        *,
        scale: float = 255.0,
    ) -> None:
        if len(image_shape) not in {2, 3}:
            raise ValueError("image_shape must be (H,W) or (H,W,C).")
        self.network = network
        self.image_shape = tuple(int(value) for value in image_shape)
        self.classes_ = np.asarray(classes)
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError("scale must be positive.")

    def _tensor(self, features: Any, *, requires_grad: bool = False):
        torch = _torch()
        array = np.asarray(features, dtype=np.float32)
        expected = int(np.prod(self.image_shape))
        if array.ndim != 2 or array.shape[1] != expected:
            raise ValueError(f"Expected flattened image rows with {expected} features.")
        images = array.reshape((len(array), *self.image_shape)) / self.scale
        if len(self.image_shape) == 2:
            images = images[:, None, :, :]
        else:
            images = np.moveaxis(images, -1, 1)
        return torch.tensor(images, dtype=torch.float32, requires_grad=requires_grad)

    def predict_proba(self, features: Any) -> np.ndarray:
        torch = _torch()
        self.network.eval()
        with torch.no_grad():
            logits = self.network(self._tensor(features))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, features: Any) -> np.ndarray:
        indices = np.argmax(self.predict_proba(features), axis=1)
        return self.classes_[indices]

    def integrated_gradients(
        self,
        features: Any,
        predictions: np.ndarray,
        *,
        steps: int = 24,
    ) -> list[list[list[float]]]:
        """Compute zero-baseline Integrated Gradients for predicted logits."""
        torch = _torch()
        if steps < 2:
            raise ValueError("Integrated Gradients requires at least two steps.")
        inputs = self._tensor(features)
        baseline = torch.zeros_like(inputs)
        class_lookup = {value: index for index, value in enumerate(self.classes_.tolist())}
        results = []
        self.network.eval()
        for row, prediction in zip(inputs, predictions, strict=True):
            total_gradient = torch.zeros_like(row)
            for alpha in torch.linspace(0.0, 1.0, steps, dtype=row.dtype):
                interpolated = (baseline[0] + alpha * (row - baseline[0])).detach()
                interpolated.requires_grad_(True)
                self.network.zero_grad(set_to_none=True)
                score = self.network(interpolated.unsqueeze(0))[0, class_lookup[prediction]]
                score.backward()
                total_gradient += interpolated.grad.detach()
            attribution = (row - baseline[0]) * total_gradient / steps
            spatial = attribution.detach().cpu().numpy().sum(axis=0)
            results.append(_normalize_map(spatial))
        return results

    def grad_cam(
        self,
        features: Any,
        predictions: np.ndarray,
    ) -> list[list[list[float]]]:
        """Compute Grad-CAM at the final convolutional layer."""
        torch = _torch()
        convolution = None
        for module in self.network.modules():
            if isinstance(module, torch.nn.Conv2d):
                convolution = module
        if convolution is None:
            raise ValueError("Grad-CAM requires at least one Conv2d layer.")
        class_lookup = {value: index for index, value in enumerate(self.classes_.tolist())}
        results = []
        self.network.eval()
        for row, prediction in zip(self._tensor(features), predictions, strict=True):
            captured: dict[str, Any] = {}

            def forward_hook(_module, _inputs, output):
                captured["activation"] = output
                output.retain_grad()

            handle = convolution.register_forward_hook(forward_hook)
            try:
                self.network.zero_grad(set_to_none=True)
                logits = self.network(row.unsqueeze(0))
                logits[0, class_lookup[prediction]].backward()
                activation = captured["activation"]
                gradient = activation.grad
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = torch.nn.functional.interpolate(
                    cam,
                    size=self.image_shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                results.append(_normalize_map(cam.detach().cpu().numpy()))
            finally:
                handle.remove()
        return results

