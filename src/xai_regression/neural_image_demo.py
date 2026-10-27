"""Create upload-ready neural image models for IG and Grad-CAM demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from xai_regression.image_bundle import _load_official_splits, prepare_image_npz_bundle
from xai_regression.neural_explanations import TorchImageClassifier, _torch


def _networks(image_shape: tuple[int, ...], class_count: int) -> list[tuple[str, Any, str]]:
    torch = _torch()
    height, width = image_shape[:2]
    channels = 1 if len(image_shape) == 2 else image_shape[2]
    feature_count = int(np.prod(image_shape))
    return [
        (
            "model_1_mlp_small.joblib",
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(feature_count, 64), torch.nn.ReLU(), torch.nn.Linear(64, class_count)),
            "Integrated Gradients",
        ),
        (
            "model_2_mlp_large.joblib",
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(feature_count, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, class_count)),
            "Integrated Gradients",
        ),
        (
            "model_3_cnn_shallow.joblib",
            torch.nn.Sequential(torch.nn.Conv2d(channels, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(8 * (height // 2) * (width // 2), class_count)),
            "Grad-CAM",
        ),
        (
            "model_4_cnn_deep.joblib",
            torch.nn.Sequential(torch.nn.Conv2d(channels, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2), torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d((4, 4)), torch.nn.Flatten(), torch.nn.Linear(16 * 4 * 4, class_count)),
            "Grad-CAM",
        ),
    ]


def _tensor(images: np.ndarray):
    torch = _torch()
    values = images.astype(np.float32) / 255.0
    values = values[:, None, :, :] if values.ndim == 3 else np.moveaxis(values, -1, 1)
    return torch.tensor(values, dtype=torch.float32)


def _train(network: Any, images: np.ndarray, labels: np.ndarray, epochs: int) -> None:
    torch = _torch()
    generator = torch.Generator().manual_seed(17)
    dataset = torch.utils.data.TensorDataset(_tensor(images), torch.tensor(labels, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    loss_function = torch.nn.CrossEntropyLoss()
    network.train()
    for _ in range(epochs):
        for features, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(features), target)
            loss.backward()
            optimizer.step()


def build_neural_image_demo(input_path: str | Path, output_dir: str | Path, *, limit: int = 1_000, test_limit: int = 500, epochs: int = 3) -> dict[str, Any]:
    """Train two MLPs and two CNNs and write a browser-upload bundle."""
    if limit < 100 or limit > 5_000:
        raise ValueError("Training limit must be between 100 and 5,000.")
    if test_limit < 20 or test_limit > 1_000:
        raise ValueError("Test limit must be between 20 and 1,000.")
    if epochs < 1 or epochs > 20:
        raise ValueError("Epochs must be between 1 and 20.")
    splits = _load_official_splits(input_path)
    if splits is None:
        raise ValueError("Neural demo requires NPZ x_train/y_train/x_test/y_test arrays.")
    train_images, train_labels, test_images, test_labels = splits
    train_images, train_labels = train_images[:limit], train_labels[:limit]
    test_images, test_labels = test_images[:test_limit], test_labels[:test_limit]
    classes = np.unique(np.concatenate([train_labels, test_labels]))
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Neural demo requires zero-based contiguous integer classes.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    torch.manual_seed(100)
    records = []
    flat_test = test_images.reshape(len(test_images), -1)
    for index, (filename, network, method) in enumerate(_networks(tuple(train_images.shape[1:]), len(classes)), start=1):
        _train(network, train_images, train_labels, epochs)
        wrapper = TorchImageClassifier(network, tuple(train_images.shape[1:]), classes.tolist())
        accuracy = float(np.mean(wrapper.predict(flat_test) == test_labels))
        joblib.dump(wrapper, destination / filename)
        records.append({"filename": filename, "architecture": str(network), "parameters": int(sum(p.numel() for p in network.parameters())), "explanation_method": method, "test_accuracy": accuracy})
    temporary = destination / "test-data.npz"
    np.savez(temporary, images=test_images, labels=test_labels)
    prepared = prepare_image_npz_bundle(
        temporary,
        destination,
        limit=len(test_images),
        class_names={str(value): f"digit {value}" for value in classes.tolist()},
    )
    temporary.unlink()
    summary = {"source": Path(input_path).name, "training_samples": len(train_images), "test_samples": len(test_images), "epochs": epochs, "models": records, "upload_files": {"dataset": "samples.csv", "images": "images.zip", "manifest": "manifest.json"}}
    (destination / "README.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": destination, "model_count": len(records), "training_samples": len(train_images), "test_samples": prepared["sample_count"], "best_accuracy": max(record["test_accuracy"] for record in records)}
