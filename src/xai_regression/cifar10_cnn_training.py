"""Validation-selected CIFAR-10 CNN training with resumable checkpoints.

This module intentionally imports the optional neural dependency only when the
training command (or one of its serialized networks) is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

from xai_regression.image_bundle import (
    _load_cifar10_python_splits,
    prepare_image_npz_bundle,
)
from xai_regression.neural_explanations import TorchImageClassifier


class NormalizeRGB(nn.Module):
    """Keep training-only normalization inside the exported model."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1).float())
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1).float())

    def forward(self, images):
        return (images - self.mean) / self.std


def build_cnn(mean, std, *, widths=(32, 64, 128)):
    """Six convolutional layers, three pooling stages and a linear head."""
    if len(widths) != 3 or any(width < 1 for width in widths):
        raise ValueError("widths must contain three positive channel counts.")
    layers = [NormalizeRGB(mean, std)]
    channels = 3
    for width, dropout in zip(widths, (0.10, 0.15, 0.20)):
        for _ in range(2):
            layers.extend([
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(),
            ])
            channels = width
        layers.extend([nn.MaxPool2d(2), nn.Dropout2d(dropout)])
    layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(widths[-1], 10)])
    return nn.Sequential(*layers)


def split_training_indices(labels, *, validation_size=5000, seed=42):
    indices = np.arange(len(labels))
    train, validation = train_test_split(
        indices, test_size=validation_size, random_state=seed, stratify=labels
    )
    return np.sort(train), np.sort(validation)


def _normalization_stats(images, indices, *, batch_size=1024):
    """Compute channel statistics without materializing all images as float64."""
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    pixels = 0
    for offset in range(0, len(indices), batch_size):
        block = images[indices[offset:offset + batch_size]].astype(np.float64) / 255.0
        total += block.sum(axis=(0, 1, 2))
        squares += np.square(block).sum(axis=(0, 1, 2))
        pixels += block.shape[0] * block.shape[1] * block.shape[2]
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean * mean, 1e-12))
    return mean.tolist(), std.tolist()


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    # Fail before loading the dataset if an explicitly requested device cannot run.
    torch.zeros(1, device=device)
    return device


def _augment(images):
    """Independent random padded crop and horizontal flip for each image."""
    count, channels, height, width = images.shape
    padded = nn.functional.pad(images, (4, 4, 4, 4), mode="reflect")
    dy = torch.randint(0, 9, (count, 1, 1), device=images.device)
    dx = torch.randint(0, 9, (count, 1, 1), device=images.device)
    rows = torch.arange(height, device=images.device).reshape(1, height, 1) + dy
    cols = torch.arange(width, device=images.device).reshape(1, 1, width) + dx
    index = (rows * (width + 8) + cols).reshape(count, 1, -1)
    cropped = padded.flatten(2).gather(2, index.expand(-1, channels, -1))
    cropped = cropped.reshape(count, channels, height, width)
    flip = torch.rand(count, 1, 1, 1, device=images.device) < 0.5
    return torch.where(flip, cropped.flip(-1), cropped)


def _batch(images, labels, indices, device):
    idx = torch.as_tensor(indices, dtype=torch.long)
    inputs = images[idx].to(device=device, dtype=torch.float32).div_(255)
    target = labels[idx].to(device=device)
    return inputs, target


def evaluate(network, images, labels, indices, device, *, batch_size=256):
    network.eval()
    loss_sum = 0.0
    probabilities = []
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            x, y = _batch(images, labels, indices[offset:offset + batch_size], device)
            logits = network(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    probability = np.concatenate(probabilities)
    actual = labels[torch.as_tensor(indices)].numpy()
    prediction = probability.argmax(axis=1)
    return {
        "loss": loss_sum / len(indices),
        "accuracy": float(accuracy_score(actual, prediction)),
        "macro_f1": float(f1_score(actual, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, prediction, labels=np.arange(10)).tolist(),
    }, probability


def _cpu_state(network):
    return {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def _random_state(device):
    """Capture augmentation and dropout RNG streams at an epoch boundary."""
    state = {"torch_cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
    if device.type == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state, device):
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["torch_mps"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _plot_history(destination, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train (augmented)")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train (augmented)")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination / "training_curves.png", dpi=160)
    plt.close(figure)


def train_cifar10_cnn(
    input_path, output_dir, *, epochs=80, batch_size=256, patience=15,
    min_epochs=30, baseline_epoch=5, seed=42, device="auto", resume=False,
    train_limit=50000, validation_size=5000, demo_limit=1000,
    widths=(32, 64, 128),
):
    if not 1 <= baseline_epoch <= min_epochs <= epochs:
        raise ValueError("Require 1 <= baseline_epoch <= min_epochs <= epochs.")
    if batch_size < 1 or patience < 1 or not 1 <= demo_limit <= 1000:
        raise ValueError("Batch size/patience must be positive; demo_limit must be 1..1000.")
    if not 100 <= train_limit <= 50000 or not 10 <= validation_size < train_limit:
        raise ValueError("Require 100 <= train_limit <= 50000 and a smaller validation size.")
    widths = tuple(widths)
    if len(widths) != 3 or any(width < 1 for width in widths):
        raise ValueError("widths must contain three positive channel counts.")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise ValueError("Output folder is not empty. Use --resume for this run or a new folder.")
    destination.mkdir(parents=True, exist_ok=True)
    selected_device = select_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    source = Path(input_path)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    print(f"Loading CIFAR-10, device={selected_device}, archive SHA-256={source_hash}", flush=True)
    train_images, train_labels, test_images, test_labels, class_names = _load_cifar10_python_splits(
        source, limit=50000
    )
    if len(train_images) != 50000 or len(test_images) != 10000:
        raise ValueError("This command requires the official 50,000/10,000 CIFAR-10 split.")
    train_images, train_labels = train_images[:train_limit], train_labels[:train_limit]
    training, validation = split_training_indices(train_labels, validation_size=validation_size, seed=seed)
    mean, std = _normalization_stats(train_images, training)
    images = torch.from_numpy(np.ascontiguousarray(train_images.transpose(0, 3, 1, 2)))
    labels = torch.from_numpy(train_labels.astype(np.int64))
    config = {
        "schema_version": 1, "dataset": "CIFAR-10", "archive_sha256": source_hash,
        "training_samples": len(training), "validation_samples": len(validation),
        "official_test_samples": len(test_images), "epochs": epochs, "min_epochs": min_epochs,
        "patience": patience, "baseline_epoch": baseline_epoch, "batch_size": batch_size,
        "seed": seed, "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0005,
        "scheduler": "CosineAnnealingLR", "minimum_learning_rate": 0.00001,
        "augmentation": "per-image reflection-padded random crop (padding=4), horizontal flip p=0.5",
        "mean": mean, "std": std, "normalization_source": "training subset only",
        "selection": "highest validation accuracy, lower validation loss breaks ties",
        "device": str(selected_device), "torch_version": torch.__version__, "demo_samples": demo_limit,
        "test_policy": "test data evaluated only after validation-based checkpoint selection",
        "architecture": f"6 convolutional layers ({widths[0]}/{widths[0]}/{widths[1]}/{widths[1]}/{widths[2]}/{widths[2]}), batch norm, pooling, dropout, global average pooling, 10 outputs",
        "widths": list(widths),
    }
    config_path = destination / "config.json"
    if resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        for key, value in config.items():
            if key not in {"device", "torch_version"} and previous.get(key) != value:
                raise ValueError(f"Resume configuration changed: {key}.")
    _write_json(config_path, config)
    np.savez(destination / "split_indices.npz", train=training, validation=validation)
    network = build_cnn(mean, std, widths=widths).to(selected_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
    history = []
    best_accuracy, best_loss, best_epoch, start_epoch = -1.0, float("inf"), 0, 1
    checkpoint_path = destination / "last.pt"
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        network.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for state_values in optimizer.state.values():
            for key, value in state_values.items():
                if torch.is_tensor(value) and key != "step":
                    state_values[key] = value.to(selected_device)
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_accuracy, best_loss, best_epoch = state["best_accuracy"], state["best_loss"], state["best_epoch"]
        start_epoch = state["epoch"] + 1
        if "random_state" not in state:
            raise ValueError("This checkpoint lacks RNG state and cannot resume reproducibly. Start a new run.")
        _restore_random_state(state["random_state"], selected_device)
        print(f"Resuming after epoch {start_epoch - 1}", flush=True)
    print(f"Train={len(training)}, validation={len(validation)}, final test=10000; parameters={sum(p.numel() for p in network.parameters()):,}", flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.monotonic()
        network.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(training)
        loss_sum, correct = 0.0, 0
        for batch_number, offset in enumerate(range(0, len(order), batch_size), start=1):
            x, y = _batch(images, labels, order[offset:offset + batch_size], selected_device)
            x = _augment(x)
            optimizer.zero_grad(set_to_none=True)
            logits = network(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            if batch_number % 50 == 0:
                print(f"  epoch={epoch} images={min(offset + batch_size, len(order))}/{len(order)} loss={loss_sum / min(offset + batch_size, len(order)):.4f}", flush=True)
        validation_metrics, _ = evaluate(network, images, labels, validation, selected_device, batch_size=batch_size)
        row = {
            "epoch": epoch, "train_loss": loss_sum / len(training), "train_accuracy": correct / len(training),
            "val_loss": validation_metrics["loss"], "val_accuracy": validation_metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        improved = row["val_accuracy"] > best_accuracy or (
            row["val_accuracy"] == best_accuracy and row["val_loss"] < best_loss
        )
        model_state = _cpu_state(network)
        if improved:
            best_accuracy, best_loss, best_epoch = row["val_accuracy"], row["val_loss"], epoch
            _save_checkpoint(destination / "best.pt", {"model": model_state, "epoch": epoch, "validation": validation_metrics})
        if epoch == baseline_epoch:
            _save_checkpoint(destination / "baseline.pt", {"model": model_state, "epoch": epoch, "validation": validation_metrics})
        scheduler.step()
        _save_checkpoint(checkpoint_path, {
            "model": model_state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "history": history, "best_accuracy": best_accuracy, "best_loss": best_loss, "best_epoch": best_epoch,
            "random_state": _random_state(selected_device),
        })
        _write_json(destination / "history.json", history)
        _write_json(destination / "progress.json", {"status": "training", "epoch": epoch, "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy, "latest": row})
        print(f"Epoch {epoch:03d}/{epochs}: train={row['train_accuracy']:.2%}, val={row['val_accuracy']:.2%}, val_loss={row['val_loss']:.4f}, best={best_accuracy:.2%}@{best_epoch}, {row['seconds']:.1f}s", flush=True)
        if epoch >= min_epochs and epoch - best_epoch >= patience:
            print(f"Early stopping: no validation improvement for {patience} epochs.", flush=True)
            break
    _plot_history(destination, history)
    print("Training complete. Evaluating the preselected V1 and V2 on all 10,000 official test images.", flush=True)
    test_tensor = torch.from_numpy(np.ascontiguousarray(test_images.transpose(0, 3, 1, 2)))
    test_targets = torch.from_numpy(test_labels.astype(np.int64))
    bundle = destination / "upload_bundle"
    bundle.mkdir(exist_ok=True)
    results, probability_rows = {}, {}
    for role, filename, checkpoint in (
        ("v1", f"cifar10_cnn_v1_epoch{baseline_epoch:02d}.joblib", "baseline.pt"),
        ("v2", "cifar10_cnn_v2_best.joblib", "best.pt"),
    ):
        saved = torch.load(destination / checkpoint, map_location="cpu", weights_only=False)
        network.load_state_dict(saved["model"])
        metrics, probabilities = evaluate(network, test_tensor, test_targets, np.arange(10000), selected_device, batch_size=batch_size)
        results[role] = {"filename": filename, "epoch": saved["epoch"], "validation": saved["validation"], "test": metrics}
        probability_rows[role] = probabilities
        cpu_network = build_cnn(mean, std, widths=widths)
        cpu_network.load_state_dict(saved["model"])
        cpu_network.eval()
        wrapper = TorchImageClassifier(cpu_network, (32, 32, 3), list(range(10)))
        joblib.dump(wrapper, bundle / filename)
        reloaded = joblib.load(bundle / filename)
        np.testing.assert_allclose(
            reloaded.predict_proba(test_images[:4].reshape(4, -1)), probabilities[:4], rtol=2e-4, atol=2e-5
        )
        # Both neural explanation paths must work on the serialized classifier.
        features = test_images[:1].reshape(1, -1)
        predicted = reloaded.predict(features)
        cam = np.asarray(reloaded.grad_cam(features, predicted))
        ig = np.asarray(reloaded.integrated_gradients(features, predicted))
        if cam.shape != (1, 32, 32) or ig.shape != (1, 32, 32) or not np.isfinite(cam).all() or not np.isfinite(ig).all():
            raise RuntimeError("Exported explanation smoke check failed.")
        print(f"{role.upper()} epoch={saved['epoch']}: full-test accuracy={metrics['accuracy']:.2%}, macro_F1={metrics['macro_f1']:.2%}", flush=True)
    predictions_v1, predictions_v2 = probability_rows["v1"].argmax(1), probability_rows["v2"].argmax(1)
    right_v1, right_v2 = predictions_v1 == test_labels, predictions_v2 == test_labels
    results["paired"] = {
        "prediction_agreement": float(np.mean(predictions_v1 == predictions_v2)),
        "both_correct": int(np.sum(right_v1 & right_v2)),
        "wrong_to_correct": int(np.sum(~right_v1 & right_v2)),
        "correct_to_wrong": int(np.sum(right_v1 & ~right_v2)),
        "both_wrong": int(np.sum(~right_v1 & ~right_v2)),
    }
    with (destination / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["official_test_index", "actual", "v1_prediction", "v1_confidence", "v2_prediction", "v2_confidence"])
        for i in range(10000):
            writer.writerow([i, int(test_labels[i]), int(predictions_v1[i]), float(probability_rows["v1"][i].max()), int(predictions_v2[i]), float(probability_rows["v2"][i].max())])
    temporary = destination / "demo_test_images.npz"
    np.savez(temporary, images=test_images[:demo_limit], labels=test_labels[:demo_limit])
    prepare_image_npz_bundle(temporary, bundle, limit=demo_limit, class_names=class_names)
    temporary.unlink()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_note"] = (
        f"CIFAR-10 official test images 0..{demo_limit-1}, for dashboard inspection only. "
        f"Training used {len(training):,} official training images and {len(validation):,} validation images. "
        "Full 10,000-test metrics are in the experiment summary; Grad-CAM is exploratory."
    )
    _write_json(manifest_path, manifest)
    summary = {
        "config": config, "parameters": sum(p.numel() for p in network.parameters()),
        "epochs_completed": history[-1]["epoch"], "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy, "elapsed_seconds_this_invocation": time.monotonic() - started,
        "models": results, "upload_bundle": str(bundle.resolve()),
        "scope": "One trained CNN family and one seed. V1 is a fixed early checkpoint; V2 is selected by validation only. No automatic calibrated CIFAR-10 neural drift threshold is claimed.",
    }
    _write_json(destination / "summary.json", summary)
    _write_json(bundle / "README.json", summary)
    _write_json(destination / "progress.json", {"status": "complete", "epoch": history[-1]["epoch"], "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy})
    return summary
