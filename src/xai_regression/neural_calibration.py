"""Calibrate neural explanation similarity under benign retraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from xai_regression.image_bundle import _load_official_splits
from xai_regression.neural_explanations import TorchImageClassifier, _torch
from xai_regression.neural_image_demo import _networks, _train
from xai_regression.upload_comparison import _explanation_similarity


def _fit_model(images, labels, image_shape, classes, seed, network_index, epochs):
    torch = _torch()
    torch.manual_seed(seed)
    network = _networks(image_shape, len(classes))[network_index][1]
    _train(network, images, labels, epochs)
    return TorchImageClassifier(network, image_shape, classes.tolist())


def _maps(model, method, flat, sample_limit):
    features = flat[:sample_limit]
    predictions = model.predict(features)
    if method == "integrated_gradients":
        return model.integrated_gradients(features, predictions)
    return model.grad_cam(features, predictions)


def run_neural_explanation_calibration(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    train_limit: int = 1_000,
    sample_limit: int = 30,
    epochs: int = 2,
    comparison_seeds: tuple[int, ...] = (211, 223, 227, 229, 233),
    calibration_runs: int = 3,
) -> dict[str, Any]:
    """Estimate method-specific fifth-percentile benign similarity thresholds."""
    splits = _load_official_splits(input_path)
    if splits is None:
        raise ValueError("Calibration requires official NPZ train/test arrays.")
    train_images, train_labels, test_images, _ = splits
    train_images, train_labels = train_images[:train_limit], train_labels[:train_limit]
    flat_test = test_images[:sample_limit].reshape(sample_limit, -1)
    image_shape = tuple(train_images.shape[1:])
    classes = np.unique(train_labels)
    if calibration_runs < 1 or calibration_runs >= len(comparison_seeds):
        raise ValueError("calibration_runs must leave at least one held-out seed.")
    results = {}
    for method, network_index in (("integrated_gradients", 0), ("grad_cam", 2)):
        baseline = _fit_model(train_images, train_labels, image_shape, classes, 199, network_index, epochs)
        baseline_maps = _maps(baseline, method, flat_test, sample_limit)
        runs = []
        for seed in comparison_seeds:
            candidate = _fit_model(train_images, train_labels, image_shape, classes, seed, network_index, epochs)
            candidate_maps = _maps(candidate, method, flat_test, sample_limit)
            similarities = []
            informative = []
            for left, right in zip(baseline_maps, candidate_maps, strict=True):
                left_signal = float(np.linalg.norm(np.asarray(left, dtype=float))) > 1e-12
                right_signal = float(np.linalg.norm(np.asarray(right, dtype=float))) > 1e-12
                informative.append(left_signal and right_signal)
                similarities.append(_explanation_similarity(left, right))
            runs.append({"seed": seed, "similarities": similarities, "informative": informative})
        calibration_values = np.asarray(
            [
                value
                for run in runs[:calibration_runs]
                for value, keep in zip(run["similarities"], run["informative"], strict=True)
                if keep
            ],
            dtype=float,
        )
        if len(calibration_values) < max(1, sample_limit // 2):
            raise ValueError(f"Too few informative {method} maps for calibration.")
        threshold = float(np.quantile(calibration_values, 0.05))
        heldout = [
            value
            for run in runs[calibration_runs:]
            for value, keep in zip(run["similarities"], run["informative"], strict=True)
            if keep
        ]
        total_pairs = len(runs) * sample_limit
        informative_pairs = sum(sum(run["informative"]) for run in runs)
        informative_coverage = informative_pairs / total_pairs
        usable = threshold > 0.05 and informative_coverage >= 0.8
        results[method] = {
            "threshold": threshold,
            "usable_threshold": threshold if usable else None,
            "status": "CALIBRATED" if usable else "INSUFFICIENT_SEPARATION",
            "calibration_pairs": int(len(calibration_values)),
            "calibration_median": float(np.median(calibration_values)),
            "heldout_pairs": len(heldout),
            "heldout_below_threshold": int(np.count_nonzero(np.asarray(heldout) < threshold)),
            "heldout_review_rate": float(np.mean(np.asarray(heldout) < threshold)),
            "informative_pairs": informative_pairs,
            "informative_coverage": informative_coverage,
            "runs": runs,
        }
    summary = {
        "dataset": "MNIST",
        "train_samples": len(train_images),
        "explanation_samples_per_run": sample_limit,
        "epochs": epochs,
        "calibration_runs": calibration_runs,
        "heldout_runs": len(comparison_seeds) - calibration_runs,
        "methods": results,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {**summary, "output_dir": destination}
