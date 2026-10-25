"""Calibrate MNIST explanation similarity from benign model retraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from xai_regression.upload_comparison import compare_classifiers, parse_manifest


DEFAULT_SEEDS = (11, 29, 47, 71)


def run_mnist_explanation_calibration(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    training_seeds: tuple[int, ...] = DEFAULT_SEEDS,
    train_limit: int = 1_000,
    sample_limit: int = 200,
    quantile: float = 0.05,
) -> dict:
    """Estimate a lower-tail similarity threshold from benign retraining."""
    if len(training_seeds) < 3:
        raise ValueError("MNIST calibration requires at least three training seeds.")
    if train_limit < 100:
        raise ValueError("MNIST calibration requires at least 100 training images.")
    if not 20 <= sample_limit <= 500:
        raise ValueError("MNIST calibration sample limit must be 20 to 500.")
    if not 0.0 < quantile < 0.5:
        raise ValueError("MNIST calibration quantile must be between 0 and 0.5.")

    source = Path(input_path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {"x_train", "y_train", "x_test", "y_test"}
            if not required <= set(archive.files):
                raise ValueError(
                    "MNIST calibration input must contain x_train, y_train, "
                    "x_test and y_test."
                )
            train_images = np.asarray(archive["x_train"][:train_limit])
            train_labels = np.asarray(archive["y_train"][:train_limit]).reshape(-1)
            test_images = np.asarray(archive["x_test"][:sample_limit])
            test_labels = np.asarray(archive["y_test"][:sample_limit]).reshape(-1)
    except OSError as error:
        raise ValueError(f"Could not read MNIST NPZ input: {source}.") from error
    if train_images.ndim != 3 or test_images.ndim != 3:
        raise ValueError("MNIST calibration expects grayscale images shaped (N,H,W).")
    if train_images.shape[1:] != test_images.shape[1:]:
        raise ValueError("MNIST training and test image shapes must match.")

    feature_count = int(np.prod(train_images.shape[1:]))
    columns = [f"pixel_{index}" for index in range(feature_count)]
    train_features = pd.DataFrame(
        train_images.reshape(len(train_images), -1), columns=columns
    )
    test_features = pd.DataFrame(
        test_images.reshape(len(test_images), -1), columns=columns
    )
    dataset = test_features.copy()
    dataset["target"] = test_labels
    manifest = parse_manifest(
        {
            "task": "classification",
            "label_column": "target",
            "image_shape": list(train_images.shape[1:]),
            "feature_columns": columns,
            "class_names": {str(value): f"digit {value}" for value in range(10)},
            "trusted_model_files": True,
            "dataset_note": "Official MNIST test subset for benign-retraining calibration.",
        }
    )
    models = [
        RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=1).fit(
            train_features, train_labels
        )
        for seed in training_seeds
    ]

    pairs = []
    calibration_similarities = []
    for left in range(len(models)):
        for right in range(left + 1, len(models)):
            comparison = compare_classifiers(
                models[left], models[right], dataset, manifest
            )
            same_prediction = [
                sample
                for sample in comparison["samples"]
                if sample["explanation_similarity"] is not None
                and not sample["prediction_changed"]
            ]
            similarities = [
                float(sample["explanation_similarity"])
                for sample in same_prediction
            ]
            calibration_similarities.extend(similarities)
            pairs.append(
                {
                    "left_seed": training_seeds[left],
                    "right_seed": training_seeds[right],
                    "prediction_agreement": comparison["summary"][
                        "prediction_agreement"
                    ],
                    "same_prediction_comparisons": len(similarities),
                    "mean_similarity": float(np.mean(similarities)),
                    "p05_similarity": float(np.quantile(similarities, 0.05)),
                }
            )

    similarities = np.asarray(calibration_similarities)
    calibrated_threshold = float(np.quantile(similarities, quantile))
    report = {
        "dataset": "MNIST 28x28 official split",
        "training_seeds": list(training_seeds),
        "train_limit": train_limit,
        "sample_limit": sample_limit,
        "model_configuration": {
            "type": "RandomForestClassifier",
            "n_estimators": 60,
        },
        "calibration_quantile": quantile,
        "same_prediction_comparisons": len(similarities),
        "mean_similarity": float(similarities.mean()),
        "median_similarity": float(np.median(similarities)),
        "calibrated_threshold": calibrated_threshold,
        "rate_below_0_80": float(np.mean(similarities < 0.80)),
        "pairs": pairs,
        "interpretation": (
            "The threshold estimates the lower tail of benign same-configuration "
            "retraining. It is a review baseline, not proof of harmful drift."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": destination, "report_path": report_path, **report}
