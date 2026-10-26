"""Calibrate CIFAR-10 RGB explanation similarity from benign retraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from xai_regression.image_bundle import CIFAR10_CLASS_NAMES, _load_cifar10_python_splits
from xai_regression.upload_comparison import compare_classifiers, parse_manifest


DEFAULT_SEEDS = (11, 29, 47)
DEFAULT_VALIDATION_SEEDS = (59, 71)


def _pair_similarities(comparison: dict) -> list[float]:
    return [
        float(sample["explanation_similarity"])
        for sample in comparison["samples"]
        if sample["explanation_similarity"] is not None
        and not sample["prediction_changed"]
    ]


def run_cifar10_explanation_calibration(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    training_seeds: tuple[int, ...] = DEFAULT_SEEDS,
    validation_seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS,
    train_limit: int = 1_000,
    sample_limit: int = 50,
    quantile: float = 0.05,
    n_estimators: int = 60,
) -> dict:
    """Estimate a held-out lower-tail threshold for RGB Tree SHAP maps."""
    if len(training_seeds) < 3:
        raise ValueError("CIFAR-10 calibration requires at least three training seeds.")
    if len(validation_seeds) < 2:
        raise ValueError("CIFAR-10 calibration requires at least two validation seeds.")
    if set(training_seeds) & set(validation_seeds):
        raise ValueError("Calibration and validation seeds must be disjoint.")
    if train_limit < 100:
        raise ValueError("CIFAR-10 calibration requires at least 100 training images.")
    if not 20 <= sample_limit <= 100:
        raise ValueError("CIFAR-10 calibration sample limit must be 20 to 100.")
    if not 0.0 < quantile < 0.5:
        raise ValueError("CIFAR-10 calibration quantile must be between 0 and 0.5.")
    if n_estimators < 1:
        raise ValueError("CIFAR-10 calibration requires at least one tree.")

    load_limit = max(train_limit, sample_limit)
    train_images, train_labels, test_images, test_labels, class_names = (
        _load_cifar10_python_splits(input_path, limit=load_limit)
    )
    train_images = train_images[:train_limit]
    train_labels = train_labels[:train_limit]
    test_images = test_images[:sample_limit]
    test_labels = test_labels[:sample_limit]
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
            "class_names": class_names,
            "trusted_model_files": True,
            "dataset_note": (
                "Official CIFAR-10 test subset for benign-retraining calibration."
            ),
        }
    )

    def train_models(seeds: tuple[int, ...]) -> list[RandomForestClassifier]:
        return [
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                n_jobs=1,
            ).fit(train_features, train_labels)
            for seed in seeds
        ]

    calibration_models = train_models(training_seeds)
    calibration_values: list[float] = []
    calibration_pairs = []
    for left in range(len(calibration_models)):
        for right in range(left + 1, len(calibration_models)):
            comparison = compare_classifiers(
                calibration_models[left],
                calibration_models[right],
                dataset,
                manifest,
            )
            pair_values = _pair_similarities(comparison)
            if not pair_values:
                raise ValueError("Calibration produced no same-prediction explanations.")
            calibration_values.extend(pair_values)
            calibration_pairs.append(
                {
                    "left_seed": training_seeds[left],
                    "right_seed": training_seeds[right],
                    "prediction_agreement": comparison["summary"][
                        "prediction_agreement"
                    ],
                    "same_prediction_comparisons": len(pair_values),
                    "mean_similarity": float(np.mean(pair_values)),
                    "p05_similarity": float(np.quantile(pair_values, 0.05)),
                }
            )
    values = np.asarray(calibration_values)
    threshold = float(np.quantile(values, quantile))

    validation_models = train_models(validation_seeds)
    validation_values: list[float] = []
    validation_pairs = []
    for left in range(len(validation_models)):
        for right in range(left + 1, len(validation_models)):
            comparison = compare_classifiers(
                validation_models[left],
                validation_models[right],
                dataset,
                manifest,
            )
            pair_values = _pair_similarities(comparison)
            if not pair_values:
                raise ValueError("Validation produced no same-prediction explanations.")
            flags = int(np.count_nonzero(np.asarray(pair_values) < threshold))
            validation_values.extend(pair_values)
            validation_pairs.append(
                {
                    "left_seed": validation_seeds[left],
                    "right_seed": validation_seeds[right],
                    "prediction_agreement": comparison["summary"][
                        "prediction_agreement"
                    ],
                    "same_prediction_comparisons": len(pair_values),
                    "flag_count": flags,
                    "flag_rate": flags / len(pair_values),
                }
            )
    heldout = np.asarray(validation_values)
    heldout_flags = int(np.count_nonzero(heldout < threshold))
    report = {
        "dataset": "CIFAR-10 32x32 RGB official Python batches",
        "training_seeds": list(training_seeds),
        "validation_seeds": list(validation_seeds),
        "train_limit": train_limit,
        "sample_limit": sample_limit,
        "model_configuration": {
            "type": "RandomForestClassifier",
            "n_estimators": n_estimators,
        },
        "explanation": {
            "method": "Tree SHAP",
            "spatial_aggregation": "sum RGB channel contributions per pixel",
        },
        "calibration_quantile": quantile,
        "same_prediction_comparisons": len(values),
        "mean_similarity": float(values.mean()),
        "median_similarity": float(np.median(values)),
        "calibrated_threshold": threshold,
        "rate_below_0_80": float(np.mean(values < 0.80)),
        "pairs": calibration_pairs,
        "heldout_validation": {
            "pairs": validation_pairs,
            "same_prediction_comparisons": len(heldout),
            "flag_count": heldout_flags,
            "flag_rate": heldout_flags / len(heldout),
            "mean_similarity": float(heldout.mean()),
            "median_similarity": float(np.median(heldout)),
        },
        "interpretation": (
            "The threshold estimates the lower tail of benign same-configuration "
            "RGB explanation variation. It is a review baseline, not proof of "
            "harmful drift."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": destination, "report_path": report_path, **report}
