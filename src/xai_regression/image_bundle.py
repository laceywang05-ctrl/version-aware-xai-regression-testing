"""Prepare generic labelled image arrays for the local comparison dashboard."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def _load_image_npz(input_path: str | Path) -> tuple[Path, np.ndarray, np.ndarray]:
    source = Path(input_path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "images" not in archive or "labels" not in archive:
                raise ValueError("NPZ input must contain images and labels arrays.")
            images = np.asarray(archive["images"])
            labels = np.asarray(archive["labels"])
    except OSError as error:
        raise ValueError(f"Could not read NPZ input: {source}.") from error
    if images.ndim not in {3, 4}:
        raise ValueError("Images must have shape (N,H,W) or (N,H,W,C).")
    if images.ndim == 4 and images.shape[-1] not in {1, 3, 4}:
        raise ValueError("Image channels must be 1, 3 or 4.")
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("Labels must be one-dimensional with one value per image.")
    if len(images) == 0:
        raise ValueError("NPZ input contains no images.")
    return source, images, labels


def prepare_image_npz_bundle(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 500,
) -> dict[str, Path | int | tuple[int, ...]]:
    """Convert NPZ ``images`` and ``labels`` arrays into dashboard upload files."""
    if limit < 1 or limit > 5_000:
        raise ValueError("Image bundle limit must be between 1 and 5,000.")
    source, images, labels = _load_image_npz(input_path)
    count = min(len(images), limit)
    images = images[:count]
    labels = labels[:count]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    flat = images.reshape(count, -1)
    feature_columns = [f"pixel_{index}" for index in range(flat.shape[1])]
    frame = pd.DataFrame(flat, columns=feature_columns)
    frame.insert(0, "sample_id", [f"image-{index:05d}" for index in range(count)])
    frame.insert(1, "image_filename", [f"images/image-{index:05d}.png" for index in range(count)])
    frame["target"] = labels
    csv_path = destination / "samples.csv"
    images_path = destination / "images.zip"
    manifest_path = destination / "manifest.json"
    frame.to_csv(csv_path, index=False)
    with ZipFile(images_path, "w", compression=ZIP_DEFLATED) as output:
        for index, image in enumerate(images):
            stream = BytesIO()
            rendered = image[..., 0] if image.ndim == 3 and image.shape[-1] == 1 else image
            options = {"cmap": "gray"} if rendered.ndim == 2 else {}
            mpimg.imsave(stream, rendered, format="png", **options)
            output.writestr(f"images/image-{index:05d}.png", stream.getvalue())
    class_values = sorted({_json_label(value) for value in labels})
    manifest = {
        "task": "classification",
        "label_column": "target",
        "id_column": "sample_id",
        "image_column": "image_filename",
        "feature_columns": feature_columns,
        "class_names": {str(value): str(value) for value in class_values},
        "trusted_model_files": True,
        "dataset_note": f"Prepared from {source.name}; first {count} labelled images.",
    }
    if images.ndim == 3:
        manifest["image_shape"] = [int(images.shape[1]), int(images.shape[2])]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": destination,
        "sample_count": count,
        "image_shape": tuple(images.shape[1:]),
        "feature_count": flat.shape[1],
    }


def _json_label(value):
    return value.item() if isinstance(value, np.generic) else value


def build_image_model_demo(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 1_000,
    test_size: float = 0.30,
) -> dict[str, Path | int | float]:
    """Train four lightweight classifiers and create an upload-ready image bundle."""
    if limit < 20 or limit > 5_000:
        raise ValueError("Image demo limit must be between 20 and 5,000.")
    source, images, labels = _load_image_npz(input_path)
    images = images[:limit]
    labels = labels[:limit]
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        raise ValueError("Image demo requires at least two classes with two samples each.")
    indices = np.arange(len(images))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=labels
    )
    feature_columns = [f"pixel_{index}" for index in range(images[0].size)]
    train_features = pd.DataFrame(
        images[train_indices].reshape(len(train_indices), -1), columns=feature_columns
    )
    test_features = pd.DataFrame(
        images[test_indices].reshape(len(test_indices), -1), columns=feature_columns
    )
    models = [
        ("model_1_dummy.joblib", DummyClassifier(strategy="most_frequent")),
        ("model_2_tree.joblib", DecisionTreeClassifier(max_depth=5, random_state=11)),
        ("model_3_random_forest.joblib", RandomForestClassifier(n_estimators=60, random_state=29, n_jobs=1)),
        ("model_4_extra_trees.joblib", ExtraTreesClassifier(n_estimators=60, random_state=37, n_jobs=1)),
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scores = []
    for filename, model in models:
        model.fit(train_features, labels[train_indices])
        scores.append(float(model.score(test_features, labels[test_indices])))
        joblib.dump(model, destination / filename)
    temporary = destination / "test-data.npz"
    np.savez(temporary, images=images[test_indices], labels=labels[test_indices])
    prepared = prepare_image_npz_bundle(temporary, destination, limit=len(test_indices))
    temporary.unlink()
    summary = {
        "source": source.name,
        "training_samples": len(train_indices),
        "test_samples": len(test_indices),
        "models": [
            {"filename": filename, "model_type": type(model).__name__, "test_accuracy": score}
            for (filename, model), score in zip(models, scores)
        ],
        "upload_files": {"dataset": "samples.csv", "images": "images.zip", "manifest": "manifest.json"},
    }
    (destination / "README.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": destination,
        "model_count": len(models),
        "training_samples": len(train_indices),
        "test_samples": prepared["sample_count"],
        "best_accuracy": max(scores),
    }
