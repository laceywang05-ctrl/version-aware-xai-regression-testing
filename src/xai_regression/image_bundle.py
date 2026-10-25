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


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _select_image_arrays(archive) -> tuple[np.ndarray, np.ndarray, str]:
    """Select a labelled image pair from generic or Keras-style NPZ keys."""
    keys = set(archive.files)
    if {"images", "labels"} <= keys:
        return np.asarray(archive["images"]), np.asarray(archive["labels"]), "images/labels"
    if {"x_test", "y_test"} <= keys:
        return np.asarray(archive["x_test"]), np.asarray(archive["y_test"]), "x_test/y_test"
    if {"x_train", "y_train"} <= keys:
        return np.asarray(archive["x_train"]), np.asarray(archive["y_train"]), "x_train/y_train"
    raise ValueError(
        "NPZ input must contain images/labels, x_test/y_test, or x_train/y_train arrays."
    )


def _normalise_labels(labels: np.ndarray) -> np.ndarray:
    """Accept the flat or single-column labels used by MNIST and CIFAR archives."""
    if labels.ndim == 2 and labels.shape[1] == 1:
        return labels.reshape(-1)
    return labels


def _validate_image_arrays(images: np.ndarray, labels: np.ndarray) -> None:
    labels = _normalise_labels(labels)
    if images.ndim not in {3, 4}:
        raise ValueError("Images must have shape (N,H,W) or (N,H,W,C).")
    if images.ndim == 4 and images.shape[-1] not in {1, 3, 4}:
        raise ValueError("Image channels must be 1, 3 or 4.")
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("Labels must be one-dimensional with one value per image.")
    if len(images) == 0:
        raise ValueError("NPZ input contains no images.")


def _load_image_npz(input_path: str | Path) -> tuple[Path, np.ndarray, np.ndarray, str]:
    source = Path(input_path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            images, labels, array_source = _select_image_arrays(archive)
    except OSError as error:
        raise ValueError(f"Could not read NPZ input: {source}.") from error
    labels = _normalise_labels(labels)
    _validate_image_arrays(images, labels)
    return source, images, labels, array_source


def _load_official_splits(
    input_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return official Keras-style train/test arrays when all four are present."""
    source = Path(input_path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if not {"x_train", "y_train", "x_test", "y_test"} <= set(archive.files):
                return None
            train_images = np.asarray(archive["x_train"])
            train_labels = _normalise_labels(np.asarray(archive["y_train"]))
            test_images = np.asarray(archive["x_test"])
            test_labels = _normalise_labels(np.asarray(archive["y_test"]))
    except OSError as error:
        raise ValueError(f"Could not read NPZ input: {source}.") from error
    _validate_image_arrays(train_images, train_labels)
    _validate_image_arrays(test_images, test_labels)
    if train_images.shape[1:] != test_images.shape[1:]:
        raise ValueError("Official train and test images must have the same shape.")
    return train_images, train_labels, test_images, test_labels


def _image_class_names(input_path: str | Path, labels: np.ndarray) -> tuple[dict[str, str], str]:
    """Load explicit class names or infer well-known MNIST/CIFAR-10 labels."""
    source = Path(input_path)
    explicit_names = None
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "class_names" in archive:
                explicit_names = [str(value) for value in np.asarray(archive["class_names"]).reshape(-1)]
    except OSError as error:
        raise ValueError(f"Could not read NPZ input: {source}.") from error
    class_values = sorted({_json_label(value) for value in labels})
    if explicit_names is not None:
        if not all(isinstance(value, (int, np.integer)) for value in class_values):
            raise ValueError("NPZ class_names require integer labels starting at zero.")
        if class_values and max(class_values) >= len(explicit_names):
            raise ValueError("NPZ class_names do not cover every numeric label.")
        return {str(value): explicit_names[int(value)] for value in class_values}, "NPZ class_names"
    lowered_name = source.name.lower().replace("-", "").replace("_", "")
    if "cifar10" in lowered_name and set(class_values) <= set(range(10)):
        return {str(value): CIFAR10_CLASS_NAMES[int(value)] for value in class_values}, "CIFAR-10"
    if ("mnist" in lowered_name or "digit" in lowered_name) and set(class_values) <= set(range(10)):
        return {str(value): f"digit {value}" for value in class_values}, "MNIST/digits"
    return {str(value): str(value) for value in class_values}, "numeric labels"


def prepare_image_npz_bundle(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 500,
    class_names: dict[str, str] | None = None,
) -> dict[str, Path | int | str | tuple[int, ...]]:
    """Convert generic or Keras-style NPZ image arrays into dashboard files."""
    if limit < 1 or limit > 5_000:
        raise ValueError("Image bundle limit must be between 1 and 5,000.")
    source, images, labels, array_source = _load_image_npz(input_path)
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
    inferred_class_names, class_name_source = _image_class_names(source, labels)
    resolved_class_names = class_names or inferred_class_names
    if class_names is not None:
        class_name_source = "source dataset"
    manifest = {
        "task": "classification",
        "label_column": "target",
        "id_column": "sample_id",
        "image_column": "image_filename",
        "feature_columns": feature_columns,
        "class_names": resolved_class_names,
        "trusted_model_files": True,
        "dataset_note": (
            f"Prepared from {source.name} ({array_source}); first {count} labelled images."
        ),
    }
    if images.ndim == 3:
        manifest["image_shape"] = [int(images.shape[1]), int(images.shape[2])]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": destination,
        "sample_count": count,
        "image_shape": tuple(images.shape[1:]),
        "feature_count": flat.shape[1],
        "array_source": array_source,
        "class_name_source": class_name_source,
    }


def _json_label(value):
    return value.item() if isinstance(value, np.generic) else value


def build_image_model_demo(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 1_000,
    test_size: float = 0.30,
) -> dict[str, Path | int | float | str]:
    """Train four lightweight classifiers and create an upload-ready image bundle."""
    if limit < 20 or limit > 5_000:
        raise ValueError("Image demo limit must be between 20 and 5,000.")
    source = Path(input_path)
    official_splits = _load_official_splits(input_path)
    if official_splits is not None:
        train_images, train_labels, test_images, test_labels = official_splits
        train_images = train_images[:limit]
        train_labels = train_labels[:limit]
        test_images = test_images[:limit]
        test_labels = test_labels[:limit]
        split_strategy = "official x_train/y_train and x_test/y_test"
    else:
        source, images, labels, _ = _load_image_npz(input_path)
        images = images[:limit]
        labels = labels[:limit]
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2 or np.any(counts < 2):
            raise ValueError("Image demo requires at least two classes with two samples each.")
        indices = np.arange(len(images))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=labels
        )
        train_images = images[train_indices]
        train_labels = labels[train_indices]
        test_images = images[test_indices]
        test_labels = labels[test_indices]
        split_strategy = f"fixed stratified split ({1 - test_size:.0%}/{test_size:.0%})"
    source_class_names, class_name_source = _image_class_names(source, test_labels)
    classes, counts = np.unique(train_labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        raise ValueError("Image demo requires at least two classes with two samples each.")
    feature_columns = [f"pixel_{index}" for index in range(train_images[0].size)]
    train_features = pd.DataFrame(
        train_images.reshape(len(train_images), -1), columns=feature_columns
    )
    test_features = pd.DataFrame(
        test_images.reshape(len(test_images), -1), columns=feature_columns
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
        model.fit(train_features, train_labels)
        scores.append(float(model.score(test_features, test_labels)))
        joblib.dump(model, destination / filename)
    temporary = destination / "test-data.npz"
    np.savez(temporary, images=test_images, labels=test_labels)
    prepared = prepare_image_npz_bundle(
        temporary,
        destination,
        limit=len(test_images),
        class_names=source_class_names,
    )
    temporary.unlink()
    summary = {
        "source": source.name,
        "split_strategy": split_strategy,
        "class_name_source": class_name_source,
        "training_samples": len(train_images),
        "test_samples": len(test_images),
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
        "training_samples": len(train_images),
        "test_samples": prepared["sample_count"],
        "best_accuracy": max(scores),
        "split_strategy": split_strategy,
    }
