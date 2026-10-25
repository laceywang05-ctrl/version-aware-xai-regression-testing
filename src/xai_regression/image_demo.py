"""Build a self-contained handwritten-digit upload demonstration."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = tuple(f"pixel_{index}" for index in range(64))


def _selected_indices(
    labels: np.ndarray,
    prediction_v1: np.ndarray,
    prediction_v2: np.ndarray,
) -> list[int]:
    """Select a compact gallery that emphasizes changed predictions."""
    correct_v1 = prediction_v1 == labels
    correct_v2 = prediction_v2 == labels
    groups = (
        (np.flatnonzero(correct_v1 & ~correct_v2), 10),
        (np.flatnonzero(~correct_v1 & correct_v2), 25),
        (np.flatnonzero(~correct_v1 & ~correct_v2), 10),
        (np.flatnonzero(correct_v1 & correct_v2), 20),
    )
    selected: list[int] = []
    for indices, limit in groups:
        selected.extend(int(value) for value in indices[:limit])
    return selected


def _png_bytes(image: np.ndarray) -> bytes:
    stream = BytesIO()
    mpimg.imsave(stream, image, cmap="gray", vmin=0, vmax=16, format="png")
    return stream.getvalue()


def build_digits_image_demo(output_dir: str | Path) -> dict[str, Path | int | float]:
    """Train four model versions and write a ready-to-upload image demo bundle."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    features = pd.DataFrame(digits.data, columns=FEATURE_COLUMNS)
    labels = np.asarray(digits.target)
    indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    train_features = features.iloc[train_indices]
    test_features = features.iloc[test_indices].reset_index(drop=True)
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    model_v1 = DecisionTreeClassifier(max_depth=3, random_state=11).fit(
        train_features, train_labels
    )
    model_v2 = RandomForestClassifier(
        n_estimators=80,
        random_state=29,
        n_jobs=1,
    ).fit(train_features, train_labels)
    model_v3 = ExtraTreesClassifier(
        n_estimators=80,
        random_state=37,
        n_jobs=1,
    ).fit(train_features, train_labels)
    model_v4 = KNeighborsClassifier(n_neighbors=5).fit(train_features, train_labels)
    prediction_v1 = model_v1.predict(test_features)
    prediction_v2 = model_v2.predict(test_features)
    selected = _selected_indices(test_labels, prediction_v1, prediction_v2)

    model_v1_path = destination / "model_v1.joblib"
    model_v2_path = destination / "model_v2.joblib"
    model_v3_path = destination / "model_v3_extra_trees.joblib"
    model_v4_path = destination / "model_v4_knn.joblib"
    csv_path = destination / "samples.csv"
    manifest_path = destination / "manifest.json"
    images_path = destination / "images.zip"
    joblib.dump(model_v1, model_v1_path)
    joblib.dump(model_v2, model_v2_path)
    joblib.dump(model_v3, model_v3_path)
    joblib.dump(model_v4, model_v4_path)

    sample_frame = test_features.iloc[selected].reset_index(drop=True).copy()
    sample_frame.insert(0, "sample_id", [f"digit-{value:03d}" for value in selected])
    sample_frame.insert(
        1,
        "image_filename",
        [f"digits/digit-{value:03d}.png" for value in selected],
    )
    sample_frame["target"] = test_labels[selected]
    sample_frame.to_csv(csv_path, index=False)

    with ZipFile(images_path, "w", compression=ZIP_DEFLATED) as archive:
        for selected_index in selected:
            original_index = test_indices[selected_index]
            archive.writestr(
                f"digits/digit-{selected_index:03d}.png",
                _png_bytes(digits.images[original_index]),
            )

    manifest = {
        "task": "classification",
        "label_column": "target",
        "id_column": "sample_id",
        "image_column": "image_filename",
        "image_shape": [8, 8],
        "dataset_note": (
            "Curated 60-sample demonstration subset enriched for changed and "
            "incorrect predictions; summary accuracy is not the full test-set accuracy."
        ),
        "feature_columns": list(FEATURE_COLUMNS),
        "class_names": {str(value): str(value) for value in range(10)},
        "trusted_model_files": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "dataset": "scikit-learn handwritten digits (8x8)",
        "sample_count": len(selected),
        "models": [
            {"filename": model_v1_path.name, "model_type": "DecisionTreeClassifier", "test_accuracy": float(model_v1.score(test_features, test_labels))},
            {"filename": model_v2_path.name, "model_type": "RandomForestClassifier", "test_accuracy": float(model_v2.score(test_features, test_labels))},
            {"filename": model_v3_path.name, "model_type": "ExtraTreesClassifier", "test_accuracy": float(model_v3.score(test_features, test_labels))},
            {"filename": model_v4_path.name, "model_type": "KNeighborsClassifier", "test_accuracy": float(model_v4.score(test_features, test_labels))},
        ],
        "model_v1": "DecisionTreeClassifier(max_depth=3)",
        "model_v2": "RandomForestClassifier(n_estimators=80)",
        "full_test_v1_accuracy": float(np.mean(prediction_v1 == test_labels)),
        "full_test_v2_accuracy": float(np.mean(prediction_v2 == test_labels)),
        "files": {
            "model_v1": model_v1_path.name,
            "model_v2": model_v2_path.name,
            "model_v3": model_v3_path.name,
            "model_v4": model_v4_path.name,
            "dataset": csv_path.name,
            "manifest": manifest_path.name,
            "images": images_path.name,
        },
    }
    summary_path = destination / "README.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": destination,
        "sample_count": len(selected),
        "model_count": 4,
        "v1_accuracy": summary["full_test_v1_accuracy"],
        "v2_accuracy": summary["full_test_v2_accuracy"],
        "best_accuracy": max(model["test_accuracy"] for model in summary["models"]),
        "summary_path": summary_path,
    }
