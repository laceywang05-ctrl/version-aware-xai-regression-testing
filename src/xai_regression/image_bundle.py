"""Prepare generic labelled image arrays for the local comparison dashboard."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def prepare_image_npz_bundle(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 500,
) -> dict[str, Path | int | tuple[int, ...]]:
    """Convert NPZ ``images`` and ``labels`` arrays into dashboard upload files."""
    if limit < 1 or limit > 5_000:
        raise ValueError("Image bundle limit must be between 1 and 5,000.")
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
    count = min(len(images), limit)
    if count == 0:
        raise ValueError("NPZ input contains no images.")
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
