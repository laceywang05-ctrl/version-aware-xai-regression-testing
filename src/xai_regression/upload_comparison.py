"""Validate and compare two user-supplied classification model versions."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import base64
from pathlib import PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import joblib
import numpy as np
import pandas as pd


MAX_DATASET_ROWS = 5_000
MAX_IMAGE_COUNT = 1_000
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 25 * 1024 * 1024
MAX_EXPLANATION_ROWS = 500
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@dataclass(frozen=True)
class UploadManifest:
    """User-provided metadata required to run a classification comparison."""

    task: str
    label_column: str
    id_column: str | None
    image_column: str | None
    image_shape: tuple[int, int] | None
    dataset_note: str | None
    feature_columns: tuple[str, ...] | None
    class_names: dict[str, str]
    trusted_model_files: bool


def parse_manifest(payload: bytes | str | dict[str, Any]) -> UploadManifest:
    """Parse and validate the uploaded comparison manifest."""
    if isinstance(payload, bytes):
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Manifest must be valid UTF-8 JSON.") from error
    elif isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("Manifest must be valid JSON.") from error
    else:
        raw = payload
    if not isinstance(raw, dict):
        raise ValueError("Manifest root must be a JSON object.")
    task = str(raw.get("task", ""))
    if task != "classification":
        raise ValueError("The current upload workflow supports classification only.")
    label_column = str(raw.get("label_column", "")).strip()
    if not label_column:
        raise ValueError("Manifest requires a non-empty label_column.")
    id_column = raw.get("id_column")
    if id_column is not None:
        id_column = str(id_column).strip() or None
    image_column = raw.get("image_column")
    if image_column is not None:
        image_column = str(image_column).strip() or None
    shape = raw.get("image_shape")
    image_shape = None
    if shape is not None:
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(value, int) or value < 1 for value in shape)
        ):
            raise ValueError("image_shape must contain two positive integers.")
        image_shape = (shape[0], shape[1])
    features = raw.get("feature_columns")
    if features is not None:
        if not isinstance(features, list) or not features:
            raise ValueError("feature_columns must be a non-empty JSON list.")
        feature_columns = tuple(str(value) for value in features)
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature_columns cannot contain duplicates.")
    else:
        feature_columns = None
    names = raw.get("class_names", {})
    if not isinstance(names, dict):
        raise ValueError("class_names must be a JSON object.")
    trusted = raw.get("trusted_model_files") is True
    if not trusted:
        raise ValueError(
            "Set trusted_model_files to true only for model files you trust."
        )
    return UploadManifest(
        task=task,
        label_column=label_column,
        id_column=id_column,
        image_column=image_column,
        image_shape=image_shape,
        dataset_note=(str(raw["dataset_note"]).strip() or None) if raw.get("dataset_note") else None,
        feature_columns=feature_columns,
        class_names={str(key): str(value) for key, value in names.items()},
        trusted_model_files=trusted,
    )


def load_uploaded_model(payload: bytes, filename: str) -> Any:
    """Load one trusted joblib model after enforcing the prototype format."""
    if not filename.lower().endswith(".joblib"):
        raise ValueError("Uploaded models must use the .joblib format.")
    try:
        model = joblib.load(BytesIO(payload))
    except Exception as error:
        raise ValueError(f"Could not load model file {filename}.") from error
    if not callable(getattr(model, "predict", None)):
        raise ValueError(f"Model {filename} does not provide predict().")
    return model


def load_uploaded_dataset(payload: bytes, filename: str) -> pd.DataFrame:
    """Load a bounded CSV dataset from memory."""
    if not filename.lower().endswith(".csv"):
        raise ValueError("The uploaded dataset must use the .csv format.")
    try:
        frame = pd.read_csv(BytesIO(payload))
    except Exception as error:
        raise ValueError(f"Could not read dataset file {filename}.") from error
    if frame.empty:
        raise ValueError("The uploaded dataset contains no rows.")
    if len(frame) > MAX_DATASET_ROWS:
        raise ValueError(
            f"The local prototype accepts at most {MAX_DATASET_ROWS} rows."
        )
    return frame


def load_uploaded_images(payload: bytes, filename: str) -> dict[str, str]:
    """Read a bounded archive of display images into data URLs."""
    if not filename.lower().endswith(".zip"):
        raise ValueError("The optional image archive must use the .zip format.")
    images: dict[str, str] = {}
    total_bytes = 0
    try:
        with ZipFile(BytesIO(payload)) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(entries) > MAX_IMAGE_COUNT:
                raise ValueError(
                    f"The image archive accepts at most {MAX_IMAGE_COUNT} files."
                )
            for entry in entries:
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("The image archive contains an unsafe path.")
                suffix = path.suffix.lower()
                media_type = IMAGE_MEDIA_TYPES.get(suffix)
                if media_type is None:
                    raise ValueError(
                        f"Unsupported image type in archive: {entry.filename}."
                    )
                if entry.file_size > MAX_IMAGE_BYTES:
                    raise ValueError(
                        f"Image {entry.filename} exceeds the 2 MB limit."
                    )
                total_bytes += entry.file_size
                if total_bytes > MAX_TOTAL_IMAGE_BYTES:
                    raise ValueError("Uncompressed images exceed the 25 MB limit.")
                content = archive.read(entry)
                encoded = base64.b64encode(content).decode("ascii")
                images[path.as_posix()] = f"data:{media_type};base64,{encoded}"
    except BadZipFile as error:
        raise ValueError(f"Could not read image archive {filename}.") from error
    if not images:
        raise ValueError("The uploaded image archive contains no images.")
    return images


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _label(value: Any, names: dict[str, str]) -> str:
    plain = _json_value(value)
    return names.get(str(plain), str(plain))


def _confidences(model: Any, features: pd.DataFrame) -> list[float | None]:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return [None] * len(features)
    probabilities = np.asarray(predict_proba(features))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
        raise ValueError("Model predict_proba() returned an unexpected shape.")
    return [float(value) for value in probabilities.max(axis=1)]


def _tree_shap_explanations(
    model: Any,
    features: pd.DataFrame,
    predictions: np.ndarray,
    image_shape: tuple[int, int] | None,
) -> tuple[list[list[list[float]] | None], dict[str, str]]:
    """Return normalized Tree SHAP maps for each model prediction when possible."""
    if image_shape is None:
        return [None] * len(features), {
            "method": "unavailable",
            "label": "Unavailable",
            "note": "Add image_shape to request pixel explanations.",
        }
    if image_shape[0] * image_shape[1] != features.shape[1]:
        raise ValueError(
            "image_shape must contain exactly one cell per selected feature."
        )
    try:
        import shap

        explained_features = features.iloc[:MAX_EXPLANATION_ROWS]
        explained_predictions = predictions[:MAX_EXPLANATION_ROWS]
        raw_values = np.asarray(
            shap.TreeExplainer(model).shap_values(explained_features)
        )
        if raw_values.ndim == 2:
            selected = raw_values
        elif raw_values.ndim == 3:
            classes = list(getattr(model, "classes_", []))
            if not classes:
                raise ValueError("Model does not expose classes_ for SHAP selection.")
            class_indices = np.asarray(
                [classes.index(value) for value in explained_predictions]
            )
            if raw_values.shape[1] == features.shape[1]:
                selected = raw_values[
                    np.arange(len(explained_features)), :, class_indices
                ]
            elif raw_values.shape[2] == features.shape[1]:
                selected = raw_values[
                    np.arange(len(explained_features)), class_indices, :
                ]
            else:
                raise ValueError("Unexpected Tree SHAP output shape.")
        else:
            raise ValueError("Unexpected Tree SHAP output dimensions.")
        maps: list[list[list[float]] | None] = []
        for row in selected:
            maximum = float(np.max(np.abs(row)))
            normalized = row / maximum if maximum > 0 else np.zeros_like(row)
            maps.append(normalized.reshape(image_shape).round(5).tolist())
        maps.extend([None] * (len(features) - len(maps)))
        return maps, {
            "method": "tree_shap",
            "label": "Tree SHAP",
            "note": (
                "Tree SHAP contribution to the predicted class; "
                f"limited to the first {MAX_EXPLANATION_ROWS} rows."
            ),
        }
    except Exception as error:
        return [None] * len(features), {
            "method": "unavailable",
            "label": "Unavailable",
            "note": f"Tree SHAP unavailable: {error}",
        }


def _occlusion_explanations(
    model: Any,
    features: pd.DataFrame,
    predictions: np.ndarray,
    image_shape: tuple[int, int] | None,
) -> tuple[list[list[list[float]] | None], dict[str, str]]:
    """Explain predicted-class probability by zeroing one pixel at a time."""
    unavailable = {
        "method": "unavailable",
        "label": "Unavailable",
        "note": "Pixel occlusion requires predict_proba(), classes_, and image_shape.",
    }
    if image_shape is None:
        return [None] * len(features), unavailable
    predict_proba = getattr(model, "predict_proba", None)
    classes = list(getattr(model, "classes_", []))
    if not callable(predict_proba) or not classes:
        return [None] * len(features), unavailable
    try:
        explained = features.iloc[:MAX_EXPLANATION_ROWS]
        explained_predictions = predictions[:MAX_EXPLANATION_ROWS]
        feature_count = explained.shape[1]
        repeated = np.repeat(explained.to_numpy(), feature_count, axis=0)
        feature_indices = np.tile(np.arange(feature_count), len(explained))
        repeated[np.arange(len(repeated)), feature_indices] = 0.0
        occluded = pd.DataFrame(repeated, columns=explained.columns)
        original_probabilities = np.asarray(predict_proba(explained))
        occluded_probabilities = np.asarray(predict_proba(occluded))
        maps: list[list[list[float]] | None] = []
        for row_index, prediction in enumerate(explained_predictions):
            class_index = classes.index(prediction)
            start = row_index * feature_count
            stop = start + feature_count
            contribution = (
                original_probabilities[row_index, class_index]
                - occluded_probabilities[start:stop, class_index]
            )
            maximum = float(np.max(np.abs(contribution)))
            normalized = (
                contribution / maximum
                if maximum > 0
                else np.zeros_like(contribution)
            )
            maps.append(normalized.reshape(image_shape).round(5).tolist())
        maps.extend([None] * (len(features) - len(maps)))
        return maps, {
            "method": "pixel_occlusion",
            "label": "Pixel occlusion",
            "note": (
                "Predicted-class probability change after replacing one pixel "
                "feature with zero."
            ),
        }
    except Exception as error:
        return [None] * len(features), {
            **unavailable,
            "note": f"Pixel occlusion unavailable: {error}",
        }


def _model_explanations(
    model: Any,
    features: pd.DataFrame,
    predictions: np.ndarray,
    image_shape: tuple[int, int] | None,
) -> tuple[list[list[list[float]] | None], dict[str, str]]:
    """Choose the most suitable installed explanation adapter."""
    maps, metadata = _tree_shap_explanations(
        model, features, predictions, image_shape
    )
    if metadata["method"] == "tree_shap":
        return maps, metadata
    fallback_maps, fallback_metadata = _occlusion_explanations(
        model, features, predictions, image_shape
    )
    if fallback_metadata["method"] != "unavailable":
        fallback_metadata["note"] += f" Tree SHAP fallback reason: {metadata['note']}"
    return fallback_maps, fallback_metadata


def compare_classifiers(
    model_v1: Any,
    model_v2: Any,
    dataset: pd.DataFrame,
    manifest: UploadManifest,
    images: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a matched, sample-level classification comparison."""
    if manifest.label_column not in dataset:
        raise ValueError(
            f"Dataset is missing label column: {manifest.label_column}."
        )
    if manifest.id_column is not None and manifest.id_column not in dataset:
        raise ValueError(f"Dataset is missing id column: {manifest.id_column}.")
    if manifest.image_column is not None and manifest.image_column not in dataset:
        raise ValueError(
            f"Dataset is missing image column: {manifest.image_column}."
        )
    if images is not None and manifest.image_column is None:
        raise ValueError("Manifest requires image_column when an image ZIP is uploaded.")
    if manifest.image_column is not None and images is None:
        raise ValueError("Upload an image ZIP when manifest image_column is set.")
    excluded = {manifest.label_column}
    if manifest.id_column is not None:
        excluded.add(manifest.id_column)
    if manifest.image_column is not None:
        excluded.add(manifest.image_column)
    feature_columns = (
        list(manifest.feature_columns)
        if manifest.feature_columns is not None
        else [column for column in dataset.columns if column not in excluded]
    )
    missing = [column for column in feature_columns if column not in dataset]
    if missing:
        raise ValueError("Dataset is missing feature columns: " + ", ".join(missing))
    if not feature_columns:
        raise ValueError("No model feature columns were selected.")
    features = dataset.loc[:, feature_columns]
    labels = dataset[manifest.label_column].to_numpy()
    try:
        prediction_v1 = np.asarray(model_v1.predict(features))
        prediction_v2 = np.asarray(model_v2.predict(features))
    except Exception as error:
        raise ValueError(
            "A model could not predict this dataset. Check feature order and preprocessing."
        ) from error
    if prediction_v1.shape != labels.shape or prediction_v2.shape != labels.shape:
        raise ValueError("Model predictions must contain one value per dataset row.")
    confidence_v1 = _confidences(model_v1, features)
    confidence_v2 = _confidences(model_v2, features)
    explanation_v1, explanation_metadata_v1 = _model_explanations(
        model_v1, features, prediction_v1, manifest.image_shape
    )
    explanation_v2, explanation_metadata_v2 = _model_explanations(
        model_v2, features, prediction_v2, manifest.image_shape
    )
    correct_v1 = prediction_v1 == labels
    correct_v2 = prediction_v2 == labels
    identifiers = (
        dataset[manifest.id_column].tolist()
        if manifest.id_column is not None
        else list(range(len(dataset)))
    )
    transitions = np.select(
        [
            correct_v1 & correct_v2,
            correct_v1 & ~correct_v2,
            ~correct_v1 & correct_v2,
        ],
        ["both_correct", "v1_correct_v2_wrong", "v1_wrong_v2_correct"],
        default="both_wrong",
    )
    samples = []
    for index in range(len(dataset)):
        image_name = None
        image_data_url = None
        if manifest.image_column is not None:
            image_name = str(dataset.iloc[index][manifest.image_column])
            image_data_url = images.get(image_name) if images is not None else None
            if image_data_url is None:
                raise ValueError(f"Image referenced by CSV was not found: {image_name}.")
        samples.append(
            {
                "sample_id": _json_value(identifiers[index]),
                "actual": _label(labels[index], manifest.class_names),
                "v1_prediction": _label(
                    prediction_v1[index], manifest.class_names
                ),
                "v2_prediction": _label(
                    prediction_v2[index], manifest.class_names
                ),
                "v1_confidence": confidence_v1[index],
                "v2_confidence": confidence_v2[index],
                "v1_correct": bool(correct_v1[index]),
                "v2_correct": bool(correct_v2[index]),
                "prediction_changed": bool(
                    prediction_v1[index] != prediction_v2[index]
                ),
                "transition": str(transitions[index]),
                "image_name": image_name,
                "image_data_url": image_data_url,
                "v1_explanation": explanation_v1[index],
                "v2_explanation": explanation_v2[index],
            }
        )
    counts = {
        name: int(np.count_nonzero(transitions == name))
        for name in (
            "both_correct",
            "v1_correct_v2_wrong",
            "v1_wrong_v2_correct",
            "both_wrong",
        )
    }
    total = len(dataset)
    return {
        "schema_version": 1,
        "task": "classification",
        "model_format": "joblib",
        "dataset": {
            "rows": total,
            "feature_count": len(feature_columns),
            "label_column": manifest.label_column,
            "id_column": manifest.id_column,
            "note": manifest.dataset_note,
        },
        "summary": {
            "total_samples": total,
            "v1_accuracy": float(correct_v1.mean()),
            "v2_accuracy": float(correct_v2.mean()),
            "prediction_agreement": float((prediction_v1 == prediction_v2).mean()),
            "prediction_changed_count": int(
                np.count_nonzero(prediction_v1 != prediction_v2)
            ),
            **counts,
        },
        "samples": samples,
        "limitations": [
            "Only trusted local .joblib classifiers are supported in this stage.",
            "Uploaded files are processed in memory and are not persisted.",
            "Tree SHAP is preferred for compatible trees; other probabilistic classifiers use pixel occlusion.",
        ],
        "explanations": {
            "image_shape": list(manifest.image_shape) if manifest.image_shape else None,
            "v1": explanation_metadata_v1,
            "v2": explanation_metadata_v2,
            "interpretation": (
                "Red pixels support the displayed predicted class and blue pixels "
                "oppose it. Each map is normalized separately, so compare spatial "
                "patterns rather than color magnitude. Contributions are associations, "
                "not causal effects."
            ),
        },
    }


def compare_upload_payloads(
    *,
    model_v1: tuple[str, bytes],
    model_v2: tuple[str, bytes],
    dataset: tuple[str, bytes],
    manifest: bytes,
    images: tuple[str, bytes] | None = None,
) -> dict[str, Any]:
    """Load the four uploaded files and return their comparison report."""
    contract = parse_manifest(manifest)
    first = load_uploaded_model(model_v1[1], model_v1[0])
    second = load_uploaded_model(model_v2[1], model_v2[0])
    frame = load_uploaded_dataset(dataset[1], dataset[0])
    image_lookup = (
        load_uploaded_images(images[1], images[0]) if images is not None else None
    )
    result = compare_classifiers(first, second, frame, contract, image_lookup)
    result["models"] = {
        "v1": {"filename": model_v1[0]},
        "v2": {"filename": model_v2[0]},
    }
    return result
