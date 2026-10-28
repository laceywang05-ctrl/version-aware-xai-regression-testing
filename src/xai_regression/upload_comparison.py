"""Validate and compare two user-supplied classification model versions."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import base64
from math import comb
from pathlib import PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from xai_regression.provenance import artifact_fingerprint


MAX_DATASET_ROWS = 5_000
MAX_IMAGE_COUNT = 1_000
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 25 * 1024 * 1024
MAX_EXPLANATION_ROWS = 500
MAX_RGB_EXPLANATION_ROWS = 100
MAX_RGB_OCCLUSION_ROWS = 20
MAX_NEURAL_EXPLANATION_ROWS = 100
HIGH_CONFIDENCE_THRESHOLD = 0.80
EXPLANATION_SIMILARITY_THRESHOLD = 0.80
MNIST_EXPLANATION_SIMILARITY_THRESHOLD = 0.486
CIFAR10_EXPLANATION_SIMILARITY_THRESHOLD = 0.04665091646
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
    image_shape: tuple[int, ...] | None
    dataset_note: str | None
    feature_columns: tuple[str, ...] | None
    class_names: dict[str, str]
    trusted_model_files: bool


def _explanation_threshold(
    manifest: UploadManifest,
    methods: tuple[str, str] | None = None,
) -> tuple[float, str, bool]:
    """Select a threshold only when calibration matches the explanation method."""
    if methods != ("tree_shap", "tree_shap"):
        method_label = " / ".join(methods or ("unknown",))
        return (
            EXPLANATION_SIMILARITY_THRESHOLD,
            f"Exploratory default (not calibrated for {method_label})",
            False,
        )
    if manifest.image_shape in {(28, 28), (28, 28, 1)}:
        return (
            MNIST_EXPLANATION_SIMILARITY_THRESHOLD,
            "MNIST Tree SHAP benign-retraining calibration",
            True,
        )
    if manifest.image_shape == (32, 32, 3):
        return (
            CIFAR10_EXPLANATION_SIMILARITY_THRESHOLD,
            "CIFAR-10 RGB Tree SHAP benign-retraining calibration",
            True,
        )
    return (
        EXPLANATION_SIMILARITY_THRESHOLD,
        "Exploratory default (not calibrated for this dataset)",
        False,
    )


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
            or len(shape) not in {2, 3}
            or any(not isinstance(value, int) or value < 1 for value in shape)
        ):
            raise ValueError("image_shape must contain two or three positive integers.")
        if len(shape) == 3 and shape[2] not in {1, 3, 4}:
            raise ValueError("RGB image_shape channels must be 1, 3 or 4.")
        image_shape = tuple(shape)
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


def _calibration_summary(
    model: Any,
    features: pd.DataFrame,
    predictions: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> dict[str, Any]:
    """Summarize top-label confidence calibration with equal-width bins."""
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return {
            "supports_confidence": False,
            "mean_confidence": None,
            "confidence_gap": None,
            "expected_calibration_error": None,
            "calibration_bins": [],
        }
    try:
        probabilities = np.asarray(predict_proba(features), dtype=float)
    except Exception:
        return {
            "supports_confidence": False,
            "mean_confidence": None,
            "confidence_gap": None,
            "expected_calibration_error": None,
            "calibration_bins": [],
        }
    if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
        raise ValueError("Model predict_proba() returned an unexpected shape.")
    confidence = probabilities.max(axis=1)
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Model predict_proba() returned invalid probabilities.")
    correct = predictions == labels
    bin_indices = np.minimum((confidence * bins).astype(int), bins - 1)
    expected_calibration_error = 0.0
    calibration_bins = []
    for bin_index in range(bins):
        mask = bin_indices == bin_index
        count = int(np.count_nonzero(mask))
        if count:
            bin_accuracy = float(correct[mask].mean())
            bin_confidence = float(confidence[mask].mean())
            expected_calibration_error += (count / len(confidence)) * abs(
                bin_accuracy - bin_confidence
            )
            calibration_bins.append(
                {
                    "lower": bin_index / bins,
                    "upper": (bin_index + 1) / bins,
                    "samples": count,
                    "mean_confidence": bin_confidence,
                    "accuracy": bin_accuracy,
                    "gap": bin_confidence - bin_accuracy,
                }
            )
    accuracy = float(correct.mean())
    mean_confidence = float(confidence.mean())
    return {
        "supports_confidence": True,
        "mean_confidence": mean_confidence,
        "confidence_gap": mean_confidence - accuracy,
        "expected_calibration_error": float(expected_calibration_error),
        "calibration_bins": calibration_bins,
    }


def _tree_shap_explanations(
    model: Any,
    features: pd.DataFrame,
    predictions: np.ndarray,
    image_shape: tuple[int, ...] | None,
) -> tuple[list[list[list[float]] | None], dict[str, str]]:
    """Return normalized Tree SHAP maps for each model prediction when possible."""
    if image_shape is None:
        return [None] * len(features), {
            "method": "unavailable",
            "label": "Unavailable",
            "note": "Add image_shape to request pixel explanations.",
        }
    if int(np.prod(image_shape)) != features.shape[1]:
        raise ValueError(
            "image_shape must contain exactly one cell per selected feature."
        )
    try:
        import shap

        row_limit = (
            MAX_RGB_EXPLANATION_ROWS
            if len(image_shape) == 3
            else MAX_EXPLANATION_ROWS
        )
        explained_features = features.iloc[:row_limit]
        explained_predictions = predictions[:row_limit]
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
            spatial = np.asarray(row).reshape(image_shape)
            if len(image_shape) == 3:
                spatial = spatial.sum(axis=2)
            maximum = float(np.max(np.abs(spatial)))
            normalized = (
                spatial / maximum if maximum > 0 else np.zeros_like(spatial)
            )
            maps.append(normalized.round(5).tolist())
        maps.extend([None] * (len(features) - len(maps)))
        return maps, {
            "method": "tree_shap",
            "label": "Tree SHAP",
            "note": (
                "Tree SHAP contribution to the predicted class; "
                f"limited to the first {row_limit} rows."
                + (
                    " RGB channel contributions are summed per spatial pixel."
                    if len(image_shape) == 3
                    else ""
                )
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
    image_shape: tuple[int, ...] | None,
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
    if int(np.prod(image_shape)) != features.shape[1]:
        raise ValueError(
            "image_shape must contain exactly one cell per selected feature."
        )
    try:
        is_rgb = len(image_shape) == 3
        row_limit = MAX_RGB_OCCLUSION_ROWS if is_rgb else MAX_EXPLANATION_ROWS
        explained = features.iloc[:row_limit]
        explained_predictions = predictions[:row_limit]
        channel_count = image_shape[2] if is_rgb else 1
        spatial_count = image_shape[0] * image_shape[1]
        original_probabilities = np.asarray(predict_proba(explained))
        maps: list[list[list[float]] | None] = []
        for row_index, prediction in enumerate(explained_predictions):
            class_index = classes.index(prediction)
            repeated = np.repeat(
                explained.iloc[[row_index]].to_numpy(), spatial_count, axis=0
            )
            for pixel_index in range(spatial_count):
                start = pixel_index * channel_count
                repeated[pixel_index, start : start + channel_count] = 0.0
            occluded = pd.DataFrame(repeated, columns=explained.columns)
            occluded_probabilities = np.asarray(predict_proba(occluded))
            contribution = (
                original_probabilities[row_index, class_index]
                - occluded_probabilities[:, class_index]
            )
            maximum = float(np.max(np.abs(contribution)))
            normalized = (
                contribution / maximum
                if maximum > 0
                else np.zeros_like(contribution)
            )
            maps.append(
                normalized.reshape(image_shape[:2]).round(5).tolist()
            )
        maps.extend([None] * (len(features) - len(maps)))
        return maps, {
            "method": "pixel_occlusion",
            "label": "Pixel occlusion",
            "note": (
                "Predicted-class probability change after replacing one pixel "
                "with zero"
                + (" across all RGB channels." if is_rgb else ".")
                + f" Limited to the first {row_limit} rows."
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
    image_shape: tuple[int, ...] | None,
) -> tuple[list[list[list[float]] | None], dict[str, str]]:
    """Choose the most suitable installed explanation adapter."""
    row_limit = min(len(features), MAX_NEURAL_EXPLANATION_ROWS)
    grad_cam = getattr(model, "grad_cam", None)
    if callable(grad_cam) and image_shape is not None:
        try:
            maps = list(grad_cam(features.iloc[:row_limit], predictions[:row_limit]))
            maps.extend([None] * (len(features) - len(maps)))
            return maps, {
                "method": "grad_cam",
                "label": "Grad-CAM",
                "note": (
                    "Predicted-class Grad-CAM at the final convolutional layer; "
                    f"limited to the first {row_limit} rows."
                ),
            }
        except Exception as grad_cam_error:
            grad_cam_note = f"Grad-CAM unavailable: {grad_cam_error}"
    else:
        grad_cam_note = "Grad-CAM requires a convolutional neural-network adapter."
    integrated_gradients = getattr(model, "integrated_gradients", None)
    if callable(integrated_gradients) and image_shape is not None:
        try:
            maps = list(
                integrated_gradients(
                    features.iloc[:row_limit], predictions[:row_limit]
                )
            )
            maps.extend([None] * (len(features) - len(maps)))
            return maps, {
                "method": "integrated_gradients",
                "label": "Integrated Gradients",
                "note": (
                    "Zero-baseline Integrated Gradients for the predicted class; "
                    f"limited to the first {row_limit} rows. {grad_cam_note}"
                ),
            }
        except Exception:
            pass
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


def _explanation_similarity(
    first: list[list[float]] | None,
    second: list[list[float]] | None,
) -> float | None:
    """Return cosine similarity for two same-shaped explanation maps."""
    if first is None or second is None:
        return None
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 and right_norm == 0:
        return 1.0
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))


def _exact_mcnemar_p(v1_correct_v2_wrong: int, v1_wrong_v2_correct: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant pairs."""
    if v1_correct_v2_wrong < 0 or v1_wrong_v2_correct < 0:
        raise ValueError("McNemar counts cannot be negative.")
    discordant = v1_correct_v2_wrong + v1_wrong_v2_correct
    if discordant == 0:
        return 1.0
    smaller = min(v1_correct_v2_wrong, v1_wrong_v2_correct)
    cumulative = sum(comb(discordant, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * cumulative / (2**discordant))


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
    explanation_threshold, explanation_threshold_source, threshold_calibrated = (
        _explanation_threshold(
            manifest,
            (
                explanation_metadata_v1["method"],
                explanation_metadata_v2["method"],
            ),
        )
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
        v1_high_confidence_wrong = bool(
            not correct_v1[index]
            and confidence_v1[index] is not None
            and confidence_v1[index] >= HIGH_CONFIDENCE_THRESHOLD
        )
        v2_high_confidence_wrong = bool(
            not correct_v2[index]
            and confidence_v2[index] is not None
            and confidence_v2[index] >= HIGH_CONFIDENCE_THRESHOLD
        )
        explanation_similarity = _explanation_similarity(
            explanation_v1[index], explanation_v2[index]
        )
        explanation_drift = bool(
            explanation_similarity is not None
            and explanation_similarity < explanation_threshold
        )
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
                "v1_high_confidence_wrong": v1_high_confidence_wrong,
                "v2_high_confidence_wrong": v2_high_confidence_wrong,
                "high_confidence_regression": bool(
                    transitions[index] == "v1_correct_v2_wrong"
                    and v2_high_confidence_wrong
                ),
                "explanation_similarity": explanation_similarity,
                "explanation_drift": explanation_drift,
                "hidden_explanation_drift": bool(
                    explanation_drift
                    and prediction_v1[index] == prediction_v2[index]
                ),
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
    mcnemar_p = _exact_mcnemar_p(
        counts["v1_correct_v2_wrong"], counts["v1_wrong_v2_correct"]
    )
    total = len(dataset)
    v1_high_confidence_wrong_count = sum(
        sample["v1_high_confidence_wrong"] for sample in samples
    )
    v2_high_confidence_wrong_count = sum(
        sample["v2_high_confidence_wrong"] for sample in samples
    )
    high_confidence_regression_count = sum(
        sample["high_confidence_regression"] for sample in samples
    )
    explanation_compared_count = sum(
        sample["explanation_similarity"] is not None for sample in samples
    )
    explanation_drift_count = sum(sample["explanation_drift"] for sample in samples)
    hidden_explanation_drift_count = sum(
        sample["hidden_explanation_drift"] for sample in samples
    )
    class_diagnostics = []
    for class_value in pd.unique(labels):
        class_mask = labels == class_value
        v1_predicted_mask = prediction_v1 == class_value
        v2_predicted_mask = prediction_v2 == class_value
        class_total = int(np.count_nonzero(class_mask))
        v1_true_positive = int(np.count_nonzero(class_mask & v1_predicted_mask))
        v2_true_positive = int(np.count_nonzero(class_mask & v2_predicted_mask))
        v1_class_accuracy = float(correct_v1[class_mask].mean())
        v2_class_accuracy = float(correct_v2[class_mask].mean())
        v1_precision = v1_true_positive / max(1, int(np.count_nonzero(v1_predicted_mask)))
        v2_precision = v2_true_positive / max(1, int(np.count_nonzero(v2_predicted_mask)))
        v1_f1 = (
            2 * v1_precision * v1_class_accuracy / (v1_precision + v1_class_accuracy)
            if v1_precision + v1_class_accuracy > 0
            else 0.0
        )
        v2_f1 = (
            2 * v2_precision * v2_class_accuracy / (v2_precision + v2_class_accuracy)
            if v2_precision + v2_class_accuracy > 0
            else 0.0
        )
        class_diagnostics.append(
            {
                "class_label": _label(class_value, manifest.class_names),
                "samples": class_total,
                "v1_correct": int(np.count_nonzero(correct_v1[class_mask])),
                "v2_correct": int(np.count_nonzero(correct_v2[class_mask])),
                "v1_accuracy": v1_class_accuracy,
                "v2_accuracy": v2_class_accuracy,
                "accuracy_change": v2_class_accuracy - v1_class_accuracy,
                "v1_precision": v1_precision,
                "v1_recall": v1_class_accuracy,
                "v1_f1": v1_f1,
                "v2_precision": v2_precision,
                "v2_recall": v2_class_accuracy,
                "v2_f1": v2_f1,
                "f1_change": v2_f1 - v1_f1,
            }
        )
    confusion_counts: dict[tuple[str, str], list[int]] = {}
    for index in range(total):
        actual_label = _label(labels[index], manifest.class_names)
        if not correct_v1[index]:
            key = (actual_label, _label(prediction_v1[index], manifest.class_names))
            confusion_counts.setdefault(key, [0, 0])[0] += 1
        if not correct_v2[index]:
            key = (actual_label, _label(prediction_v2[index], manifest.class_names))
            confusion_counts.setdefault(key, [0, 0])[1] += 1
    confusion_diagnostics = [
        {
            "actual_class": actual,
            "predicted_class": predicted,
            "v1_count": counts[0],
            "v2_count": counts[1],
            "error_change": counts[1] - counts[0],
        }
        for (actual, predicted), counts in confusion_counts.items()
    ]
    confusion_diagnostics.sort(
        key=lambda item: (
            -max(item["v1_count"], item["v2_count"]),
            item["actual_class"],
            item["predicted_class"],
        )
    )
    actual_named = np.asarray([_label(value, manifest.class_names) for value in labels])
    v1_named = np.asarray([_label(value, manifest.class_names) for value in prediction_v1])
    v2_named = np.asarray([_label(value, manifest.class_names) for value in prediction_v2])
    matrix_labels = sorted({*actual_named, *v1_named, *v2_named})

    def confusion_matrix(named_predictions: np.ndarray) -> list[list[int]]:
        return [
            [
                int(
                    np.count_nonzero(
                        (actual_named == actual_label)
                        & (named_predictions == predicted_label)
                    )
                )
                for predicted_label in matrix_labels
            ]
            for actual_label in matrix_labels
        ]
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
            "v1_macro_f1": float(f1_score(labels, prediction_v1, average="macro", zero_division=0)),
            "v2_macro_f1": float(f1_score(labels, prediction_v2, average="macro", zero_division=0)),
            "prediction_agreement": float((prediction_v1 == prediction_v2).mean()),
            "accuracy_change": float(correct_v2.mean() - correct_v1.mean()),
            "discordant_correctness_count": (
                counts["v1_correct_v2_wrong"] + counts["v1_wrong_v2_correct"]
            ),
            "mcnemar_exact_p": mcnemar_p,
            "mcnemar_alpha": 0.05,
            "mcnemar_significant": bool(mcnemar_p < 0.05),
            "prediction_changed_count": int(
                np.count_nonzero(prediction_v1 != prediction_v2)
            ),
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "v1_high_confidence_wrong_count": v1_high_confidence_wrong_count,
            "v2_high_confidence_wrong_count": v2_high_confidence_wrong_count,
            "high_confidence_regression_count": high_confidence_regression_count,
            "explanation_similarity_threshold": explanation_threshold,
            "explanation_similarity_threshold_source": explanation_threshold_source,
            "explanation_similarity_threshold_calibrated": threshold_calibrated,
            "explanation_compared_count": explanation_compared_count,
            "explanation_drift_count": explanation_drift_count,
            "hidden_explanation_drift_count": hidden_explanation_drift_count,
            **counts,
        },
        "class_diagnostics": class_diagnostics,
        "confusion_matrices": {
            "labels": matrix_labels,
            "v1": confusion_matrix(v1_named),
            "v2": confusion_matrix(v2_named),
        },
        "confusion_diagnostics": confusion_diagnostics,
        "samples": samples,
        "limitations": [
            "Only trusted local .joblib classifiers are supported in this stage; neural networks can be wrapped with TorchImageClassifier.",
            "Uploaded files are processed in memory and are not persisted.",
            "CNN adapters use Grad-CAM, other differentiable neural adapters use Integrated Gradients, compatible trees use Tree SHAP, and remaining probabilistic classifiers use pixel occlusion.",
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
        "v1": artifact_fingerprint(*model_v1),
        "v2": artifact_fingerprint(*model_v2),
    }
    result["artifacts"] = {
        "dataset": artifact_fingerprint(*dataset),
        "manifest": artifact_fingerprint("manifest.json", manifest),
    }
    if images is not None:
        result["artifacts"]["images"] = artifact_fingerprint(*images)
    return result


def summarize_upload_models(
    *,
    models: list[tuple[str, bytes]],
    dataset: tuple[str, bytes],
    manifest: bytes,
) -> dict[str, Any]:
    """Score two to five uploaded classifiers on one labelled dataset."""
    if not 2 <= len(models) <= 5:
        raise ValueError("Upload between two and five models for the overview.")
    contract = parse_manifest(manifest)
    frame = load_uploaded_dataset(dataset[1], dataset[0])
    if contract.label_column not in frame:
        raise ValueError(f"Dataset is missing label column: {contract.label_column}.")
    excluded = {contract.label_column}
    if contract.id_column is not None:
        excluded.add(contract.id_column)
    if contract.image_column is not None:
        excluded.add(contract.image_column)
    feature_columns = (
        list(contract.feature_columns)
        if contract.feature_columns is not None
        else [column for column in frame.columns if column not in excluded]
    )
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise ValueError("Dataset is missing feature columns: " + ", ".join(missing))
    if not feature_columns:
        raise ValueError("No model feature columns were selected.")
    features = frame.loc[:, feature_columns]
    labels = frame[contract.label_column].to_numpy()
    predictions: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for index, (filename, payload) in enumerate(models):
        model = load_uploaded_model(payload, filename)
        try:
            prediction = np.asarray(model.predict(features))
        except Exception as error:
            raise ValueError(
                f"Model {filename} could not predict this dataset. "
                "Check feature order and preprocessing."
            ) from error
        if prediction.shape != labels.shape:
            raise ValueError(
                f"Model {filename} must return one prediction per dataset row."
            )
        predictions.append(prediction)
        calibration = _calibration_summary(
            model, features, prediction, labels
        )
        summaries.append(
            {
                "index": index,
                **artifact_fingerprint(filename, payload),
                "accuracy": float((prediction == labels).mean()),
                "macro_f1": float(
                    f1_score(labels, prediction, average="macro", zero_division=0)
                ),
                "correct_count": int(np.count_nonzero(prediction == labels)),
                **calibration,
            }
        )
    ranked = sorted(
        summaries,
        key=lambda item: (-item["accuracy"], -item["macro_f1"], item["index"]),
    )
    for rank, item in enumerate(ranked, start=1):
        summaries[item["index"]]["rank"] = rank
    agreements = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            agreements.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "agreement": float((predictions[left] == predictions[right]).mean()),
                }
            )
    return {
        "schema_version": 1,
        "task": "classification",
        "dataset": {
            "rows": len(frame),
            "feature_count": len(feature_columns),
            "label_column": contract.label_column,
            **artifact_fingerprint(*dataset),
        },
        "manifest": artifact_fingerprint("manifest.json", manifest),
        "models": summaries,
        "pairwise_agreements": agreements,
        "best_model_index": ranked[0]["index"],
        "best_macro_f1_index": max(
            summaries, key=lambda item: (item["macro_f1"], item["accuracy"])
        )["index"],
    }


def validate_upload_models(
    *,
    models: list[tuple[str, bytes]],
    dataset: tuple[str, bytes],
    manifest: bytes,
) -> dict[str, Any]:
    """Return a per-model compatibility check without persisting uploads."""
    if not 2 <= len(models) <= 5:
        raise ValueError("Upload between two and five models for validation.")
    contract = parse_manifest(manifest)
    frame = load_uploaded_dataset(dataset[1], dataset[0])
    if contract.label_column not in frame:
        raise ValueError(f"Dataset is missing label column: {contract.label_column}.")
    excluded = {contract.label_column}
    if contract.id_column is not None:
        excluded.add(contract.id_column)
    if contract.image_column is not None:
        excluded.add(contract.image_column)
    feature_columns = (
        list(contract.feature_columns)
        if contract.feature_columns is not None
        else [column for column in frame.columns if column not in excluded]
    )
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise ValueError("Dataset is missing feature columns: " + ", ".join(missing))
    if not feature_columns:
        raise ValueError("No model feature columns were selected.")
    features = frame.loc[:, feature_columns]
    checks = []
    for index, (filename, payload) in enumerate(models):
        check: dict[str, Any] = {
            "index": index,
            "filename": filename,
            "compatible": False,
            "supports_confidence": False,
        }
        try:
            model = load_uploaded_model(payload, filename)
            prediction = np.asarray(model.predict(features))
            if prediction.shape != (len(frame),):
                raise ValueError("predict() did not return one value per dataset row.")
            check.update(
                {
                    "compatible": True,
                    "model_type": type(model).__name__,
                    "supports_confidence": callable(getattr(model, "predict_proba", None)),
                    "message": "Ready for comparison.",
                }
            )
        except ValueError as error:
            check["message"] = str(error)
        except Exception:
            check["message"] = (
                "Prediction failed. Check feature order, preprocessing, and installed dependencies."
            )
        checks.append(check)
    return {
        "schema_version": 1,
        "ready": all(check["compatible"] for check in checks),
        "dataset": {
            "filename": dataset[0],
            "rows": len(frame),
            "feature_count": len(feature_columns),
            "label_column": contract.label_column,
        },
        "models": checks,
    }
