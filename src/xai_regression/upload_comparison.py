"""Validate and compare two user-supplied classification model versions."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd


MAX_DATASET_ROWS = 5_000


@dataclass(frozen=True)
class UploadManifest:
    """User-provided metadata required to run a classification comparison."""

    task: str
    label_column: str
    id_column: str | None
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


def compare_classifiers(
    model_v1: Any,
    model_v2: Any,
    dataset: pd.DataFrame,
    manifest: UploadManifest,
) -> dict[str, Any]:
    """Run a matched, sample-level classification comparison."""
    if manifest.label_column not in dataset:
        raise ValueError(
            f"Dataset is missing label column: {manifest.label_column}."
        )
    if manifest.id_column is not None and manifest.id_column not in dataset:
        raise ValueError(f"Dataset is missing id column: {manifest.id_column}.")
    excluded = {manifest.label_column}
    if manifest.id_column is not None:
        excluded.add(manifest.id_column)
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
            "Explanation generation will be added through model-compatible adapters.",
        ],
    }


def compare_upload_payloads(
    *,
    model_v1: tuple[str, bytes],
    model_v2: tuple[str, bytes],
    dataset: tuple[str, bytes],
    manifest: bytes,
) -> dict[str, Any]:
    """Load the four uploaded files and return their comparison report."""
    contract = parse_manifest(manifest)
    first = load_uploaded_model(model_v1[1], model_v1[0])
    second = load_uploaded_model(model_v2[1], model_v2[0])
    frame = load_uploaded_dataset(dataset[1], dataset[0])
    return compare_classifiers(first, second, frame, contract)
