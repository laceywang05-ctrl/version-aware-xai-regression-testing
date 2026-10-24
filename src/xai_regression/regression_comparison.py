"""Compare two trusted user-supplied regression model versions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd

from xai_regression.upload_comparison import (
    _exact_mcnemar_p,
    _json_value,
    load_uploaded_dataset,
    load_uploaded_model,
)


@dataclass(frozen=True)
class RegressionManifest:
    label_column: str
    id_column: str | None
    dataset_note: str | None
    feature_columns: tuple[str, ...] | None
    trusted_model_files: bool


def parse_regression_manifest(payload: bytes | str | dict[str, Any]) -> RegressionManifest:
    """Parse the minimal manifest used by the regression upload API."""
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
    if str(raw.get("task", "")) != "regression":
        raise ValueError("Regression comparison requires task set to regression.")
    label_column = str(raw.get("label_column", "")).strip()
    if not label_column:
        raise ValueError("Manifest requires a non-empty label_column.")
    id_column = raw.get("id_column")
    id_column = str(id_column).strip() or None if id_column is not None else None
    features = raw.get("feature_columns")
    if features is not None:
        if not isinstance(features, list) or not features:
            raise ValueError("feature_columns must be a non-empty JSON list.")
        feature_columns = tuple(str(value) for value in features)
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature_columns cannot contain duplicates.")
    else:
        feature_columns = None
    if raw.get("trusted_model_files") is not True:
        raise ValueError(
            "Set trusted_model_files to true only for model files you trust."
        )
    return RegressionManifest(
        label_column=label_column,
        id_column=id_column,
        dataset_note=(str(raw["dataset_note"]).strip() or None)
        if raw.get("dataset_note")
        else None,
        feature_columns=feature_columns,
        trusted_model_files=True,
    )


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    centered = actual - actual.mean()
    denominator = float(np.sum(np.square(centered)))
    r2 = None if denominator == 0 else float(1 - np.sum(np.square(residual)) / denominator)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _paired_mean_bootstrap_interval(
    differences: np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 311,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for the paired mean change."""
    if differences.size == 0:
        raise ValueError("Bootstrap differences cannot be empty.")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    batch_size = 250
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = generator.integers(0, differences.size, size=(stop - start, differences.size))
        means[start:stop] = differences[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return float(lower), float(upper)


def _permutation_feature_importance(
    model_v1: Any,
    model_v2: Any,
    features: pd.DataFrame,
    actual: np.ndarray,
    baseline_v1: float,
    baseline_v2: float,
    *,
    repeats: int = 3,
    feature_limit: int = 50,
    seed: int = 311,
) -> tuple[list[dict[str, Any]], bool]:
    """Compare deterministic MAE permutation importance for two regressors."""
    selected = list(features.columns[:feature_limit])
    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for column in selected:
        v1_increases = []
        v2_increases = []
        for _ in range(repeats):
            permuted = features.copy()
            permuted[column] = generator.permutation(permuted[column].to_numpy())
            v1_prediction = np.asarray(model_v1.predict(permuted), dtype=float)
            v2_prediction = np.asarray(model_v2.predict(permuted), dtype=float)
            v1_increases.append(float(np.mean(np.abs(v1_prediction - actual)) - baseline_v1))
            v2_increases.append(float(np.mean(np.abs(v2_prediction - actual)) - baseline_v2))
        v1_importance = float(np.mean(v1_increases))
        v2_importance = float(np.mean(v2_increases))
        rows.append(
            {
                "feature": str(column),
                "v1_importance": v1_importance,
                "v2_importance": v2_importance,
                "importance_change": v2_importance - v1_importance,
            }
        )
    rows.sort(key=lambda row: max(abs(row["v1_importance"]), abs(row["v2_importance"])), reverse=True)
    return rows, len(features.columns) > feature_limit


def compare_regressors(
    model_v1: Any,
    model_v2: Any,
    dataset: pd.DataFrame,
    manifest: RegressionManifest,
) -> dict[str, Any]:
    """Return matched aggregate and sample-level regression errors."""
    if manifest.label_column not in dataset:
        raise ValueError(f"Dataset is missing label column: {manifest.label_column}.")
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
    try:
        actual = dataset[manifest.label_column].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Regression labels must be numeric.") from error
    if not np.isfinite(actual).all():
        raise ValueError("Regression labels must contain only finite values.")
    features = dataset.loc[:, feature_columns]
    try:
        prediction_v1 = np.asarray(model_v1.predict(features), dtype=float)
        prediction_v2 = np.asarray(model_v2.predict(features), dtype=float)
    except Exception as error:
        raise ValueError(
            "A model could not predict this dataset. Check feature order and preprocessing."
        ) from error
    if prediction_v1.shape != actual.shape or prediction_v2.shape != actual.shape:
        raise ValueError("Model predictions must contain one value per dataset row.")
    if not np.isfinite(prediction_v1).all() or not np.isfinite(prediction_v2).all():
        raise ValueError("Regression predictions must contain only finite values.")
    error_v1 = np.abs(prediction_v1 - actual)
    error_v2 = np.abs(prediction_v2 - actual)
    feature_importance, importance_truncated = _permutation_feature_importance(
        model_v1,
        model_v2,
        features,
        actual,
        float(error_v1.mean()),
        float(error_v2.mean()),
    )
    tolerance = 1e-12
    improved = error_v2 < error_v1 - tolerance
    worsened = error_v2 > error_v1 + tolerance
    unchanged = ~(improved | worsened)
    error_change = error_v2 - error_v1
    mae_change_ci_lower, mae_change_ci_upper = _paired_mean_bootstrap_interval(error_change)
    sign_test_p = _exact_mcnemar_p(
        int(np.count_nonzero(worsened)), int(np.count_nonzero(improved))
    )
    identifiers = (
        dataset[manifest.id_column].tolist()
        if manifest.id_column is not None
        else list(range(len(dataset)))
    )
    samples = []
    for index in range(len(dataset)):
        outcome = "improved" if improved[index] else "worsened" if worsened[index] else "unchanged"
        samples.append(
            {
                "sample_id": _json_value(identifiers[index]),
                "actual": float(actual[index]),
                "v1_prediction": float(prediction_v1[index]),
                "v2_prediction": float(prediction_v2[index]),
                "v1_absolute_error": float(error_v1[index]),
                "v2_absolute_error": float(error_v2[index]),
                "absolute_error_change": float(error_change[index]),
                "outcome": outcome,
            }
        )
    return {
        "schema_version": 1,
        "task": "regression",
        "dataset": {
            "rows": len(dataset),
            "feature_count": len(feature_columns),
            "label_column": manifest.label_column,
            "id_column": manifest.id_column,
            "note": manifest.dataset_note,
        },
        "summary": {
            "total_samples": len(dataset),
            "v1": _regression_metrics(actual, prediction_v1),
            "v2": _regression_metrics(actual, prediction_v2),
            "mae_change": float(error_v2.mean() - error_v1.mean()),
            "mae_change_ci_95": {
                "lower": mae_change_ci_lower,
                "upper": mae_change_ci_upper,
                "method": "paired_percentile_bootstrap",
                "resamples": 5000,
            },
            "mae_change_significant": bool(
                mae_change_ci_upper < 0 or mae_change_ci_lower > 0
            ),
            "improved_count": int(np.count_nonzero(improved)),
            "worsened_count": int(np.count_nonzero(worsened)),
            "unchanged_count": int(np.count_nonzero(unchanged)),
            "paired_sign_test_p": sign_test_p,
            "paired_significant": bool(sign_test_p < 0.05),
        },
        "feature_importance": {
            "metric": "mae_increase_after_permutation",
            "repeats": 3,
            "evaluated_feature_count": len(feature_importance),
            "truncated": importance_truncated,
            "features": feature_importance,
        },
        "samples": samples,
        "limitations": [
            "The detailed regression view compares one selected pair from two to five trusted local .joblib models.",
            "Uploaded files are processed in memory and are not persisted.",
            "The paired sign test uses error direction only, not error magnitude.",
            "The MAE-change interval is a paired percentile bootstrap with 5,000 resamples.",
            "Feature importance is model-agnostic permutation importance averaged over three deterministic shuffles.",
        ],
    }


def compare_regression_upload_payloads(
    *,
    model_v1: tuple[str, bytes],
    model_v2: tuple[str, bytes],
    dataset: tuple[str, bytes],
    manifest: bytes,
) -> dict[str, Any]:
    """Load trusted regression upload fields and return their comparison."""
    contract = parse_regression_manifest(manifest)
    first = load_uploaded_model(model_v1[1], model_v1[0])
    second = load_uploaded_model(model_v2[1], model_v2[0])
    frame = load_uploaded_dataset(dataset[1], dataset[0])
    result = compare_regressors(first, second, frame, contract)
    result["models"] = {
        "v1": {"filename": model_v1[0]},
        "v2": {"filename": model_v2[0]},
    }
    return result


def summarize_regression_upload_models(
    *, models: list[tuple[str, bytes]], dataset: tuple[str, bytes], manifest: bytes
) -> dict[str, Any]:
    """Rank two to five uploaded regressors on one labelled dataset."""
    if not 2 <= len(models) <= 5:
        raise ValueError("Upload between two and five regression models for the overview.")
    contract = parse_regression_manifest(manifest)
    frame = load_uploaded_dataset(dataset[1], dataset[0])
    if contract.label_column not in frame:
        raise ValueError(f"Dataset is missing label column: {contract.label_column}.")
    excluded = {contract.label_column}
    if contract.id_column is not None:
        excluded.add(contract.id_column)
    feature_columns = list(contract.feature_columns) if contract.feature_columns is not None else [
        column for column in frame.columns if column not in excluded
    ]
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise ValueError("Dataset is missing feature columns: " + ", ".join(missing))
    try:
        actual = frame[contract.label_column].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Regression labels must be numeric.") from error
    features = frame.loc[:, feature_columns]
    summaries = []
    for index, (filename, payload) in enumerate(models):
        model = load_uploaded_model(payload, filename)
        try:
            predicted = np.asarray(model.predict(features), dtype=float)
        except Exception as error:
            raise ValueError(
                f"Model {filename} could not predict this dataset. Check feature order and preprocessing."
            ) from error
        if predicted.shape != actual.shape:
            raise ValueError(f"Model {filename} must return one prediction per dataset row.")
        summaries.append({"index": index, "filename": filename, **_regression_metrics(actual, predicted)})
    ranked = sorted(summaries, key=lambda item: (item["mae"], item["index"]))
    for rank, item in enumerate(ranked, start=1):
        summaries[item["index"]]["rank"] = rank
    return {
        "task": "regression",
        "dataset": {"rows": len(frame), "feature_count": len(feature_columns)},
        "models": summaries,
        "best_model_index": ranked[0]["index"],
    }


def validate_regression_upload_models(
    *, models: list[tuple[str, bytes]], dataset: tuple[str, bytes], manifest: bytes
) -> dict[str, Any]:
    """Check each uploaded regressor against the shared numeric dataset."""
    if not 2 <= len(models) <= 5:
        raise ValueError("Upload between two and five regression models for validation.")
    contract = parse_regression_manifest(manifest)
    frame = load_uploaded_dataset(dataset[1], dataset[0])
    if contract.label_column not in frame:
        raise ValueError(f"Dataset is missing label column: {contract.label_column}.")
    if contract.id_column is not None and contract.id_column not in frame:
        raise ValueError(f"Dataset is missing id column: {contract.id_column}.")
    excluded = {contract.label_column}
    if contract.id_column is not None:
        excluded.add(contract.id_column)
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
    try:
        actual = frame[contract.label_column].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Regression labels must be numeric.") from error
    if not np.isfinite(actual).all():
        raise ValueError("Regression labels must contain only finite values.")
    features = frame.loc[:, feature_columns]
    reports = []
    for index, (filename, payload) in enumerate(models):
        model_type = None
        try:
            model = load_uploaded_model(payload, filename)
            model_type = type(model).__name__
            try:
                predicted = np.asarray(model.predict(features), dtype=float)
            except Exception as error:
                raise ValueError(
                    "Could not produce numeric predictions. Check feature order and preprocessing."
                ) from error
            if predicted.shape != actual.shape:
                raise ValueError("Must return one numeric prediction per dataset row.")
            if not np.isfinite(predicted).all():
                raise ValueError("Predictions must contain only finite numeric values.")
            reports.append(
                {
                    "index": index,
                    "filename": filename,
                    "compatible": True,
                    "model_type": model_type,
                    "message": "Ready for numeric prediction.",
                }
            )
        except ValueError as error:
            reports.append(
                {
                    "index": index,
                    "filename": filename,
                    "compatible": False,
                    "model_type": model_type,
                    "message": str(error),
                }
            )
    return {
        "task": "regression",
        "ready": all(report["compatible"] for report in reports),
        "dataset": {"rows": len(frame), "feature_count": len(feature_columns)},
        "models": reports,
    }
