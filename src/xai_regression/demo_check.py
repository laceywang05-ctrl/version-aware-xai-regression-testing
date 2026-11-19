"""Preflight a local dashboard demonstration bundle before a live upload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xai_regression.regression_comparison import (
    summarize_regression_upload_models,
    validate_regression_upload_models,
)
from xai_regression.upload_comparison import (
    compare_upload_payloads,
    load_uploaded_images,
    summarize_upload_models,
    validate_upload_models,
)


def check_demo_bundle(
    bundle_dir: Path, *, check_explanations: bool = False
) -> dict[str, Any]:
    """Validate the files and model/data compatibility in one upload bundle."""
    root = Path(bundle_dir)
    if not root.is_dir():
        raise ValueError(f"Demo bundle directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    dataset_path = root / "samples.csv"
    for path in (manifest_path, dataset_path):
        if not path.is_file():
            raise ValueError(f"Demo bundle is missing required file: {path.name}")
    model_paths = sorted(root.glob("*.joblib"))
    if not 2 <= len(model_paths) <= 5:
        raise ValueError("Demo bundle must contain between two and five .joblib models.")

    manifest = manifest_path.read_bytes()
    dataset = (dataset_path.name, dataset_path.read_bytes())
    try:
        manifest_data = json.loads(manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest.json must contain valid UTF-8 JSON.") from error
    task = str(manifest_data.get("task", ""))
    models = [(path.name, path.read_bytes()) for path in model_paths]

    images_path = root / "images.zip"
    image_count = None
    images_payload = None
    if task == "classification":
        if manifest_data.get("image_column") and not images_path.is_file():
            raise ValueError("Manifest uses image_column but images.zip is missing.")
        if images_path.is_file():
            images_payload = (images_path.name, images_path.read_bytes())
            image_count = len(
                load_uploaded_images(images_payload[1], images_payload[0])
            )
        validation = validate_upload_models(models=models, dataset=dataset, manifest=manifest)
        overview = summarize_upload_models(models=models, dataset=dataset, manifest=manifest)
    elif task == "regression":
        validation = validate_regression_upload_models(
            models=models, dataset=dataset, manifest=manifest
        )
        overview = summarize_regression_upload_models(
            models=models, dataset=dataset, manifest=manifest
        )
    else:
        raise ValueError("Manifest task must be classification or regression.")

    explanation_checks = []
    if check_explanations and task == "classification":
        if manifest_data.get("image_shape") is None:
            raise ValueError(
                "Explanation preflight requires image_shape in manifest.json."
            )
        for model in models:
            comparison = compare_upload_payloads(
                model_v1=model,
                model_v2=model,
                dataset=dataset,
                manifest=manifest,
                images=images_payload,
            )
            explanation = comparison["explanations"]["v1"]
            available = explanation["method"] != "unavailable" and any(
                row["v1_explanation"] is not None for row in comparison["samples"]
            )
            explanation_checks.append(
                {
                    "filename": model[0],
                    "available": available,
                    "method": explanation["method"],
                    "label": explanation["label"],
                    "note": explanation["note"],
                }
            )

    explanations_ready = not check_explanations or all(
        row["available"] for row in explanation_checks
    )
    return {
        "ready": bool(validation["ready"]) and explanations_ready,
        "task": task,
        "bundle_dir": str(root.resolve()),
        "dataset": validation["dataset"],
        "models": validation["models"],
        "model_count": len(models),
        "image_count": image_count,
        "best_model_index": overview["best_model_index"],
        "explanations": explanation_checks,
    }
