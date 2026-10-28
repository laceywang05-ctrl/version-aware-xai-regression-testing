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
    load_uploaded_images,
    summarize_upload_models,
    validate_upload_models,
)


def check_demo_bundle(bundle_dir: Path) -> dict[str, Any]:
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
    if task == "classification":
        if manifest_data.get("image_column") and not images_path.is_file():
            raise ValueError("Manifest uses image_column but images.zip is missing.")
        if images_path.is_file():
            image_count = len(load_uploaded_images(images_path.read_bytes(), images_path.name))
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

    return {
        "ready": bool(validation["ready"]),
        "task": task,
        "bundle_dir": str(root.resolve()),
        "dataset": validation["dataset"],
        "models": validation["models"],
        "model_count": len(models),
        "image_count": image_count,
        "best_model_index": overview["best_model_index"],
    }
