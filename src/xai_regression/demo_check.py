"""Preflight a local dashboard demonstration bundle before a live upload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xai_regression.regression_comparison import (
    compare_regression_upload_payloads,
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
    model_sizes = {name: len(payload) for name, payload in models}

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
        for model in models:
            comparison = compare_upload_payloads(
                model_v1=model,
                model_v2=model,
                dataset=dataset,
                manifest=manifest,
                images=images_payload,
            )
            feature_explanations = comparison["feature_explanations"]
            if feature_explanations is not None:
                methods = feature_explanations["methods"]
                method_names = list(methods)
                labels = [methods[name]["label"] for name in method_names]
                available = bool(method_names)
                method = "+".join(method_names) if method_names else "unavailable"
                label = " + ".join(labels) if labels else "Unavailable"
                note = (
                    "Tabular feature explanation methods available for this model."
                    if available
                    else feature_explanations["v1"]["note"]
                )
            else:
                explanation = comparison["explanations"]["v1"]
                available = explanation["method"] != "unavailable" and any(
                    row["v1_explanation"] is not None
                    for row in comparison["samples"]
                )
                method = explanation["method"]
                label = explanation["label"]
                note = explanation["note"]
            explanation_checks.append(
                {
                    "filename": model[0],
                    "available": available,
                    "method": method,
                    "label": label,
                    "note": note,
                }
            )
    elif check_explanations and task == "regression":
        for model in models:
            comparison = compare_regression_upload_payloads(
                model_v1=model,
                model_v2=model,
                dataset=dataset,
                manifest=manifest,
            )
            features = comparison["feature_importance"]["features"]
            available = bool(features)
            explanation_checks.append(
                {
                    "filename": model[0],
                    "available": available,
                    "method": "permutation_importance" if available else "unavailable",
                    "label": "Permutation importance" if available else "Unavailable",
                    "note": (
                        "Model-agnostic MAE permutation importance was generated "
                        f"for {len(features)} feature(s)."
                        if available
                        else "No numeric feature importance could be generated."
                    ),
                }
            )

    explanations_ready = not check_explanations or all(
        row["available"] for row in explanation_checks
    )
    explanation_by_filename = {
        row["filename"]: row for row in explanation_checks
    }
    model_details = []
    for model in validation["models"]:
        explanation = explanation_by_filename.get(model["filename"])
        model_details.append(
            {
                **model,
                "file_size_bytes": model_sizes[model["filename"]],
                "explanation_method": (
                    explanation["label"] if explanation is not None else "Not checked"
                ),
                "explanation_ready": (
                    explanation["available"] if explanation is not None else None
                ),
            }
        )
    return {
        "ready": bool(validation["ready"]) and explanations_ready,
        "task": task,
        "bundle_dir": str(root.resolve()),
        "dataset": validation["dataset"],
        "models": model_details,
        "model_count": len(models),
        "image_count": image_count,
        "best_model_index": overview["best_model_index"],
        "explanations": explanation_checks,
    }


def check_demo_matrix(
    bundle_dirs: list[Path], *, check_explanations: bool = False
) -> dict[str, Any]:
    """Validate several upload bundles without stopping after one failure."""
    if not bundle_dirs:
        raise ValueError("Provide at least one demo bundle directory.")
    bundles = []
    for bundle_dir in bundle_dirs:
        try:
            result = check_demo_bundle(
                bundle_dir, check_explanations=check_explanations
            )
        except Exception as error:
            bundles.append(
                {
                    "bundle_dir": str(Path(bundle_dir).resolve()),
                    "ready": False,
                    "error": str(error),
                }
            )
        else:
            bundles.append({**result, "error": None})
    ready_count = sum(bool(bundle["ready"]) for bundle in bundles)
    return {
        "ready": ready_count == len(bundles),
        "bundle_count": len(bundles),
        "ready_count": ready_count,
        "failed_count": len(bundles) - ready_count,
        "bundles": bundles,
    }


def write_demo_matrix_report(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write machine-readable and presentation-ready matrix reports."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "demo_validation_matrix.json"
    markdown_path = destination / "demo_validation_matrix.md"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Dashboard Demo Validation Matrix",
        "",
        f"**Overall status:** {'READY' if result['ready'] else 'FAILED'}  ",
        f"**Bundles ready:** {result['ready_count']}/{result['bundle_count']}",
        "",
        "| Bundle | Task | Models | Samples | Features | Explanations | Status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for bundle in result["bundles"]:
        name = Path(bundle["bundle_dir"]).name
        if bundle["error"] is not None:
            lines.append(
                f"| {name} | — | — | — | — | {bundle['error']} | FAILED |"
            )
            continue
        explanation_labels = sorted(
            {row["label"] for row in bundle["explanations"]}
        )
        explanation_text = ", ".join(explanation_labels) or "Not checked"
        status = "READY" if bundle["ready"] else "FAILED"
        lines.append(
            f"| {name} | {bundle['task']} | {bundle['model_count']} | "
            f"{bundle['dataset']['rows']} | {bundle['dataset']['feature_count']} | "
            f"{explanation_text} | {status} |"
        )
    lines.extend(["", "## Model Details", ""])
    for bundle in result["bundles"]:
        if bundle["error"] is not None:
            continue
        name = Path(bundle["bundle_dir"]).name
        lines.extend(
            [
                f"### {name}",
                "",
                "| Model file | Model type | Size | Compatible | Explanation |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for model in bundle["models"]:
            size_kib = model["file_size_bytes"] / 1024
            compatible = "Yes" if model["compatible"] else "No"
            lines.append(
                f"| {model['filename']} | {model['model_type'] or 'Unknown'} | "
                f"{size_kib:.1f} KiB | {compatible} | "
                f"{model['explanation_method']} |"
            )
        lines.append("")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
