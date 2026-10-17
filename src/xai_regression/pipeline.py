"""End-to-end orchestration for SHAP, LIME and unified reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xai_regression.config import load_config
from xai_regression.lime_workflow import run_lime_cross_version
from xai_regression.reporting import run_unified_report
from xai_regression.runner import run_validation


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def run_pipeline(
    *,
    shap_config_path: str | Path,
    lime_config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Run both explainers and build the report consumed by the dashboard."""
    root = Path(project_root)
    shap_path = _resolve(root, shap_config_path)
    lime_path = _resolve(root, lime_config_path)
    output = _resolve(root, output_dir)
    shap_config = load_config(shap_path)
    lime_config = load_config(lime_path)

    if shap_config.explainer.method != "tree_shap":
        raise ValueError("The SHAP workflow requires explainer.method=tree_shap.")
    if lime_config.explainer.method != "lime" or lime_config.lime is None:
        raise ValueError("The LIME workflow requires a complete lime configuration.")

    print("End-to-end XAI regression workflow", flush=True)
    print("==================================", flush=True)
    print("[1/3] Running SHAP validation", flush=True)
    shap_summary = run_validation(
        shap_config,
        config_path=shap_path,
        project_root=root,
    )

    print("[2/3] Running LIME cross-version comparison", flush=True)
    lime_summary = run_lime_cross_version(
        lime_config,
        config_path=lime_path,
        project_root=root,
    )

    shap_output = _resolve(root, shap_config.output_dir)
    lime_output = _resolve(root, lime_config.lime.output_dir)
    print("[3/3] Building unified report", flush=True)
    unified_summary = run_unified_report(
        shap_report_dir=shap_output / "reports",
        lime_report_dir=lime_output / "reports",
        output_dir=output,
    )

    report_path = output / "reports" / "summary.json"
    manifest = {
        "workflow": "shap-lime-unified-report",
        "shap_experiment_id": shap_config.experiment_id,
        "lime_experiment_id": lime_config.experiment_id,
        "shap_status": shap_summary["status"],
        "lime_status": lime_summary["status"],
        "overall_status": unified_summary["overall_status"],
        "dashboard_report": str(report_path),
    }
    manifest_path = output / "reports" / "workflow_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Dashboard report: {report_path}", flush=True)
    print(f"Workflow manifest: {manifest_path}", flush=True)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "report_path": report_path,
    }
