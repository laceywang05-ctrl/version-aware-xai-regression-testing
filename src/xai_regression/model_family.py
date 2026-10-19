"""Summarise held-out SHAP evidence across datasets and model families."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DATASET_LABELS = {
    "breast_cancer": "Breast Cancer Wisconsin",
    "wine_binary": "Wine (binary)",
}
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Required model-family report not found: {report_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Model-family report is not valid JSON: {report_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Model-family report must be a JSON object: {report_path}")
    return payload


def normalize_benchmark(
    summary: dict[str, Any],
    *,
    expected_dataset: str,
    model_family: str,
) -> list[dict[str, Any]]:
    """Return one comparable row per changed scenario."""
    if summary.get("benchmark") != "controlled-model-version-scenarios":
        raise ValueError("Expected a controlled model-version benchmark summary.")
    reported_dataset = summary.get("dataset")
    if reported_dataset is not None and reported_dataset != expected_dataset:
        raise ValueError(
            f"Expected dataset {expected_dataset}, got {reported_dataset}."
        )
    reported_family = summary.get("model_family")
    if reported_family is not None and reported_family != model_family:
        raise ValueError(
            f"Expected model family {model_family}, got {reported_family}."
        )
    scenarios = summary.get("scenario_results")
    comparisons = summary.get("paired_control_comparisons")
    if not isinstance(scenarios, list) or not isinstance(comparisons, list):
        raise ValueError("Benchmark summary is missing scenario comparison lists.")
    control = next(
        (row for row in scenarios if row.get("change_type") == "control"), None
    )
    if control is None:
        raise ValueError("Benchmark summary is missing a control scenario.")
    scenarios_by_id = {row["scenario_id"]: row for row in scenarios}
    rows = []
    for comparison in comparisons:
        scenario_id = comparison["candidate_scenario_id"]
        candidate = scenarios_by_id.get(scenario_id)
        if candidate is None:
            raise ValueError(f"Missing scenario result for {scenario_id}.")
        adjusted_p = float(comparison["bonferroni_adjusted_p"])
        rows.append(
            {
                "dataset_id": expected_dataset,
                "dataset": DATASET_LABELS.get(expected_dataset, expected_dataset),
                "model_family": model_family,
                "model": MODEL_LABELS.get(model_family, model_family),
                "scenario_id": scenario_id,
                "paired_runs": int(comparison["paired_runs"]),
                "control_review_rate": float(control["review_rate"]),
                "candidate_review_rate": float(candidate["review_rate"]),
                "review_rate_difference": float(
                    comparison["review_rate_difference"]
                ),
                "prediction_agreement": float(
                    candidate["mean_prediction_agreement"]
                ),
                "cosine_similarity": float(candidate["mean_cosine_similarity"]),
                "bonferroni_adjusted_p": adjusted_p,
                "separated": adjusted_p < 0.05,
            }
        )
    return rows


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-Model-Family SHAP Evidence",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "| Dataset | Model | Scenario | Control review | Candidate review | Agreement | Adjusted p | Separated |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["entries"]:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['scenario_id']} | "
            f"{row['control_review_rate']:.1%} | "
            f"{row['candidate_review_rate']:.1%} | "
            f"{row['prediction_agreement']:.1%} | "
            f"{row['bonferroni_adjusted_p']:.3g} | "
            f"{'yes' if row['separated'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Scenario consistency",
            "",
            "| Scenario | Separated cells | Total cells |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["scenario_consistency"]:
        lines.append(
            f"| {row['scenario_id']} | {row['separated_cells']} | "
            f"{row['total_cells']} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    return "\n".join(lines)


def run_model_family_summary(
    *,
    random_forest_breast_report: str | Path,
    random_forest_wine_report: str | Path,
    gradient_boosting_breast_report: str | Path,
    gradient_boosting_wine_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a 2-dataset by 2-model-family scenario evidence matrix."""
    inputs = (
        (random_forest_breast_report, "breast_cancer", "random_forest"),
        (random_forest_wine_report, "wine_binary", "random_forest"),
        (gradient_boosting_breast_report, "breast_cancer", "gradient_boosting"),
        (gradient_boosting_wine_report, "wine_binary", "gradient_boosting"),
    )
    entries = [
        row
        for path, dataset, family in inputs
        for row in normalize_benchmark(
            _read_json(path),
            expected_dataset=dataset,
            model_family=family,
        )
    ]
    scenario_ids = list(dict.fromkeys(row["scenario_id"] for row in entries))
    consistency = []
    for scenario_id in scenario_ids:
        matching = [row for row in entries if row["scenario_id"] == scenario_id]
        consistency.append(
            {
                "scenario_id": scenario_id,
                "separated_cells": sum(row["separated"] for row in matching),
                "total_cells": len(matching),
            }
        )
    universal = [
        row["scenario_id"]
        for row in consistency
        if row["separated_cells"] == row["total_cells"]
    ]
    summary = {
        "schema_version": 1,
        "title": "Cross-model-family Tree SHAP evidence summary",
        "status": "CROSS_FAMILY_SEPARATION" if universal else "MIXED_EVIDENCE",
        "coverage": {
            "datasets": 2,
            "model_families": 2,
            "dataset_model_cells": 4,
            "scenario_comparisons": len(entries),
        },
        "universally_separated_scenarios": universal,
        "scenario_consistency": consistency,
        "entries": entries,
        "interpretation": (
            "At least one controlled structural change separates from benign "
            "retraining in both datasets and both model families. This supports "
            "cross-family sensitivity, but review flags remain diagnostic signals "
            "rather than proof that a candidate model is harmful."
            if universal
            else
            "Separation depends on the dataset, model family or structural change; "
            "the current evidence does not support a universal sensitivity claim."
        ),
        "sources": [str(Path(path)) for path, _, _ in inputs],
    }

    reports = Path(output_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports / "report.md").write_text(_markdown(summary), encoding="utf-8")
    with (reports / "model_family_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)

    print("Cross-model-family SHAP summary")
    print("-------------------------------")
    print(f"Status: {summary['status']}")
    print(f"Universal scenarios: {', '.join(universal) or 'none'}")
    print(f"Reports saved to: {reports}")
    return summary
