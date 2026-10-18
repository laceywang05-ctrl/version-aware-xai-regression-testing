"""Cross-dataset summary for the completed SHAP and LIME experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DATASET_LABELS = {
    "breast_cancer": "Breast Cancer Wisconsin",
    "wine_binary": "Wine (binary)",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Required cross-dataset input not found: {report_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Cross-dataset input is not valid JSON: {report_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Cross-dataset input must be a JSON object: {report_path}")
    return payload


def normalize_shap_benchmark(
    summary: dict[str, Any],
    *,
    dataset_id: str,
) -> dict[str, Any]:
    """Normalize one held-out SHAP benchmark without inventing a pass/fail status."""
    scenarios = summary.get("scenario_results")
    comparisons = summary.get("paired_control_comparisons")
    if not isinstance(scenarios, list) or not isinstance(comparisons, list):
        raise ValueError(
            "SHAP benchmark summary must contain scenario_results and "
            "paired_control_comparisons lists."
        )
    control = next(
        (row for row in scenarios if row.get("change_type") == "control"),
        None,
    )
    candidates = [row for row in scenarios if row.get("change_type") != "control"]
    if control is None or not candidates:
        raise ValueError("SHAP benchmark requires a control and candidate scenarios.")
    strongest = max(candidates, key=lambda row: float(row["review_rate"]))
    separated = sum(
        float(row["bonferroni_adjusted_p"]) < 0.05 for row in comparisons
    )
    return {
        "dataset_id": dataset_id,
        "dataset": DATASET_LABELS.get(dataset_id, dataset_id),
        "method": "SHAP",
        "evidence_type": "held-out repeated benchmark",
        "status": "SEPARATED" if separated else "NO_SEPARATION",
        "observations": int(summary["candidate_run_count"]),
        "prediction_agreement": float(strongest["mean_prediction_agreement"]),
        "cosine_similarity": float(strongest["mean_cosine_similarity"]),
        "headline_metric": "strongest_candidate_review_rate",
        "headline_value": float(strongest["review_rate"]),
        "control_review_rate": float(control["review_rate"]),
        "strongest_scenario": strongest["scenario_id"],
        "separated_scenarios": separated,
        "scenario_comparisons": len(comparisons),
        "flagged_metrics": [],
        "interpretation": (
            "Repeated held-out sensitivity evidence. Separation from the "
            "seed-only control does not show that the candidate is harmful."
        ),
    }


def normalize_lime_comparison(
    summary: dict[str, Any],
    *,
    dataset_id: str,
) -> dict[str, Any]:
    """Normalize one LIME cross-version comparison."""
    if "heldout_control" in summary and "candidate" in summary:
        required = (
            "status",
            "matched_candidate_comparisons",
            "probe_prediction_agreement",
            "heldout_candidate_means",
            "paired_seed_cross_version_means",
        )
        missing = [key for key in required if key not in summary]
        if missing:
            raise ValueError(
                "Held-out LIME summary is missing: " + ", ".join(missing)
            )
        return {
            "dataset_id": dataset_id,
            "dataset": DATASET_LABELS.get(dataset_id, dataset_id),
            "method": "LIME",
            "evidence_type": "held-out paired-seed validation",
            "status": str(summary["status"]),
            "observations": int(summary["matched_candidate_comparisons"]),
            "prediction_agreement": float(summary["probe_prediction_agreement"]),
            "cosine_similarity": float(
                summary["heldout_candidate_means"]["cosine_similarity"]
            ),
            "paired_seed_cosine_similarity": float(
                summary["paired_seed_cross_version_means"]["cosine_similarity"]
            ),
            "headline_metric": "candidate_review_rate",
            "headline_value": float(summary["candidate"]["review_rate"]),
            "control_review_rate": float(
                summary["heldout_control"]["review_rate"]
            ),
            "strongest_scenario": None,
            "separated_scenarios": None,
            "scenario_comparisons": 1,
            "flagged_metrics": list(summary.get("flagged_metrics", [])),
            "interpretation": (
                "Calibration and evaluation LIME seeds are disjoint. "
                "SEPARATED is paired evidence of a change relative to the "
                "fixed-model control, not evidence of harm."
            ),
        }

    required = (
        "status",
        "matched_cross_version_comparisons",
        "probe_prediction_agreement",
        "noise_matched_cross_version_means",
        "paired_seed_cross_version_means",
        "any_metric_breach_rate",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(
            "LIME comparison summary is missing: " + ", ".join(missing)
        )
    noise_matched = summary["noise_matched_cross_version_means"]
    paired_seed = summary["paired_seed_cross_version_means"]
    return {
        "dataset_id": dataset_id,
        "dataset": DATASET_LABELS.get(dataset_id, dataset_id),
        "method": "LIME",
        "evidence_type": "noise-calibrated cross-version comparison",
        "status": str(summary["status"]),
        "observations": int(summary["matched_cross_version_comparisons"]),
        "prediction_agreement": float(summary["probe_prediction_agreement"]),
        "cosine_similarity": float(noise_matched["cosine_similarity"]),
        "paired_seed_cosine_similarity": float(
            paired_seed["cosine_similarity"]
        ),
        "headline_metric": "any_metric_breach_rate",
        "headline_value": float(summary["any_metric_breach_rate"]),
        "control_review_rate": None,
        "strongest_scenario": None,
        "separated_scenarios": None,
        "scenario_comparisons": None,
        "flagged_metrics": list(summary.get("flagged_metrics", [])),
        "interpretation": (
            "Noise-calibrated advisory evidence. REVIEW selects explanations "
            "for investigation and is not proof of a defective model."
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-Dataset SHAP and LIME Summary",
        "",
        f"Coverage: **{summary['coverage']['completed_cells']} / "
        f"{summary['coverage']['expected_cells']}** dataset-method cells",
        "",
        "| Dataset | Method | Evidence | Observations | Prediction agreement | Cosine similarity | Headline result | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["entries"]:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['evidence_type']} | "
            f"{row['observations']} | {row['prediction_agreement']:.3f} | "
            f"{row['cosine_similarity']:.3f} | "
            f"{row['headline_metric'].replace('_', ' ')}: "
            f"{row['headline_value']:.3f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            summary["interpretation"],
            "",
            "SHAP benchmark review rates and LIME breach rates answer different "
            "questions. They are displayed together for coverage, not ranked as "
            "interchangeable measurements.",
            "",
        ]
    )
    return "\n".join(lines)


def run_cross_dataset_summary(
    *,
    breast_shap_report: str | Path,
    wine_shap_report: str | Path,
    breast_lime_report: str | Path,
    wine_lime_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a 2-dataset by 2-method evidence portfolio from existing reports."""
    entries = [
        normalize_shap_benchmark(
            _read_json(breast_shap_report), dataset_id="breast_cancer"
        ),
        normalize_lime_comparison(
            _read_json(breast_lime_report), dataset_id="breast_cancer"
        ),
        normalize_shap_benchmark(
            _read_json(wine_shap_report), dataset_id="wine_binary"
        ),
        normalize_lime_comparison(
            _read_json(wine_lime_report), dataset_id="wine_binary"
        ),
    ]
    summary = {
        "schema_version": 1,
        "title": "Cross-dataset SHAP and LIME evidence summary",
        "coverage": {
            "datasets": 2,
            "methods": 2,
            "completed_cells": len(entries),
            "expected_cells": 4,
        },
        "entries": entries,
        "interpretation": (
            "The completed experiments show method- and dataset-dependent "
            "explanation signals while prediction agreement remains high. The "
            "portfolio is descriptive and does not define a universal threshold."
        ),
        "comparability_note": (
            "SHAP summarizes the strongest of several held-out model-change "
            "scenarios; LIME summarizes one paired held-out candidate. "
            "Headline rates must not be compared as if they were identical "
            "experimental contrasts."
        ),
        "sources": {
            "breast_cancer_shap": str(Path(breast_shap_report)),
            "breast_cancer_lime": str(Path(breast_lime_report)),
            "wine_shap": str(Path(wine_shap_report)),
            "wine_lime": str(Path(wine_lime_report)),
        },
    }

    reports = Path(output_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports / "report.md").write_text(_markdown(summary), encoding="utf-8")
    fieldnames = list(
        dict.fromkeys(key for entry in entries for key in entry)
    )
    with (reports / "cross_dataset_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)

    print("Cross-dataset SHAP and LIME summary")
    print("-----------------------------------")
    print("Coverage: 4 / 4 dataset-method cells")
    print(f"Reports saved to: {reports}")
    return summary
