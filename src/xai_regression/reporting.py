"""Unified reporting for SHAP and LIME regression experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS = (
    "cosine_similarity",
    "rank_correlation",
    "top_k_overlap",
    "direction_flip_rate",
)

BASELINE_FILES = {
    "SHAP": "benign_per_sample.csv",
    "LIME": "fixed_model_baseline.csv",
}

BASELINE_TYPES = {
    "SHAP": "benign model retraining",
    "LIME": "fixed-model explainer randomness",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required report file not found: {path}") from error


def normalize_method_reports(
    method: str,
    summary: dict[str, Any],
    policy: pd.DataFrame,
    baseline_samples: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize one explainer's outputs without hiding baseline semantics."""
    method = method.upper()
    if method not in BASELINE_TYPES:
        raise ValueError(f"Unsupported explanation method: {method}")

    if method == "SHAP":
        candidate_metrics = {
            key.removeprefix("mean_"): value
            for key, value in summary["candidate_explanation_metrics"].items()
        }
        prediction_agreement = summary["candidate_prediction_agreement"]
    else:
        candidate_metrics = summary["noise_matched_cross_version_means"]
        prediction_agreement = summary["probe_prediction_agreement"]

    method_row = {
        "method": method,
        "status": summary["status"],
        "baseline_type": BASELINE_TYPES[method],
        "probe_samples": int(summary["probe_samples"]),
        "prediction_agreement": float(prediction_agreement),
        "reference_accuracy": float(summary["reference_performance"]["accuracy"]),
        "candidate_accuracy": float(summary["candidate_performance"]["accuracy"]),
        "flagged_metrics": ", ".join(summary.get("flagged_metrics", [])),
    }

    policy_by_metric = policy.set_index("metric")
    metric_rows = []
    for metric in METRICS:
        policy_row = policy_by_metric.loc[metric]
        metric_rows.append(
            {
                "method": method,
                "baseline_type": BASELINE_TYPES[method],
                "metric": metric,
                "baseline_mean": float(baseline_samples[metric].mean()),
                "candidate_mean": float(candidate_metrics[metric]),
                "sample_limit": float(policy_row["sample_limit"]),
                "candidate_breach_rate": float(
                    policy_row["candidate_breach_rate"]
                ),
                "run_level_limit": float(
                    policy_row["run_level_limit_95_percent"]
                ),
                "review_flag": bool(policy_row["review_flag"]),
            }
        )
    return method_row, metric_rows


def _load_method(report_dir: Path, method: str):
    summary = _read_json(report_dir / "summary.json")
    policy = pd.read_csv(report_dir / "policy_evaluation.csv")
    baseline = pd.read_csv(report_dir / BASELINE_FILES[method])
    return normalize_method_reports(method, summary, policy, baseline)


def _save_figure(metrics: pd.DataFrame, path: Path) -> None:
    labels = [metric.replace("_", "\n") for metric in METRICS]
    positions = np.arange(len(METRICS))
    width = 0.18
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for method_index, method in enumerate(("SHAP", "LIME")):
        subset = metrics[metrics["method"] == method].set_index("metric")
        offset = (method_index - 0.5) * width * 2.2
        axes[0].bar(
            positions + offset - width / 2,
            [subset.loc[metric, "baseline_mean"] for metric in METRICS],
            width,
            alpha=0.65,
            label=f"{method} baseline",
        )
        axes[0].bar(
            positions + offset + width / 2,
            [subset.loc[metric, "candidate_mean"] for metric in METRICS],
            width,
            label=f"{method} candidate",
        )
        policy_positions = positions + offset
        axes[1].bar(
            policy_positions,
            [subset.loc[metric, "candidate_breach_rate"] for metric in METRICS],
            width * 1.6,
            alpha=0.75,
            label=f"{method} breach rate",
        )
        axes[1].scatter(
            policy_positions,
            [subset.loc[metric, "run_level_limit"] for metric in METRICS],
            marker="_",
            s=350,
            linewidths=3,
            label=f"{method} limit",
        )

    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Baseline and Candidate Explanation Metrics")
    axes[0].legend(fontsize=8)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Fraction of matched samples")
    axes[1].set_title("Policy Breach Rates and Run-Level Limits")
    axes[1].legend(fontsize=8)
    figure.suptitle("Unified SHAP and LIME Regression Report")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(
    overall_status: str,
    methods: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    lines = [
        "# Unified SHAP and LIME Regression Report",
        "",
        f"Overall advisory status: **{overall_status}**",
        "",
        "A review flag identifies an explanation change for investigation. It "
        "does not prove a software defect, harmful regression or concept drift.",
        "",
        "## Method summary",
        "",
        "| Method | Baseline | Probe samples | Prediction agreement | Status | Flagged metrics |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in methods.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['baseline_type']} | "
            f"{row['probe_samples']} | {row['prediction_agreement']:.3f} | "
            f"{row['status']} | {row['flagged_metrics'] or 'None'} |"
        )

    lines.extend(
        [
            "",
            "## Policy results",
            "",
            "| Method | Metric | Baseline mean | Candidate mean | Candidate breach rate | Run-level limit | Review |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in metrics.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['metric']} | "
            f"{row['baseline_mean']:.4f} | {row['candidate_mean']:.4f} | "
            f"{row['candidate_breach_rate']:.4f} | "
            f"{row['run_level_limit']:.4f} | "
            f"{'Yes' if row['review_flag'] else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "SHAP and LIME use different natural-variation baselines. Their "
            "raw thresholds should not be treated as interchangeable or ranked "
            "as if one explainer were universally more stable.",
            "",
        ]
    )
    return "\n".join(lines)


def run_unified_report(
    *,
    shap_report_dir: str | Path,
    lime_report_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Combine existing SHAP and LIME artifacts into one report bundle."""
    shap_method, shap_metrics = _load_method(Path(shap_report_dir), "SHAP")
    lime_method, lime_metrics = _load_method(Path(lime_report_dir), "LIME")
    methods = pd.DataFrame([shap_method, lime_method])
    metrics = pd.DataFrame([*shap_metrics, *lime_metrics])
    overall_status = (
        "REVIEW" if (methods["status"] == "REVIEW").any() else "WITHIN_BASELINE"
    )

    output = Path(output_dir)
    reports_dir = output / "reports"
    figures_dir = output / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    methods.to_csv(reports_dir / "method_summary.csv", index=False)
    metrics.to_csv(reports_dir / "metric_comparison.csv", index=False)
    (reports_dir / "report.md").write_text(
        _markdown_report(overall_status, methods, metrics),
        encoding="utf-8",
    )

    combined = {
        "overall_status": overall_status,
        "methods": methods.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "interpretation": (
            "Advisory synthesis. Method-specific baselines are preserved and "
            "must not be treated as interchangeable thresholds."
        ),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    figure_path = figures_dir / "unified_method_comparison.png"
    _save_figure(metrics, figure_path)

    print("Unified SHAP and LIME report")
    print("----------------------------")
    print(f"Overall status: {overall_status}")
    for row in methods.to_dict(orient="records"):
        print(
            f"{row['method']}: status={row['status']}, "
            f"flagged={row['flagged_metrics'] or 'none'}"
        )
    print(f"Reports saved to: {reports_dir}")
    print(f"Figure saved to: {figure_path}")
    return combined
