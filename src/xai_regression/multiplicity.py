"""Compare ordinary and multiplicity-aware LIME run policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from xai_regression.benchmark import exact_mcnemar_p, wilson_interval


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REQUIRED_REPORTS = (
    "calibration_fixed_model.csv",
    "heldout_control_per_sample.csv",
    "heldout_candidate_per_sample.csv",
    "policy_evaluation.csv",
    "summary.json",
)


def _load_report_dir(report_dir: Path) -> tuple[
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    missing = [name for name in REQUIRED_REPORTS if not (report_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing held-out LIME report files in {report_dir}: "
            + ", ".join(missing)
        )
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    return (
        str(summary["dataset"]),
        pd.read_csv(report_dir / "calibration_fixed_model.csv"),
        pd.read_csv(report_dir / "heldout_control_per_sample.csv"),
        pd.read_csv(report_dir / "heldout_candidate_per_sample.csv"),
        pd.read_csv(report_dir / "policy_evaluation.csv"),
    )


def seed_breach_rates(
    samples: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate sample-level metric breaches into one row per random seed."""
    required_policy = {
        "metric",
        "sample_limit_direction",
        "sample_limit",
    }
    if not required_policy.issubset(policy.columns):
        raise ValueError("Policy table is missing sample-limit columns.")
    if "seed" not in samples.columns:
        raise ValueError("Sample table requires a seed column.")

    metrics = policy["metric"].astype(str).tolist()
    missing_metrics = [metric for metric in metrics if metric not in samples]
    if missing_metrics:
        raise ValueError(
            "Sample table is missing metrics: " + ", ".join(missing_metrics)
        )

    rows: list[dict[str, Any]] = []
    for seed, group in samples.groupby("seed", sort=True):
        row: dict[str, Any] = {"seed": int(seed)}
        for item in policy.itertuples(index=False):
            values = group[item.metric]
            if item.sample_limit_direction == "lower":
                breaches = values < float(item.sample_limit)
            elif item.sample_limit_direction == "upper":
                breaches = values > float(item.sample_limit)
            else:
                raise ValueError(
                    f"Unsupported sample limit direction: {item.sample_limit_direction}"
                )
            row[item.metric] = float(breaches.mean())
        rows.append(row)
    if not rows:
        raise ValueError("Sample table contains no seeds.")
    return pd.DataFrame(rows)


def evaluate_multiplicity_rules(
    calibration_samples: pd.DataFrame,
    control_samples: pd.DataFrame,
    candidate_samples: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    dataset: str,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate per-metric and Bonferroni run limits on held-out seeds."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    metrics = policy["metric"].astype(str).tolist()
    if not metrics:
        raise ValueError("Policy table must contain at least one metric.")

    calibration = seed_breach_rates(calibration_samples, policy)
    control = seed_breach_rates(control_samples, policy)
    candidate = seed_breach_rates(candidate_samples, policy)
    if control["seed"].tolist() != candidate["seed"].tolist():
        raise ValueError("Held-out control and candidate seeds must match.")

    rules = (
        ("unadjusted", 1.0 - alpha),
        ("bonferroni", 1.0 - alpha / len(metrics)),
    )
    comparison_rows: list[dict[str, Any]] = []
    limit_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for rule, quantile in rules:
        limits = calibration[metrics].quantile(quantile)
        for metric in metrics:
            limit_rows.append(
                {
                    "dataset": dataset,
                    "rule": rule,
                    "metric": metric,
                    "run_limit_quantile": quantile,
                    "run_level_limit": float(limits[metric]),
                }
            )

        flags: dict[str, np.ndarray] = {}
        for scenario, frame in (
            ("heldout_control", control),
            ("candidate", candidate),
        ):
            metric_flags = frame[metrics].gt(limits, axis="columns")
            review_flags = metric_flags.any(axis=1).to_numpy(dtype=bool)
            flags[scenario] = review_flags
            for position, (_, row) in enumerate(frame.iterrows()):
                flagged_metrics = [
                    metric for metric in metrics if bool(metric_flags.iloc[position][metric])
                ]
                decision_rows.append(
                    {
                        "dataset": dataset,
                        "rule": rule,
                        "scenario": scenario,
                        "seed": int(row["seed"]),
                        "review_flag": bool(review_flags[position]),
                        "flagged_metrics": ",".join(flagged_metrics),
                    }
                )

        control_flags = flags["heldout_control"]
        candidate_flags = flags["candidate"]
        run_count = len(control_flags)
        control_count = int(control_flags.sum())
        candidate_count = int(candidate_flags.sum())
        control_ci = wilson_interval(control_count, run_count)
        candidate_ci = wilson_interval(candidate_count, run_count)
        clear_review, review_clear, exact_p = exact_mcnemar_p(
            control_flags,
            candidate_flags,
        )
        comparison_rows.append(
            {
                "dataset": dataset,
                "rule": rule,
                "family_alpha": alpha,
                "run_limit_quantile": quantile,
                "run_count": run_count,
                "control_review_count": control_count,
                "control_review_rate": control_count / run_count,
                "control_ci_lower": control_ci[0],
                "control_ci_upper": control_ci[1],
                "candidate_review_count": candidate_count,
                "candidate_review_rate": candidate_count / run_count,
                "candidate_ci_lower": candidate_ci[0],
                "candidate_ci_upper": candidate_ci[1],
                "control_clear_candidate_review": clear_review,
                "control_review_candidate_clear": review_clear,
                "exact_mcnemar_p": exact_p,
                "status": "SEPARATED" if exact_p < alpha else "NO_SEPARATION",
            }
        )

    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(limit_rows),
        pd.DataFrame(decision_rows),
    )


def _save_figure(comparison: pd.DataFrame, path: Path) -> None:
    datasets = comparison["dataset"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.2))
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        subset = comparison[comparison["dataset"] == dataset].reset_index(drop=True)
        positions = np.arange(len(subset))
        width = 0.34
        control_rates = subset["control_review_rate"].to_numpy()
        candidate_rates = subset["candidate_review_rate"].to_numpy()
        control_error = np.vstack(
            [
                np.maximum(
                    control_rates - subset["control_ci_lower"].to_numpy(),
                    0.0,
                ),
                np.maximum(
                    subset["control_ci_upper"].to_numpy() - control_rates,
                    0.0,
                ),
            ]
        )
        candidate_error = np.vstack(
            [
                np.maximum(
                    candidate_rates - subset["candidate_ci_lower"].to_numpy(),
                    0.0,
                ),
                np.maximum(
                    subset["candidate_ci_upper"].to_numpy() - candidate_rates,
                    0.0,
                ),
            ]
        )
        axis.bar(
            positions - width / 2,
            control_rates,
            width,
            label="held-out control",
            color="#6baed6",
            yerr=control_error,
            capsize=5,
        )
        axis.bar(
            positions + width / 2,
            candidate_rates,
            width,
            label="candidate",
            color="#fd8d3c",
            yerr=candidate_error,
            capsize=5,
        )
        axis.set_xticks(positions, subset["rule"])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Run review rate")
        axis.set_title(dataset.replace("_", " ").title())
        axis.legend()
    figure.suptitle("LIME Policy: Unadjusted vs Multiplicity-Aware Limits")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_multiplicity_analysis(
    *,
    breast_report_dir: Path,
    wine_report_dir: Path,
    output_dir: Path,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run a two-dataset multiplicity-policy analysis from existing reports."""
    comparisons = []
    limits = []
    decisions = []
    for report_dir in (breast_report_dir, wine_report_dir):
        dataset, calibration, control, candidate, policy = _load_report_dir(
            report_dir
        )
        result = evaluate_multiplicity_rules(
            calibration,
            control,
            candidate,
            policy,
            dataset=dataset,
            alpha=alpha,
        )
        comparisons.append(result[0])
        limits.append(result[1])
        decisions.append(result[2])

    comparison = pd.concat(comparisons, ignore_index=True)
    metric_limits = pd.concat(limits, ignore_index=True)
    run_decisions = pd.concat(decisions, ignore_index=True)
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "policy_comparison.csv", index=False)
    metric_limits.to_csv(reports_dir / "metric_limits.csv", index=False)
    run_decisions.to_csv(reports_dir / "run_decisions.csv", index=False)

    figure_path = figures_dir / "multiplicity_policy_comparison.png"
    _save_figure(comparison, figure_path)
    summary: dict[str, Any] = {
        "experiment_id": "lime-multiplicity-aware-policy-v1",
        "status": "VERIFIED",
        "family_alpha": alpha,
        "metric_count": int(metric_limits["metric"].nunique()),
        "interpretation": (
            "The Bonferroni rule raises each metric's run-limit quantile to "
            "control family-wise review risk across the four monitored metrics. "
            "This is a diagnostic policy comparison, not proof that a reviewed "
            "candidate is harmful."
        ),
        "datasets": comparison.to_dict(orient="records"),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Multiplicity-aware LIME policy analysis")
    print("---------------------------------------")
    for row in comparison.itertuples(index=False):
        print(
            f"{row.dataset} / {row.rule}: control="
            f"{row.control_review_count}/{row.run_count}, candidate="
            f"{row.candidate_review_count}/{row.run_count}, "
            f"paired p={row.exact_mcnemar_p:.6f}, status={row.status}"
        )
    print(f"Reports saved to: {reports_dir}")
    print(f"Figure saved to: {figure_path}")
    return summary
