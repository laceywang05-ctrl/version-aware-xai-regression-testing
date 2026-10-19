"""Compare ordinary and multiplicity-aware Tree SHAP run decisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from xai_regression.benchmark import exact_mcnemar_p, wilson_interval
from xai_regression.multiplicity import seed_breach_rates


def evaluate_shap_run_rules(
    calibration_samples: pd.DataFrame,
    scenario_samples: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    dataset: str,
    model_family: str,
    split_seed: int,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate unadjusted and Bonferroni decisions for one benchmark cell."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    required = {"scenario_id", "random_seed"}
    if not required.issubset(scenario_samples.columns):
        raise ValueError("Scenario samples require scenario_id and random_seed.")
    base_policy = policy.drop_duplicates("metric").copy()
    metrics = base_policy["metric"].astype(str).tolist()
    if not metrics:
        raise ValueError("Policy must contain at least one metric.")
    calibration_rates = seed_breach_rates(calibration_samples, base_policy)
    scenario_rates = []
    for scenario_id, group in scenario_samples.groupby("scenario_id", sort=False):
        rates = seed_breach_rates(
            group.rename(columns={"random_seed": "seed"}),
            base_policy,
        )
        rates.insert(0, "scenario_id", scenario_id)
        scenario_rates.append(rates)
    rates = pd.concat(scenario_rates, ignore_index=True)
    control = rates[rates["scenario_id"] == "seed_only_control"].set_index("seed")
    candidate = rates[rates["scenario_id"] == "constrained_candidate"].set_index("seed")
    if control.empty or candidate.empty:
        raise ValueError("Both control and constrained candidate samples are required.")
    paired_seeds = control.index.intersection(candidate.index)
    if len(paired_seeds) != len(control) or len(paired_seeds) != len(candidate):
        raise ValueError("Control and candidate model seeds must match.")

    comparison_rows = []
    decision_rows = []
    for rule, quantile in (
        ("unadjusted", 1.0 - alpha),
        ("bonferroni", 1.0 - alpha / len(metrics)),
    ):
        limits = calibration_rates[metrics].quantile(quantile)
        flags = {}
        for scenario_id, frame in (
            ("seed_only_control", control.loc[paired_seeds]),
            ("constrained_candidate", candidate.loc[paired_seeds]),
        ):
            metric_flags = frame[metrics].gt(limits, axis="columns")
            review = metric_flags.any(axis=1)
            flags[scenario_id] = review.to_numpy(dtype=bool)
            for seed, row in metric_flags.iterrows():
                decision_rows.append(
                    {
                        "dataset": dataset,
                        "model_family": model_family,
                        "split_seed": split_seed,
                        "rule": rule,
                        "scenario_id": scenario_id,
                        "model_seed": int(seed),
                        "review_flag": bool(row.any()),
                        "flagged_metrics": ",".join(
                            metric for metric in metrics if bool(row[metric])
                        ),
                    }
                )
        control_flags = flags["seed_only_control"]
        candidate_flags = flags["constrained_candidate"]
        count = len(control_flags)
        control_count = int(control_flags.sum())
        candidate_count = int(candidate_flags.sum())
        control_ci = wilson_interval(control_count, count)
        candidate_ci = wilson_interval(candidate_count, count)
        clear_review, review_clear, p_value = exact_mcnemar_p(
            control_flags, candidate_flags
        )
        comparison_rows.append(
            {
                "dataset": dataset,
                "model_family": model_family,
                "split_seed": split_seed,
                "rule": rule,
                "run_limit_quantile": quantile,
                "paired_runs": count,
                "control_review_count": control_count,
                "control_review_rate": control_count / count,
                "control_ci_lower": control_ci[0],
                "control_ci_upper": control_ci[1],
                "candidate_review_count": candidate_count,
                "candidate_review_rate": candidate_count / count,
                "candidate_ci_lower": candidate_ci[0],
                "candidate_ci_upper": candidate_ci[1],
                "control_clear_candidate_review": clear_review,
                "control_review_candidate_clear": review_clear,
                "exact_mcnemar_p": p_value,
                "separated": p_value < alpha,
            }
        )
    return pd.DataFrame(comparison_rows), pd.DataFrame(decision_rows)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Tree SHAP Multiplicity-Aware Policy",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "| Dataset | Model | Split | Rule | Control review | Candidate review | p-value | Separated |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in summary["entries"]:
        lines.append(
            f"| {row['dataset']} | {row['model_family']} | {row['split_seed']} | "
            f"{row['rule']} | {row['control_review_rate']:.1%} | "
            f"{row['candidate_review_rate']:.1%} | "
            f"{row['exact_mcnemar_p']:.3g} | "
            f"{'yes' if row['separated'] else 'no'} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    return "\n".join(lines)


def run_shap_multiplicity_analysis(
    *,
    robustness_dir: str | Path,
    output_dir: str | Path,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Analyze all split-robustness cells under two run-level rules."""
    root = Path(robustness_dir)
    report_dirs = sorted(root.glob("runs/*/split_*/reports"))
    if not report_dirs:
        raise FileNotFoundError(f"No split-robustness reports found under {root}.")
    comparisons = []
    decisions = []
    for reports in report_dirs:
        required = (
            "benign_per_sample.csv",
            "scenario_per_sample.csv",
            "scenario_policy_results.csv",
            "summary.json",
        )
        missing = [name for name in required if not (reports / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing SHAP multiplicity inputs in {reports}: "
                + ", ".join(missing)
            )
        metadata = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
        split_seed = int(reports.parent.name.removeprefix("split_"))
        comparison, decision = evaluate_shap_run_rules(
            pd.read_csv(reports / "benign_per_sample.csv"),
            pd.read_csv(reports / "scenario_per_sample.csv"),
            pd.read_csv(reports / "scenario_policy_results.csv"),
            dataset=str(metadata["dataset"]),
            model_family=str(metadata["model_family"]),
            split_seed=split_seed,
            alpha=alpha,
        )
        comparisons.append(comparison)
        decisions.append(decision)
    comparison = pd.concat(comparisons, ignore_index=True)
    decision = pd.concat(decisions, ignore_index=True)
    by_rule = []
    for rule, group in comparison.groupby("rule", sort=False):
        by_rule.append(
            {
                "rule": rule,
                "cells": len(group),
                "separated_cells": int(group["separated"].sum()),
                "control_reviews": int(group["control_review_count"].sum()),
                "control_runs": int(group["paired_runs"].sum()),
                "pooled_control_review_rate": float(
                    group["control_review_count"].sum()
                    / group["paired_runs"].sum()
                ),
                "candidate_reviews": int(group["candidate_review_count"].sum()),
                "candidate_runs": int(group["paired_runs"].sum()),
                "pooled_candidate_review_rate": float(
                    group["candidate_review_count"].sum()
                    / group["paired_runs"].sum()
                ),
            }
        )
    bonferroni = next(row for row in by_rule if row["rule"] == "bonferroni")
    unadjusted = next(row for row in by_rule if row["rule"] == "unadjusted")
    status = (
        "CONTROLLED_AND_SENSITIVE"
        if bonferroni["separated_cells"] == bonferroni["cells"]
        and bonferroni["pooled_control_review_rate"]
        <= unadjusted["pooled_control_review_rate"]
        else "TRADE_OFF"
    )
    summary = {
        "schema_version": 1,
        "title": "Tree SHAP multiplicity-aware run policy",
        "status": status,
        "alpha": alpha,
        "metric_count": 4,
        "rule_summary": by_rule,
        "entries": comparison.to_dict(orient="records"),
        "interpretation": (
            "The Bonferroni run rule retains separation in every robustness "
            "cell while using equal or fewer control reviews."
            if status == "CONTROLLED_AND_SENSITIVE"
            else
            "Multiplicity adjustment changes the balance between control warnings "
            "and candidate sensitivity; the trade-off requires review."
        ),
    }
    reports = Path(output_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports / "report.md").write_text(_markdown(summary), encoding="utf-8")
    comparison.to_csv(reports / "rule_comparison.csv", index=False)
    decision.to_csv(reports / "run_decisions.csv", index=False)
    print("Tree SHAP multiplicity analysis")
    print("-------------------------------")
    print(f"Status: {status}")
    for row in by_rule:
        print(
            f"{row['rule']}: control={row['control_reviews']}/{row['control_runs']}, "
            f"candidate={row['candidate_reviews']}/{row['candidate_runs']}, "
            f"separated={row['separated_cells']}/{row['cells']}"
        )
    print(f"Reports saved to: {reports}")
    return summary
