"""Evaluate cross-model Tree SHAP evidence on additional data splits."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import yaml

from xai_regression.benchmark import VersionScenario, run_version_benchmark
from xai_regression.config import load_config


@dataclass(frozen=True)
class RobustnessCase:
    case_id: str
    dataset: str
    model_family: str
    base_config: Path
    candidate_max_depth: int
    candidate_min_samples_leaf: int


@dataclass(frozen=True)
class RobustnessPlan:
    output_dir: Path
    split_seeds: tuple[int, ...]
    heldout_model_seeds: tuple[int, ...]
    cases: tuple[RobustnessCase, ...]


def load_robustness_plan(path: str | Path) -> RobustnessPlan:
    """Load and validate the additional-split experiment plan."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Robustness configuration not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid robustness YAML: {error}") from error
    try:
        output_dir = Path(raw["experiment"]["output_dir"])
        split_seeds = tuple(int(seed) for seed in raw["split_seeds"])
        model_seeds = tuple(int(seed) for seed in raw["heldout_model_seeds"])
        cases = tuple(
            RobustnessCase(
                case_id=str(item["id"]),
                dataset=str(item["dataset"]),
                model_family=str(item["model_family"]),
                base_config=Path(item["base_config"]),
                candidate_max_depth=int(item["candidate_max_depth"]),
                candidate_min_samples_leaf=int(
                    item["candidate_min_samples_leaf"]
                ),
            )
            for item in raw["cases"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid robustness configuration value: {error}") from error
    if not split_seeds or not model_seeds or not cases:
        raise ValueError("Robustness splits, model seeds and cases cannot be empty.")
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("Robustness split seeds must be unique.")
    if len(set(case.case_id for case in cases)) != len(cases):
        raise ValueError("Robustness case ids must be unique.")
    return RobustnessPlan(output_dir, split_seeds, model_seeds, cases)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Tree SHAP Split Robustness",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "| Dataset | Model | Split | Control review | Candidate review | Agreement | Adjusted p | Separated |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["entries"]:
        lines.append(
            f"| {row['dataset']} | {row['model_family']} | {row['split_seed']} | "
            f"{row['control_review_rate']:.1%} | "
            f"{row['candidate_review_rate']:.1%} | "
            f"{row['prediction_agreement']:.1%} | "
            f"{row['adjusted_p']:.3g} | "
            f"{'yes' if row['separated'] else 'no'} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    return "\n".join(lines)


def run_shap_split_robustness(
    plan_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Run the preselected combined-change scenario on additional splits."""
    root = Path(project_root)
    plan = load_robustness_plan(plan_path)
    output_dir = plan.output_dir if plan.output_dir.is_absolute() else root / plan.output_dir
    entries = []
    for case in plan.cases:
        config_path = case.base_config if case.base_config.is_absolute() else root / case.base_config
        base_config = load_config(config_path)
        if base_config.dataset.name != case.dataset:
            raise ValueError(f"Case {case.case_id} dataset does not match its config.")
        if base_config.model.family != case.model_family:
            raise ValueError(f"Case {case.case_id} model family does not match its config.")
        scenarios = (
            VersionScenario(
                scenario_id="seed_only_control",
                change_type="control",
                description="Same settings with a held-out model seed",
                random_seeds=plan.heldout_model_seeds,
                max_depth=None,
                min_samples_leaf=1,
            ),
            VersionScenario(
                scenario_id="constrained_candidate",
                change_type="model_configuration",
                description="Preselected combined depth and leaf constraints",
                random_seeds=plan.heldout_model_seeds,
                max_depth=case.candidate_max_depth,
                min_samples_leaf=case.candidate_min_samples_leaf,
            ),
        )
        for split_seed in plan.split_seeds:
            run_config = replace(
                base_config,
                experiment_id=f"{base_config.experiment_id}-split-{split_seed}",
                dataset=replace(base_config.dataset, split_seed=split_seed),
            )
            run_dir = output_dir / "runs" / case.case_id / f"split_{split_seed}"
            result = run_version_benchmark(
                run_config,
                project_root=root,
                scenarios_override=scenarios,
                output_dir_override=run_dir,
            )
            control = next(
                row for row in result["scenario_results"]
                if row["change_type"] == "control"
            )
            candidate = next(
                row for row in result["scenario_results"]
                if row["scenario_id"] == "constrained_candidate"
            )
            comparison = result["paired_control_comparisons"][0]
            adjusted_p = float(comparison["bonferroni_adjusted_p"])
            entries.append(
                {
                    "case_id": case.case_id,
                    "dataset": case.dataset,
                    "model_family": case.model_family,
                    "split_seed": split_seed,
                    "paired_runs": int(comparison["paired_runs"]),
                    "control_review_rate": float(control["review_rate"]),
                    "candidate_review_rate": float(candidate["review_rate"]),
                    "review_rate_difference": float(
                        comparison["review_rate_difference"]
                    ),
                    "prediction_agreement": float(
                        candidate["mean_prediction_agreement"]
                    ),
                    "cosine_similarity": float(
                        candidate["mean_cosine_similarity"]
                    ),
                    "adjusted_p": adjusted_p,
                    "separated": adjusted_p < 0.05,
                }
            )

    separated = sum(row["separated"] for row in entries)
    status = "ROBUST" if separated == len(entries) else "SPLIT_SENSITIVE"
    summary = {
        "schema_version": 1,
        "title": "Cross-model Tree SHAP split robustness",
        "status": status,
        "coverage": {
            "cases": len(plan.cases),
            "additional_splits": len(plan.split_seeds),
            "comparisons": len(entries),
            "separated_comparisons": separated,
        },
        "entries": entries,
        "interpretation": (
            "The preselected combined structural change separates from benign "
            "retraining in every additional dataset, model-family and split cell."
            if status == "ROBUST"
            else
            "The preselected combined structural change does not separate in every "
            "additional split; the cross-model conclusion is split-sensitive."
        ),
    }
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports / "report.md").write_text(_markdown(summary), encoding="utf-8")
    with (reports / "split_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    print("Tree SHAP split robustness")
    print("--------------------------")
    print(f"Status: {status} ({separated}/{len(entries)} separated)")
    print(f"Reports saved to: {reports}")
    return summary
