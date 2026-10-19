"""Evaluate the LIME policy across dataset splits and probe subsets."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

from xai_regression.config import ExperimentConfig, load_config
from xai_regression.lime_workflow import run_lime_heldout_validation
from xai_regression.multiplicity import run_multiplicity_analysis


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class SplitRobustnessConfig:
    """Configuration for the cross-split confirmation experiment."""

    output_dir: Path
    family_alpha: float
    reference_split_seed: int
    new_split_seeds: tuple[int, ...]
    breast_config: Path
    wine_config: Path
    reference_policy_report: Path


def load_split_robustness_config(
    path: str | Path,
) -> SplitRobustnessConfig:
    """Load and validate a split-robustness scenario file."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"Split robustness configuration not found: {config_path}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid split robustness YAML in {config_path}: {error}"
        ) from error
    section = raw.get("robustness") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise ValueError("Configuration requires a 'robustness' section.")
    try:
        new_split_seeds = tuple(int(seed) for seed in section["new_split_seeds"])
        config = SplitRobustnessConfig(
            output_dir=Path(section["output_dir"]),
            family_alpha=float(section["family_alpha"]),
            reference_split_seed=int(section["reference_split_seed"]),
            new_split_seeds=new_split_seeds,
            breast_config=Path(section["breast_config"]),
            wine_config=Path(section["wine_config"]),
            reference_policy_report=Path(section["reference_policy_report"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid split robustness configuration value: {error}"
        ) from error
    if not 0 < config.family_alpha < 1:
        raise ValueError("family_alpha must be between zero and one.")
    if not config.new_split_seeds:
        raise ValueError("new_split_seeds must contain at least one seed.")
    if len(set(config.new_split_seeds)) != len(config.new_split_seeds):
        raise ValueError("new_split_seeds must not contain duplicates.")
    if config.reference_split_seed in config.new_split_seeds:
        raise ValueError("Reference and new split seeds must be distinct.")
    return config


def config_for_split(
    config: ExperimentConfig,
    *,
    split_seed: int,
    output_dir: Path,
) -> ExperimentConfig:
    """Return an immutable experiment configuration for one data split."""
    return replace(
        config,
        experiment_id=f"{config.dataset.name}-lime-split-{split_seed}",
        output_dir=output_dir,
        dataset=replace(config.dataset, split_seed=split_seed),
        lime=replace(config.lime, output_dir=output_dir),
    )


def _policy_rows(summary_path: Path, split_seed: int) -> list[dict[str, Any]]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Multiplicity policy report not found: {summary_path}"
        ) from error
    rows = summary.get("datasets")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid multiplicity summary: {summary_path}")
    selected = []
    for row in rows:
        if row.get("rule") == "bonferroni":
            selected.append({"split_seed": split_seed, **row})
    if len(selected) != 2:
        raise ValueError(
            f"Expected two Bonferroni dataset rows in {summary_path}."
        )
    return selected


def summarize_split_results(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize per-split evidence without treating splits as independent."""
    required = {
        "split_seed",
        "dataset",
        "control_review_rate",
        "candidate_review_rate",
        "status",
    }
    if not required.issubset(results.columns):
        raise ValueError("Split result table is missing required columns.")
    datasets = []
    for dataset, group in results.groupby("dataset", sort=True):
        candidate_exceeds_control = (
            group["candidate_review_rate"] > group["control_review_rate"]
        )
        datasets.append(
            {
                "dataset": dataset,
                "split_count": int(len(group)),
                "all_splits_separated": bool((group["status"] == "SEPARATED").all()),
                "candidate_exceeds_control_on_all_splits": bool(
                    candidate_exceeds_control.all()
                ),
                "control_review_rate_min": float(group["control_review_rate"].min()),
                "control_review_rate_max": float(group["control_review_rate"].max()),
                "candidate_review_rate_min": float(
                    group["candidate_review_rate"].min()
                ),
                "candidate_review_rate_max": float(
                    group["candidate_review_rate"].max()
                ),
            }
        )
    robust = all(
        row["all_splits_separated"]
        and row["candidate_exceeds_control_on_all_splits"]
        for row in datasets
    )
    return {
        "status": "ROBUST_ACROSS_SPLITS" if robust else "SPLIT_SENSITIVE",
        "datasets": datasets,
    }


def _save_figure(results: pd.DataFrame, path: Path) -> None:
    datasets = results["dataset"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 5.2))
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        subset = results[results["dataset"] == dataset].sort_values("split_seed")
        positions = np.arange(len(subset))
        width = 0.34
        axis.bar(
            positions - width / 2,
            subset["control_review_rate"],
            width,
            label="held-out control",
            color="#6baed6",
        )
        axis.bar(
            positions + width / 2,
            subset["candidate_review_rate"],
            width,
            label="candidate",
            color="#fd8d3c",
        )
        axis.set_xticks(positions, subset["split_seed"].astype(str))
        axis.set_xlabel("Dataset split seed")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Bonferroni-adjusted review rate")
        axis.set_title(dataset.replace("_", " ").title())
        axis.legend()
    figure.suptitle("LIME Policy Robustness Across Data Splits and Probes")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate_existing_split_results(
    scenario: SplitRobustnessConfig,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Aggregate completed policy reports without rerunning LIME."""
    output_dir = project_root / scenario.output_dir
    rows = _policy_rows(
        project_root / scenario.reference_policy_report,
        scenario.reference_split_seed,
    )
    for split_seed in scenario.new_split_seeds:
        policy_report = (
            output_dir
            / "policies"
            / f"split_{split_seed}"
            / "reports"
            / "summary.json"
        )
        rows.extend(_policy_rows(policy_report, split_seed))

    results = pd.DataFrame(rows).sort_values(["dataset", "split_seed"])
    aggregate = summarize_split_results(results)
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports_dir / "split_policy_results.csv", index=False)
    figure_path = figures_dir / "split_probe_robustness.png"
    _save_figure(results, figure_path)

    summary: dict[str, Any] = {
        "experiment_id": "lime-split-probe-robustness-v1",
        "status": aggregate["status"],
        "family_alpha": scenario.family_alpha,
        "split_seeds": [
            scenario.reference_split_seed,
            *scenario.new_split_seeds,
        ],
        "interpretation": (
            "Each split recalibrates the same Bonferroni-adjusted LIME policy. "
            "ROBUST_ACROSS_SPLITS means candidate review rates exceeded control "
            "rates with paired separation on every tested split; it does not "
            "establish a universal threshold or harmful model behaviour."
        ),
        "dataset_summaries": aggregate["datasets"],
        "split_results": results.to_dict(orient="records"),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("LIME split and probe robustness")
    print("-------------------------------")
    for row in results.itertuples(index=False):
        print(
            f"{row.dataset} / split {row.split_seed}: control="
            f"{row.control_review_count}/{row.run_count}, candidate="
            f"{row.candidate_review_count}/{row.run_count}, "
            f"p={row.exact_mcnemar_p:.6f}, status={row.status}"
        )
    print(f"Overall status: {summary['status']}")
    print(f"Reports saved to: {reports_dir}")
    print(f"Figure saved to: {figure_path}")
    return summary


def run_split_robustness(
    scenario_path: str | Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Run new splits, apply the fixed rule and aggregate the evidence."""
    scenario = load_split_robustness_config(scenario_path)
    output_dir = project_root / scenario.output_dir

    base_configs = {
        "breast": load_config(project_root / scenario.breast_config),
        "wine": load_config(project_root / scenario.wine_config),
    }
    for split_seed in scenario.new_split_seeds:
        report_dirs: dict[str, Path] = {}
        for label, base_config in base_configs.items():
            run_dir = output_dir / "runs" / f"{label}_split_{split_seed}"
            config = config_for_split(
                base_config,
                split_seed=split_seed,
                output_dir=run_dir.relative_to(project_root),
            )
            run_lime_heldout_validation(
                config,
                config_path=scenario_path,
                project_root=project_root,
            )
            report_dirs[label] = run_dir / "reports"

        policy_dir = output_dir / "policies" / f"split_{split_seed}"
        run_multiplicity_analysis(
            breast_report_dir=report_dirs["breast"],
            wine_report_dir=report_dirs["wine"],
            output_dir=policy_dir,
            alpha=scenario.family_alpha,
        )
    return aggregate_existing_split_results(scenario, project_root=project_root)
