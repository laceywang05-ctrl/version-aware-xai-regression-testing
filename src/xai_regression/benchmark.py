"""Evaluate the explanation policy across several controlled model versions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from xai_regression.config import ExperimentConfig
from xai_regression.runner import METRIC_DIRECTIONS, calibrate_policy
from xai_regression.workflow import (
    compare_explanations,
    load_dataset,
    positive_class_shap_values,
    predictive_metrics,
    train_reference_model,
    validate_supported_workflow,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class VersionScenario:
    """One controlled candidate model configuration."""

    scenario_id: str
    change_type: str
    description: str
    random_seeds: tuple[int, ...]
    max_depth: int | None
    min_samples_leaf: int


def load_scenarios(path: str | Path) -> tuple[Path, tuple[VersionScenario, ...]]:
    """Load and validate the version scenarios and benchmark output path."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Scenario configuration not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid scenario YAML in {config_path}: {error}") from error

    if not isinstance(raw, dict) or not isinstance(raw.get("benchmark"), dict):
        raise ValueError("Scenario configuration requires a 'benchmark' section.")
    raw_scenarios = raw.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("Scenario configuration requires a non-empty 'scenarios' list.")

    scenarios = []
    seen_ids: set[str] = set()
    try:
        output_dir = Path(raw["benchmark"]["output_dir"])
        for item in raw_scenarios:
            if not isinstance(item, dict):
                raise ValueError("Each scenario must be a mapping.")
            scenario_id = str(item["id"])
            if scenario_id in seen_ids:
                raise ValueError(f"Duplicate scenario id: {scenario_id}")
            seen_ids.add(scenario_id)
            min_samples_leaf = int(item["min_samples_leaf"])
            if min_samples_leaf < 1:
                raise ValueError("min_samples_leaf must be at least 1.")
            max_depth = item.get("max_depth")
            if max_depth is not None and int(max_depth) < 1:
                raise ValueError("max_depth must be null or at least 1.")
            random_seeds = tuple(
                int(seed)
                for seed in item.get(
                    "random_seeds",
                    [item["random_seed"]] if "random_seed" in item else [],
                )
            )
            if not random_seeds:
                raise ValueError("random_seeds must contain at least one seed.")
            scenarios.append(
                VersionScenario(
                    scenario_id=scenario_id,
                    change_type=str(item["change_type"]),
                    description=str(item["description"]),
                    random_seeds=random_seeds,
                    max_depth=None if max_depth is None else int(max_depth),
                    min_samples_leaf=min_samples_leaf,
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            (
                "Duplicate",
                "min_samples",
                "max_depth",
                "Each scenario",
                "random_seeds",
            )
        ):
            raise
        raise ValueError(f"Invalid scenario configuration value: {error}") from error
    return output_dir, tuple(scenarios)


def _train_scenario_model(
    config: ExperimentConfig,
    scenario: VersionScenario,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=scenario.max_depth,
        min_samples_leaf=scenario.min_samples_leaf,
        class_weight=config.model.class_weight,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _build_benign_baseline(
    config: ExperimentConfig,
    dataset,
    reference_values: np.ndarray,
    reference_predictions: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for seed in config.model.benign_seeds:
        model = train_reference_model(
            config,
            seed,
            dataset.X_train,
            dataset.y_train,
        )
        values = positive_class_shap_values(model, dataset.X_probe)
        predictions = model.predict(dataset.X_probe)
        comparison = compare_explanations(
            config,
            reference_values,
            values,
            reference_predictions != predictions,
        )
        comparison.insert(0, "seed", seed)
        frames.append(comparison)
    return pd.concat(frames, ignore_index=True)


def aggregate_scenario_runs(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated candidate seeds into one row per scenario."""
    rows = []
    for scenario_id, group in run_summary.groupby("scenario_id", sort=False):
        flagged_counts: dict[str, int] = {}
        for value in group["flagged_metrics"]:
            for metric in filter(None, str(value).split(",")):
                flagged_counts[metric] = flagged_counts.get(metric, 0) + 1
        review_count = int(group["review_flag"].sum())
        review_rate = float(group["review_flag"].mean())
        ci_lower, ci_upper = wilson_interval(review_count, len(group))
        rows.append(
            {
                "scenario_id": scenario_id,
                "change_type": group["change_type"].iloc[0],
                "description": group["description"].iloc[0],
                "runs": len(group),
                "mean_accuracy": float(group["accuracy"].mean()),
                "std_accuracy": float(group["accuracy"].std(ddof=0)),
                "mean_prediction_agreement": float(
                    group["prediction_agreement"].mean()
                ),
                "std_prediction_agreement": float(
                    group["prediction_agreement"].std(ddof=0)
                ),
                "mean_cosine_similarity": float(
                    group["mean_cosine_similarity"].mean()
                ),
                "std_cosine_similarity": float(
                    group["mean_cosine_similarity"].std(ddof=0)
                ),
                "review_count": review_count,
                "review_rate": review_rate,
                "review_rate_ci_lower": ci_lower,
                "review_rate_ci_upper": ci_upper,
                "flagged_metric_frequency": ",".join(
                    f"{metric}:{count}"
                    for metric, count in sorted(flagged_counts.items())
                ),
            }
        )
    return pd.DataFrame(rows)


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")

    proportion = successes / trials
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2)
    z_squared = z_score**2
    denominator = 1 + z_squared / trials
    centre = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        z_score
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z_squared / (4 * trials**2)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def exact_mcnemar_p(control_flags, candidate_flags) -> tuple[int, int, float]:
    """Compare paired review flags with an exact two-sided McNemar test."""
    control = np.asarray(control_flags, dtype=bool)
    candidate = np.asarray(candidate_flags, dtype=bool)
    if control.shape != candidate.shape:
        raise ValueError("Paired review flags must have matching shapes.")
    control_clear_candidate_review = int((~control & candidate).sum())
    control_review_candidate_clear = int((control & ~candidate).sum())
    discordant = control_clear_candidate_review + control_review_candidate_clear
    if discordant == 0:
        return control_clear_candidate_review, control_review_candidate_clear, 1.0
    smaller = min(control_clear_candidate_review, control_review_candidate_clear)
    lower_tail = sum(
        math.comb(discordant, count) for count in range(smaller + 1)
    ) / (2**discordant)
    return (
        control_clear_candidate_review,
        control_review_candidate_clear,
        min(1.0, 2 * lower_tail),
    )


def build_paired_control_comparisons(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare each changed scenario with the control using matched seeds."""
    control_ids = run_summary.loc[
        run_summary["change_type"] == "control", "scenario_id"
    ].unique()
    if len(control_ids) != 1:
        raise ValueError("Exactly one control scenario is required.")
    pivot = run_summary.pivot(
        index="random_seed",
        columns="scenario_id",
        values="review_flag",
    )
    control_id = control_ids[0]
    changed_ids = run_summary.loc[
        run_summary["change_type"] != "control", "scenario_id"
    ].drop_duplicates()
    rows = []
    comparison_count = len(changed_ids)
    for scenario_id in changed_ids:
        paired = pivot[[control_id, scenario_id]].dropna().astype(bool)
        control_clear, control_review, p_value = exact_mcnemar_p(
            paired[control_id],
            paired[scenario_id],
        )
        rows.append(
            {
                "control_scenario_id": control_id,
                "candidate_scenario_id": scenario_id,
                "paired_runs": len(paired),
                "control_review_rate": float(paired[control_id].mean()),
                "candidate_review_rate": float(paired[scenario_id].mean()),
                "review_rate_difference": float(
                    paired[scenario_id].mean() - paired[control_id].mean()
                ),
                "control_clear_candidate_review": control_clear,
                "control_review_candidate_clear": control_review,
                "exact_mcnemar_p": p_value,
                "bonferroni_adjusted_p": min(1.0, p_value * comparison_count),
            }
        )
    return pd.DataFrame(rows)


def save_benchmark_figure(summary: pd.DataFrame, path: Path) -> None:
    """Visualise mean prediction agreement and explanation review rate."""
    positions = np.arange(len(summary))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        positions,
        summary["mean_prediction_agreement"],
        yerr=summary["std_prediction_agreement"],
        capsize=4,
        color="#3A86A8",
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Prediction agreement")
    axes[0].set_xticks(positions, summary["scenario_id"], rotation=25, ha="right")
    axes[0].set_title("Prediction behaviour")

    colors = [
        "#D97732" if value else "#67A878" for value in summary["review_rate"]
    ]
    lower_error = summary["review_rate"] - summary["review_rate_ci_lower"]
    upper_error = summary["review_rate_ci_upper"] - summary["review_rate"]
    axes[1].bar(
        positions,
        summary["review_rate"],
        yerr=np.vstack([lower_error, upper_error]),
        capsize=4,
        color=colors,
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Fraction of runs flagged for review")
    axes[1].set_xticks(positions, summary["scenario_id"], rotation=25, ha="right")
    axes[1].set_title("Explanation policy response")
    figure.suptitle("Controlled Model-Version Benchmark")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_version_benchmark(
    config: ExperimentConfig,
    *,
    scenario_config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Run controlled candidates against one calibrated SHAP policy."""
    validate_supported_workflow(config)
    root = Path(project_root)
    scenario_path = Path(scenario_config_path)
    output_dir, scenarios = load_scenarios(scenario_path)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    report_dir = output_dir / "reports"
    figure_dir = output_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config)
    reference_model = train_reference_model(
        config,
        config.model.reference_seed,
        dataset.X_train,
        dataset.y_train,
    )
    reference_values = positive_class_shap_values(reference_model, dataset.X_probe)
    reference_predictions = reference_model.predict(dataset.X_probe)
    benign_samples = _build_benign_baseline(
        config,
        dataset,
        reference_values,
        reference_predictions,
    )

    run_rows = []
    policy_frames = []
    per_sample_frames = []
    for scenario in scenarios:
        for random_seed in scenario.random_seeds:
            model = _train_scenario_model(
                config,
                scenario,
                dataset.X_train,
                dataset.y_train,
                random_seed,
            )
            values = positive_class_shap_values(model, dataset.X_probe)
            predictions = model.predict(dataset.X_probe)
            comparison = compare_explanations(
                config,
                reference_values,
                values,
                reference_predictions != predictions,
            )
            policy, _ = calibrate_policy(config, benign_samples, comparison)
            policy.insert(0, "random_seed", random_seed)
            policy.insert(0, "scenario_id", scenario.scenario_id)
            policy_frames.append(policy)
            tagged_comparison = comparison.copy()
            tagged_comparison.insert(0, "random_seed", random_seed)
            tagged_comparison.insert(0, "scenario_id", scenario.scenario_id)
            per_sample_frames.append(tagged_comparison)
            performance = predictive_metrics(model, dataset.X_test, dataset.y_test)
            flagged_metrics = policy.loc[policy["review_flag"], "metric"].tolist()
            run_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "change_type": scenario.change_type,
                    "description": scenario.description,
                    "random_seed": random_seed,
                    "max_depth": scenario.max_depth,
                    "min_samples_leaf": scenario.min_samples_leaf,
                    **performance,
                    "prediction_agreement": float(
                        1 - comparison["prediction_changed"].mean()
                    ),
                    "mean_cosine_similarity": float(
                        comparison["cosine_similarity"].mean()
                    ),
                    "mean_rank_correlation": float(
                        comparison["rank_correlation"].mean()
                    ),
                    "mean_top_k_overlap": float(
                        comparison["top_k_overlap"].mean()
                    ),
                    "mean_direction_flip_rate": float(
                        comparison["direction_flip_rate"].mean()
                    ),
                    "flagged_metric_count": len(flagged_metrics),
                    "flagged_metrics": ",".join(flagged_metrics),
                    "review_flag": bool(flagged_metrics),
                }
            )

    run_summary = pd.DataFrame(run_rows)
    scenario_summary = aggregate_scenario_runs(run_summary)
    paired_comparisons = build_paired_control_comparisons(run_summary)
    policy_results = pd.concat(policy_frames, ignore_index=True)
    per_sample = pd.concat(per_sample_frames, ignore_index=True)
    scenario_summary.to_csv(report_dir / "scenario_summary.csv", index=False)
    run_summary.to_csv(report_dir / "scenario_run_summary.csv", index=False)
    paired_comparisons.to_csv(
        report_dir / "paired_control_comparisons.csv",
        index=False,
    )
    policy_results.to_csv(report_dir / "scenario_policy_results.csv", index=False)
    per_sample.to_csv(report_dir / "scenario_per_sample.csv", index=False)
    save_benchmark_figure(scenario_summary, figure_dir / "version_benchmark.png")

    control_rows = run_summary[run_summary["change_type"] == "control"]
    summary = {
        "benchmark": "controlled-model-version-scenarios",
        "dataset": config.dataset.name,
        "samples": int(len(dataset.X_train) + len(dataset.X_test)),
        "features": int(dataset.X_train.shape[1]),
        "probe_samples": int(len(dataset.X_probe)),
        "positive_class": config.explainer.target_class,
        "calibration_run_count": len(config.model.benign_seeds),
        "scenario_count": len(scenario_summary),
        "candidate_run_count": len(run_summary),
        "control_run_count": len(control_rows),
        "control_review_count": int(control_rows["review_flag"].sum()),
        "review_rate_interval": "Wilson 95% confidence interval",
        "paired_control_comparisons": paired_comparisons.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "scenario_results": scenario_summary.replace({np.nan: None}).to_dict(
            orient="records"
        ),
        "interpretation": (
            "Control warnings estimate false-positive behaviour. Warnings for configured "
            "model changes indicate sensitivity, not automatically harmful regressions."
        ),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Controlled model-version benchmark")
    print("----------------------------------")
    for row in scenario_summary.to_dict(orient="records"):
        print(
            f"{row['scenario_id']}: agreement={row['mean_prediction_agreement']:.3f}, "
            f"review={row['review_count']}/{row['runs']} "
            f"({row['flagged_metric_frequency'] or 'none'})"
        )
    print(f"Reports saved to: {report_dir}")
    print(f"Figure saved to: {figure_dir}")
    return summary
