"""Evaluate the explanation policy across several controlled model versions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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
    random_seed: int
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
            scenarios.append(
                VersionScenario(
                    scenario_id=scenario_id,
                    change_type=str(item["change_type"]),
                    description=str(item["description"]),
                    random_seed=int(item["random_seed"]),
                    max_depth=None if max_depth is None else int(max_depth),
                    min_samples_leaf=min_samples_leaf,
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("Duplicate", "min_samples", "max_depth", "Each scenario")
        ):
            raise
        raise ValueError(f"Invalid scenario configuration value: {error}") from error
    return output_dir, tuple(scenarios)


def _train_scenario_model(
    config: ExperimentConfig,
    scenario: VersionScenario,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=scenario.max_depth,
        min_samples_leaf=scenario.min_samples_leaf,
        class_weight=config.model.class_weight,
        random_state=scenario.random_seed,
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


def save_benchmark_figure(summary: pd.DataFrame, path: Path) -> None:
    """Visualise predictive agreement and number of explanation warnings."""
    positions = np.arange(len(summary))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(positions, summary["prediction_agreement"], color="#3A86A8")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Prediction agreement")
    axes[0].set_xticks(positions, summary["scenario_id"], rotation=25, ha="right")
    axes[0].set_title("Prediction behaviour")

    colors = ["#D97732" if value else "#67A878" for value in summary["review_flag"]]
    axes[1].bar(positions, summary["flagged_metric_count"], color=colors)
    axes[1].set_ylim(0, max(4, int(summary["flagged_metric_count"].max()) + 1))
    axes[1].set_ylabel("Flagged explanation metrics")
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

    scenario_rows = []
    policy_frames = []
    per_sample_frames = []
    for scenario in scenarios:
        model = _train_scenario_model(
            config,
            scenario,
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
        policy, _ = calibrate_policy(config, benign_samples, comparison)
        policy.insert(0, "scenario_id", scenario.scenario_id)
        policy_frames.append(policy)
        tagged_comparison = comparison.copy()
        tagged_comparison.insert(0, "scenario_id", scenario.scenario_id)
        per_sample_frames.append(tagged_comparison)
        performance = predictive_metrics(model, dataset.X_test, dataset.y_test)
        flagged_metrics = policy.loc[policy["review_flag"], "metric"].tolist()
        scenario_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "change_type": scenario.change_type,
                "description": scenario.description,
                "random_seed": scenario.random_seed,
                "max_depth": scenario.max_depth,
                "min_samples_leaf": scenario.min_samples_leaf,
                **performance,
                "prediction_agreement": float(1 - comparison["prediction_changed"].mean()),
                "mean_cosine_similarity": float(comparison["cosine_similarity"].mean()),
                "mean_rank_correlation": float(comparison["rank_correlation"].mean()),
                "mean_top_k_overlap": float(comparison["top_k_overlap"].mean()),
                "mean_direction_flip_rate": float(comparison["direction_flip_rate"].mean()),
                "flagged_metric_count": len(flagged_metrics),
                "flagged_metrics": ",".join(flagged_metrics),
                "review_flag": bool(flagged_metrics),
            }
        )

    scenario_summary = pd.DataFrame(scenario_rows)
    policy_results = pd.concat(policy_frames, ignore_index=True)
    per_sample = pd.concat(per_sample_frames, ignore_index=True)
    scenario_summary.to_csv(report_dir / "scenario_summary.csv", index=False)
    policy_results.to_csv(report_dir / "scenario_policy_results.csv", index=False)
    per_sample.to_csv(report_dir / "scenario_per_sample.csv", index=False)
    save_benchmark_figure(scenario_summary, figure_dir / "version_benchmark.png")

    control_rows = scenario_summary[scenario_summary["change_type"] == "control"]
    summary = {
        "benchmark": "controlled-model-version-scenarios",
        "scenario_count": len(scenario_summary),
        "control_count": len(control_rows),
        "control_review_count": int(control_rows["review_flag"].sum()),
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
    for row in scenario_rows:
        print(
            f"{row['scenario_id']}: agreement={row['prediction_agreement']:.3f}, "
            f"flagged={row['flagged_metric_count']} ({row['flagged_metrics'] or 'none'})"
        )
    print(f"Reports saved to: {report_dir}")
    print(f"Figure saved to: {figure_dir}")
    return summary
