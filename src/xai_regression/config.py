"""Load and validate experiment configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    test_size: float
    split_seed: int
    probe_size: int


@dataclass(frozen=True)
class ModelConfig:
    family: str
    n_estimators: int
    class_weight: str | None
    reference_seed: int
    benign_seeds: tuple[int, ...]
    candidate_random_seed: int
    candidate_max_depth: int | None
    candidate_min_samples_leaf: int


@dataclass(frozen=True)
class ExplainerConfig:
    method: str
    target_class: str


@dataclass(frozen=True)
class ComparisonConfig:
    metrics: tuple[str, ...]
    top_k: int


@dataclass(frozen=True)
class PolicyConfig:
    lower_quantile: float
    upper_quantile: float


@dataclass(frozen=True)
class ExperimentConfig:
    project_name: str
    experiment_id: str
    output_dir: Path
    dataset: DatasetConfig
    model: ModelConfig
    explainer: ExplainerConfig
    comparison: ComparisonConfig
    policy: PolicyConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid '{name}' section.")
    return value


def _validate(config: ExperimentConfig) -> None:
    if not 0 < config.dataset.test_size < 1:
        raise ConfigError("dataset.test_size must be between 0 and 1.")
    if config.dataset.probe_size < 1:
        raise ConfigError("dataset.probe_size must be at least 1.")
    if config.model.n_estimators < 1:
        raise ConfigError("model.n_estimators must be at least 1.")
    if not config.model.benign_seeds:
        raise ConfigError("model.benign_seeds must contain at least one seed.")
    if config.comparison.top_k < 1:
        raise ConfigError("comparison.top_k must be at least 1.")
    if not config.comparison.metrics:
        raise ConfigError("comparison.metrics must contain at least one metric.")
    if not 0 < config.policy.lower_quantile < config.policy.upper_quantile < 1:
        raise ConfigError(
            "policy quantiles must satisfy 0 < lower_quantile < "
            "upper_quantile < 1."
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML file and return a validated experiment configuration."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError("The configuration root must be a mapping.")

    project = _section(raw, "project")
    experiment = _section(raw, "experiment")
    dataset = _section(raw, "dataset")
    model = _section(raw, "model")
    candidate = _section(model, "candidate")
    explainer = _section(raw, "explainer")
    comparison = _section(raw, "comparison")
    policy = _section(raw, "policy")

    try:
        config = ExperimentConfig(
            project_name=str(project["name"]),
            experiment_id=str(experiment["id"]),
            output_dir=Path(experiment["output_dir"]),
            dataset=DatasetConfig(
                name=str(dataset["name"]),
                test_size=float(dataset["test_size"]),
                split_seed=int(dataset["split_seed"]),
                probe_size=int(dataset["probe_size"]),
            ),
            model=ModelConfig(
                family=str(model["family"]),
                n_estimators=int(model["n_estimators"]),
                class_weight=model.get("class_weight"),
                reference_seed=int(model["reference_seed"]),
                benign_seeds=tuple(int(seed) for seed in model["benign_seeds"]),
                candidate_random_seed=int(candidate["random_seed"]),
                candidate_max_depth=(
                    None
                    if candidate.get("max_depth") is None
                    else int(candidate["max_depth"])
                ),
                candidate_min_samples_leaf=int(candidate["min_samples_leaf"]),
            ),
            explainer=ExplainerConfig(
                method=str(explainer["method"]),
                target_class=str(explainer["target_class"]),
            ),
            comparison=ComparisonConfig(
                metrics=tuple(str(metric) for metric in comparison["metrics"]),
                top_k=int(comparison["top_k"]),
            ),
            policy=PolicyConfig(
                lower_quantile=float(policy["lower_quantile"]),
                upper_quantile=float(policy["upper_quantile"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"Invalid configuration value: {error}") from error

    _validate(config)
    return config
