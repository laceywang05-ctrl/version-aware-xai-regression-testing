"""Command-line interface for reproducible XAI regression experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from xai_regression.config import ConfigError, load_config
from xai_regression.lime_workflow import (
    run_lime_cross_version,
    run_lime_stability,
)
from xai_regression.runner import run_validation
from xai_regression.reporting import run_unified_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xai_regression",
        description="Run version-aware explanation regression experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the configured real-dataset validation workflow.",
    )
    run_parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the experiment YAML file.",
    )
    lime_parser = subparsers.add_parser(
        "lime-stability",
        help="Measure repeated-seed LIME variance for one fixed model.",
    )
    lime_parser.add_argument(
        "--config",
        default="config/lime_stability.yaml",
        help="Path to the LIME stability YAML file.",
    )
    compare_parser = subparsers.add_parser(
        "lime-compare",
        help="Compare two model versions against fixed-model LIME variation.",
    )
    compare_parser.add_argument(
        "--config",
        default="config/lime_cross_version.yaml",
        help="Path to the LIME cross-version YAML file.",
    )
    report_parser = subparsers.add_parser(
        "report",
        help="Combine existing SHAP and LIME experiment artifacts.",
    )
    report_parser.add_argument(
        "--shap-report-dir",
        default="artifacts/experiment_05_real_dataset/reports",
    )
    report_parser.add_argument(
        "--lime-report-dir",
        default="artifacts/experiment_07_lime_cross_version/reports",
    )
    report_parser.add_argument(
        "--output-dir",
        default="artifacts/experiment_08_unified_report",
    )
    return parser


def run_experiment(config_path: str | Path) -> int:
    """Validate a config and execute the current experiment workflow."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    config = load_config(path)
    print(f"Experiment: {config.experiment_id}", flush=True)
    print(f"Configuration: {path}", flush=True)
    print(f"Output directory: {config.output_dir}", flush=True)

    run_validation(
        config,
        config_path=path,
        project_root=PROJECT_ROOT,
    )
    return 0


def run_lime_experiment(config_path: str | Path) -> int:
    """Validate a config and execute the fixed-model LIME experiment."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    config = load_config(path)
    print(f"Experiment: {config.experiment_id}", flush=True)
    print(f"Configuration: {path}", flush=True)
    if config.lime is not None:
        print(f"Output directory: {config.lime.output_dir}", flush=True)

    run_lime_stability(
        config,
        config_path=path,
        project_root=PROJECT_ROOT,
    )
    return 0


def run_lime_comparison(config_path: str | Path) -> int:
    """Execute the LIME cross-version comparison workflow."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = load_config(path)
    print(f"Experiment: {config.experiment_id}", flush=True)
    print(f"Configuration: {path}", flush=True)
    run_lime_cross_version(
        config,
        config_path=path,
        project_root=PROJECT_ROOT,
    )
    return 0


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_report(
    shap_report_dir: str | Path,
    lime_report_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """Build one report from previously generated experiment artifacts."""
    run_unified_report(
        shap_report_dir=_project_path(shap_report_dir),
        lime_report_dir=_project_path(lime_report_dir),
        output_dir=_project_path(output_dir),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return run_experiment(args.config)
        if args.command == "lime-stability":
            return run_lime_experiment(args.config)
        if args.command == "lime-compare":
            return run_lime_comparison(args.config)
        if args.command == "report":
            return run_report(
                args.shap_report_dir,
                args.lime_report_dir,
                args.output_dir,
            )
    except ConfigError as error:
        parser.error(str(error))

    parser.error(f"Unsupported command: {args.command}")
    return 2
