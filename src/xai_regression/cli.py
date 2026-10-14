"""Command-line interface for reproducible XAI regression experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from xai_regression.config import ConfigError, load_config
from xai_regression.runner import run_validation


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return run_experiment(args.config)
    except ConfigError as error:
        parser.error(str(error))

    parser.error(f"Unsupported command: {args.command}")
    return 2
