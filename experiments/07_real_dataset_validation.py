"""Compatibility entry point for the real-dataset validation workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_regression.config import load_config  # noqa: E402
from xai_regression.runner import run_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate explanation regression on a real tabular dataset."
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the experiment YAML file.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    run_validation(
        config,
        config_path=config_path,
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
