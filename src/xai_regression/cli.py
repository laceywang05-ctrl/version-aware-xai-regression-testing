"""Command-line interface for reproducible XAI regression experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from xai_regression.benchmark import run_version_benchmark
from xai_regression.config import ConfigError, load_config
from xai_regression.lime_workflow import (
    run_lime_cross_version,
    run_lime_stability,
)
from xai_regression.pipeline import run_pipeline
from xai_regression.portfolio import run_cross_dataset_summary
from xai_regression.runner import run_validation
from xai_regression.reporting import run_unified_report
from xai_regression.server import serve_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_REPORTS = (
    "artifacts/experiment_10_heldout_benchmark/reports/summary.json",
    "artifacts/experiment_11_wine_heldout_benchmark/reports/summary.json",
)
DEFAULT_PORTFOLIO_REPORT = (
    "artifacts/experiment_14_cross_dataset_summary/reports/summary.json"
)


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
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Run SHAP, LIME and unified reporting as one workflow.",
    )
    workflow_parser.add_argument(
        "--shap-config",
        default="config/default.yaml",
    )
    workflow_parser.add_argument(
        "--lime-config",
        default="config/lime_cross_version.yaml",
    )
    workflow_parser.add_argument(
        "--output-dir",
        default="artifacts/experiment_08_unified_report",
    )
    workflow_parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the dashboard with the newly generated report.",
    )
    workflow_parser.add_argument("--host", default="127.0.0.1")
    workflow_parser.add_argument("--port", type=int, default=8000)
    workflow_parser.add_argument(
        "--benchmark-report",
        action="append",
        default=None,
        help="Optional benchmark summary JSON; repeat for multiple datasets.",
    )
    serve_parser = subparsers.add_parser(
        "serve",
        help="Serve the dashboard with an existing unified report.",
    )
    serve_parser.add_argument(
        "--report",
        default="artifacts/experiment_08_unified_report/reports/summary.json",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--benchmark-report",
        action="append",
        default=None,
        help="Optional benchmark summary JSON; repeat for multiple datasets.",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate the SHAP policy across controlled model-version scenarios.",
    )
    benchmark_parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the base SHAP experiment YAML file.",
    )
    benchmark_parser.add_argument(
        "--scenarios",
        default="config/version_benchmark.yaml",
        help="Path to the controlled version-scenario YAML file.",
    )
    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="Summarize SHAP and LIME evidence across both datasets.",
    )
    portfolio_parser.add_argument(
        "--breast-shap-report",
        default="artifacts/experiment_10_heldout_benchmark/reports/summary.json",
    )
    portfolio_parser.add_argument(
        "--wine-shap-report",
        default="artifacts/experiment_11_wine_heldout_benchmark/reports/summary.json",
    )
    portfolio_parser.add_argument(
        "--breast-lime-report",
        default="artifacts/experiment_07_lime_cross_version/reports/summary.json",
    )
    portfolio_parser.add_argument(
        "--wine-lime-report",
        default="artifacts/experiment_13_wine_lime_cross_version/reports/summary.json",
    )
    portfolio_parser.add_argument(
        "--output-dir",
        default="artifacts/experiment_14_cross_dataset_summary",
    )
    serve_parser.add_argument(
        "--portfolio-report",
        default=None,
        help="Optional cross-dataset summary JSON.",
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


def _benchmark_paths(values: Sequence[str] | None) -> list[Path]:
    candidates = values if values is not None else DEFAULT_BENCHMARK_REPORTS
    paths = [_project_path(value) for value in candidates]
    if values is None:
        return [path for path in paths if path.is_file()]
    return paths


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


def run_workflow(
    shap_config: str | Path,
    lime_config: str | Path,
    output_dir: str | Path,
    *,
    serve: bool,
    host: str,
    port: int,
    benchmark_reports: Sequence[str] | None = None,
) -> int:
    """Run the complete workflow and optionally start the local dashboard."""
    result = run_pipeline(
        shap_config_path=_project_path(shap_config),
        lime_config_path=_project_path(lime_config),
        output_dir=_project_path(output_dir),
        project_root=PROJECT_ROOT,
    )
    if serve:
        serve_dashboard(
            report_path=result["report_path"],
            dashboard_dir=PROJECT_ROOT / "dashboard",
            benchmark_paths=_benchmark_paths(benchmark_reports),
            portfolio_path=(
                _project_path(DEFAULT_PORTFOLIO_REPORT)
                if _project_path(DEFAULT_PORTFOLIO_REPORT).is_file()
                else None
            ),
            host=host,
            port=port,
        )
    return 0


def run_dashboard(
    report: str | Path,
    *,
    host: str,
    port: int,
    benchmark_reports: Sequence[str] | None = None,
    portfolio_report: str | Path | None = None,
) -> int:
    """Serve the dashboard with a previously generated unified report."""
    serve_dashboard(
        report_path=_project_path(report),
        dashboard_dir=PROJECT_ROOT / "dashboard",
        benchmark_paths=_benchmark_paths(benchmark_reports),
        portfolio_path=(
            _project_path(portfolio_report)
            if portfolio_report is not None
            else (
                _project_path(DEFAULT_PORTFOLIO_REPORT)
                if _project_path(DEFAULT_PORTFOLIO_REPORT).is_file()
                else None
            )
        ),
        host=host,
        port=port,
    )
    return 0


def run_portfolio(
    breast_shap_report: str | Path,
    wine_shap_report: str | Path,
    breast_lime_report: str | Path,
    wine_lime_report: str | Path,
    output_dir: str | Path,
) -> int:
    """Build the cross-dataset evidence summary from completed reports."""
    run_cross_dataset_summary(
        breast_shap_report=_project_path(breast_shap_report),
        wine_shap_report=_project_path(wine_shap_report),
        breast_lime_report=_project_path(breast_lime_report),
        wine_lime_report=_project_path(wine_lime_report),
        output_dir=_project_path(output_dir),
    )
    return 0


def run_benchmark(config_path: str | Path, scenario_path: str | Path) -> int:
    """Run the controlled multi-version SHAP benchmark."""
    config_file = _project_path(config_path)
    scenario_file = _project_path(scenario_path)
    config = load_config(config_file)
    run_version_benchmark(
        config,
        scenario_config_path=scenario_file,
        project_root=PROJECT_ROOT,
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
        if args.command == "workflow":
            return run_workflow(
                args.shap_config,
                args.lime_config,
                args.output_dir,
                serve=args.serve,
                host=args.host,
                port=args.port,
                benchmark_reports=args.benchmark_report,
            )
        if args.command == "serve":
            return run_dashboard(
                args.report,
                host=args.host,
                port=args.port,
                benchmark_reports=args.benchmark_report,
                portfolio_report=args.portfolio_report,
            )
        if args.command == "benchmark":
            return run_benchmark(args.config, args.scenarios)
        if args.command == "portfolio":
            return run_portfolio(
                args.breast_shap_report,
                args.wine_shap_report,
                args.breast_lime_report,
                args.wine_lime_report,
                args.output_dir,
            )
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    parser.error(f"Unsupported command: {args.command}")
    return 2
