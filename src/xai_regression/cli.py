"""Command-line interface for reproducible XAI regression experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from xai_regression.benchmark import run_version_benchmark
from xai_regression.config import ConfigError, load_config
from xai_regression.cifar10_calibration import run_cifar10_explanation_calibration
from xai_regression.digits_benchmark import (
    run_digits_close_version_benchmark,
    run_digits_explanation_split_robustness,
    run_digits_explanation_stability,
    run_digits_explanation_threshold_sensitivity,
    run_digits_split_robustness,
    run_digits_training_robustness,
)
from xai_regression.lime_workflow import (
    run_lime_cross_version,
    run_lime_heldout_validation,
    run_lime_stability,
)
from xai_regression.image_demo import build_digits_image_demo
from xai_regression.image_bundle import build_image_model_demo, prepare_image_npz_bundle
from xai_regression.regression_demo import build_regression_demo
from xai_regression.multiplicity import run_multiplicity_analysis
from xai_regression.model_family import run_model_family_summary
from xai_regression.mnist_calibration import run_mnist_explanation_calibration
from xai_regression.pipeline import run_pipeline
from xai_regression.portfolio import run_cross_dataset_summary
from xai_regression.runner import run_validation
from xai_regression.reporting import run_unified_report
from xai_regression.server import serve_dashboard
from xai_regression.shap_split_robustness import run_shap_split_robustness
from xai_regression.shap_multiplicity import run_shap_multiplicity_analysis
from xai_regression.split_robustness import run_split_robustness


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_REPORTS = (
    "artifacts/experiment_10_heldout_benchmark/reports/summary.json",
    "artifacts/experiment_11_wine_heldout_benchmark/reports/summary.json",
)
DEFAULT_PORTFOLIO_REPORT = (
    "artifacts/experiment_14_cross_dataset_summary/reports/summary.json"
)
DEFAULT_ROBUSTNESS_REPORT = (
    "artifacts/experiment_27_lime_split_probe_robustness/reports/summary.json"
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
    heldout_parser = subparsers.add_parser(
        "lime-heldout",
        help="Validate a LIME policy on disjoint calibration and evaluation seeds.",
    )
    heldout_parser.add_argument(
        "--config",
        default="config/lime_heldout.yaml",
        help="Path to the held-out LIME validation YAML file.",
    )
    multiplicity_parser = subparsers.add_parser(
        "lime-multiplicity",
        help="Compare ordinary and multiplicity-aware held-out LIME policies.",
    )
    multiplicity_parser.add_argument(
        "--breast-report-dir",
        default="artifacts/experiment_15_breast_lime_heldout/reports",
    )
    multiplicity_parser.add_argument(
        "--wine-report-dir",
        default="artifacts/experiment_16_wine_lime_heldout/reports",
    )
    multiplicity_parser.add_argument(
        "--output-dir",
        default="artifacts/experiment_17_lime_multiplicity_policy",
    )
    multiplicity_parser.add_argument("--alpha", type=float, default=0.05)
    split_parser = subparsers.add_parser(
        "lime-split-robustness",
        help="Confirm the adjusted LIME policy across data splits and probes.",
    )
    split_parser.add_argument(
        "--config",
        default="config/lime_split_robustness.yaml",
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
        default="artifacts/experiment_15_breast_lime_heldout/reports/summary.json",
    )
    portfolio_parser.add_argument(
        "--wine-lime-report",
        default="artifacts/experiment_16_wine_lime_heldout/reports/summary.json",
    )
    portfolio_parser.add_argument(
        "--output-dir",
        default="artifacts/experiment_14_cross_dataset_summary",
    )
    family_parser = subparsers.add_parser(
        "model-family-summary",
        help="Summarize SHAP evidence across two datasets and model families.",
    )
    family_parser.add_argument(
        "--random-forest-breast-report",
        default="artifacts/experiment_10_heldout_benchmark/reports/summary.json",
    )
    family_parser.add_argument(
        "--random-forest-wine-report",
        default="artifacts/experiment_11_wine_heldout_benchmark/reports/summary.json",
    )
    family_parser.add_argument(
        "--gradient-boosting-breast-report",
        default="artifacts/experiment_28_breast_gradient_boosting/reports/summary.json",
    )
    family_parser.add_argument(
        "--gradient-boosting-wine-report",
        default="artifacts/experiment_29_wine_gradient_boosting/reports/summary.json",
    )
    family_parser.add_argument(
        "--output-dir",
        default="artifacts/experiment_30_model_family_summary",
    )
    shap_split_parser = subparsers.add_parser(
        "shap-split-robustness",
        help="Retest the cross-model SHAP result on additional data splits.",
    )
    shap_split_parser.add_argument(
        "--config",
        default="config/shap_split_robustness.yaml",
    )
    shap_multiplicity_parser = subparsers.add_parser(
        "shap-multiplicity",
        help="Compare ordinary and multiplicity-aware SHAP run policies.",
    )
    shap_multiplicity_parser.add_argument(
        "--robustness-dir",
        default="artifacts/experiment_31_shap_split_robustness",
    )
    shap_multiplicity_parser.add_argument(
        "--output-dir",
        default="artifacts/experiment_32_shap_multiplicity_policy",
    )
    shap_multiplicity_parser.add_argument("--alpha", type=float, default=0.05)
    image_demo_parser = subparsers.add_parser(
        "image-demo",
        help="Build a ready-to-upload handwritten-digit comparison bundle.",
    )
    image_demo_parser.add_argument(
        "--output-dir",
        default="artifacts/demo_image_upload",
    )
    image_bundle_parser = subparsers.add_parser(
        "prepare-image-data",
        help="Prepare MNIST/CIFAR-style NPZ arrays for dashboard upload.",
    )
    image_bundle_parser.add_argument("--input", required=True)
    image_bundle_parser.add_argument("--output-dir", default="artifacts/prepared_image_upload")
    image_bundle_parser.add_argument("--limit", type=int, default=500)
    image_model_demo_parser = subparsers.add_parser(
        "image-model-demo",
        help="Train four classifiers on image NPZ arrays or the official CIFAR-10 archive.",
    )
    image_model_demo_parser.add_argument("--input", required=True)
    image_model_demo_parser.add_argument("--output-dir", default="artifacts/image_model_demo")
    image_model_demo_parser.add_argument("--limit", type=int, default=1000)
    mnist_calibration_parser = subparsers.add_parser(
        "mnist-explanation-calibration",
        help="Calibrate MNIST explanation similarity under benign retraining.",
    )
    mnist_calibration_parser.add_argument("--input", required=True)
    mnist_calibration_parser.add_argument(
        "--output-dir", default="artifacts/mnist_explanation_calibration"
    )
    mnist_calibration_parser.add_argument("--train-limit", type=int, default=1000)
    mnist_calibration_parser.add_argument("--sample-limit", type=int, default=200)
    cifar10_calibration_parser = subparsers.add_parser(
        "cifar10-explanation-calibration",
        help="Calibrate CIFAR-10 RGB explanation similarity under benign retraining.",
    )
    cifar10_calibration_parser.add_argument("--input", required=True)
    cifar10_calibration_parser.add_argument(
        "--output-dir", default="artifacts/cifar10_explanation_calibration"
    )
    cifar10_calibration_parser.add_argument("--train-limit", type=int, default=1000)
    cifar10_calibration_parser.add_argument("--sample-limit", type=int, default=50)
    regression_demo_parser = subparsers.add_parser(
        "regression-demo",
        help="Build a ready-to-upload four-model regression bundle.",
    )
    regression_demo_parser.add_argument(
        "--output-dir",
        default="artifacts/demo_regression_upload",
    )
    digits_benchmark_parser = subparsers.add_parser(
        "digits-close-benchmark",
        help="Compare four closely matched random-forest digit models.",
    )
    digits_benchmark_parser.add_argument(
        "--output-dir",
        default="artifacts/digits_close_version_benchmark",
    )
    digits_robustness_parser = subparsers.add_parser(
        "digits-split-robustness",
        help="Repeat the four-version digit benchmark across ten data splits.",
    )
    digits_robustness_parser.add_argument(
        "--output-dir",
        default="artifacts/digits_split_robustness",
    )
    digits_training_parser = subparsers.add_parser(
        "digits-training-robustness",
        help="Repeat the four-version digit benchmark across training seeds.",
    )
    digits_training_parser.add_argument(
        "--output-dir",
        default="artifacts/digits_training_robustness",
    )
    digits_explanation_parser = subparsers.add_parser(
        "digits-explanation-stability",
        help="Compare explanations for closely matched digit model versions.",
    )
    digits_explanation_parser.add_argument(
        "--output-dir",
        default="artifacts/digits_explanation_stability",
    )
    digits_explanation_parser.add_argument("--sample-limit", type=int, default=120)
    digits_explanation_parser.add_argument("--threshold", type=float, default=0.80)
    digits_explanation_robustness_parser = subparsers.add_parser(
        "digits-explanation-split-robustness",
        help="Repeat digit explanation comparisons across data splits.",
    )
    digits_explanation_robustness_parser.add_argument(
        "--output-dir",
        default="artifacts/digits_explanation_split_robustness",
    )
    digits_explanation_robustness_parser.add_argument(
        "--sample-limit", type=int, default=80
    )
    digits_explanation_robustness_parser.add_argument(
        "--threshold", type=float, default=0.80
    )
    digits_threshold_parser = subparsers.add_parser(
        "digits-explanation-threshold-sensitivity",
        help="Compare hidden-drift workload across explanation thresholds.",
    )
    digits_threshold_parser.add_argument(
        "--output-dir",
        default="artifacts/digits_explanation_threshold_sensitivity",
    )
    digits_threshold_parser.add_argument("--sample-limit", type=int, default=80)
    serve_parser.add_argument(
        "--portfolio-report",
        default=None,
        help="Optional cross-dataset summary JSON.",
    )
    serve_parser.add_argument(
        "--robustness-report",
        default=None,
        help="Optional split-and-probe robustness summary JSON.",
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


def run_lime_heldout(config_path: str | Path) -> int:
    """Execute held-out LIME policy validation."""
    path = _project_path(config_path)
    config = load_config(path)
    print(f"Experiment: {config.experiment_id}", flush=True)
    print(f"Configuration: {path}", flush=True)
    run_lime_heldout_validation(
        config,
        config_path=path,
        project_root=PROJECT_ROOT,
    )
    return 0


def run_lime_multiplicity(
    breast_report_dir: str | Path,
    wine_report_dir: str | Path,
    output_dir: str | Path,
    *,
    alpha: float,
) -> int:
    """Compare unadjusted and family-wise LIME run policies."""
    run_multiplicity_analysis(
        breast_report_dir=_project_path(breast_report_dir),
        wine_report_dir=_project_path(wine_report_dir),
        output_dir=_project_path(output_dir),
        alpha=alpha,
    )
    return 0


def run_lime_split_robustness(config_path: str | Path) -> int:
    """Run the final cross-split and probe robustness experiment."""
    run_split_robustness(
        _project_path(config_path),
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
            robustness_path=(
                _project_path(DEFAULT_ROBUSTNESS_REPORT)
                if _project_path(DEFAULT_ROBUSTNESS_REPORT).is_file()
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
    robustness_report: str | Path | None = None,
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
        robustness_path=(
            _project_path(robustness_report)
            if robustness_report is not None
            else (
                _project_path(DEFAULT_ROBUSTNESS_REPORT)
                if _project_path(DEFAULT_ROBUSTNESS_REPORT).is_file()
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


def run_family_summary(args: argparse.Namespace) -> int:
    """Build the cross-model-family evidence summary."""
    run_model_family_summary(
        random_forest_breast_report=_project_path(
            args.random_forest_breast_report
        ),
        random_forest_wine_report=_project_path(args.random_forest_wine_report),
        gradient_boosting_breast_report=_project_path(
            args.gradient_boosting_breast_report
        ),
        gradient_boosting_wine_report=_project_path(
            args.gradient_boosting_wine_report
        ),
        output_dir=_project_path(args.output_dir),
    )
    return 0


def run_image_demo(output_dir: str | Path) -> int:
    """Build and describe the local image upload demonstration."""
    result = build_digits_image_demo(_project_path(output_dir))
    print("Handwritten-digit upload demo", flush=True)
    print("-----------------------------", flush=True)
    print(f"Samples: {result['sample_count']}", flush=True)
    print(f"Models: {result['model_count']}", flush=True)
    print(f"Model V1 accuracy: {result['v1_accuracy']:.1%}", flush=True)
    print(f"Model V2 accuracy: {result['v2_accuracy']:.1%}", flush=True)
    print(f"Best model accuracy: {result['best_accuracy']:.1%}", flush=True)
    print(f"Files saved to: {result['output_dir']}", flush=True)
    return 0


def run_prepare_image_data(input_path: str | Path, output_dir: str | Path, limit: int) -> int:
    """Prepare a user-provided labelled image NPZ for browser upload."""
    result = prepare_image_npz_bundle(
        _project_path(input_path), _project_path(output_dir), limit=limit
    )
    print("Image upload data prepared", flush=True)
    print("--------------------------", flush=True)
    print(f"Samples: {result['sample_count']}", flush=True)
    print(f"Image shape: {result['image_shape']}", flush=True)
    print(f"Arrays used: {result['array_source']}", flush=True)
    print(f"Class names: {result['class_name_source']}", flush=True)
    print(f"Features: {result['feature_count']}", flush=True)
    print(f"Files saved to: {result['output_dir']}", flush=True)
    return 0


def run_image_model_demo(input_path: str | Path, output_dir: str | Path, limit: int) -> int:
    """Train and describe a four-model image comparison demonstration."""
    result = build_image_model_demo(
        _project_path(input_path), _project_path(output_dir), limit=limit
    )
    print("Four-model image upload demo", flush=True)
    print("----------------------------", flush=True)
    print(f"Models: {result['model_count']}", flush=True)
    print(f"Training samples: {result['training_samples']}", flush=True)
    print(f"Test samples: {result['test_samples']}", flush=True)
    print(f"Split: {result['split_strategy']}", flush=True)
    print(f"Best accuracy: {result['best_accuracy']:.1%}", flush=True)
    print(f"Files saved to: {result['output_dir']}", flush=True)
    return 0


def run_regression_demo(output_dir: str | Path) -> int:
    """Build and describe the local multi-model regression demonstration."""
    result = build_regression_demo(_project_path(output_dir))
    print("Multi-model regression upload demo", flush=True)
    print("----------------------------------", flush=True)
    print(f"Models: {result['model_count']}", flush=True)
    print(f"Samples: {result['sample_count']}", flush=True)
    print(f"Best MAE: {result['best_mae']:.3f}", flush=True)
    print(f"Files saved to: {result['output_dir']}", flush=True)
    return 0


def run_digits_benchmark(output_dir: str | Path) -> int:
    """Run and describe the close-version handwritten-digit benchmark."""
    result = run_digits_close_version_benchmark(_project_path(output_dir))
    best = max(result["models"], key=lambda model: model["accuracy"])
    print("Close-version handwritten-digit benchmark", flush=True)
    print("-------------------------------------------", flush=True)
    print(f"Models: {len(result['models'])}", flush=True)
    print(f"Test samples: {result['split']['test_samples']}", flush=True)
    print(f"Best accuracy: {best['accuracy']:.1%} ({best['name']})", flush=True)
    print(f"Report saved to: {result['report_path']}", flush=True)
    return 0


def run_digits_robustness(output_dir: str | Path) -> int:
    """Run and describe the repeated-split digit benchmark."""
    result = run_digits_split_robustness(_project_path(output_dir))
    best = max(result["models"], key=lambda model: model["mean_accuracy"])
    print("Handwritten-digit split robustness", flush=True)
    print("----------------------------------", flush=True)
    print(f"Splits: {len(result['split_seeds'])}", flush=True)
    print(f"Models per split: {len(result['models'])}", flush=True)
    print(
        f"Best mean accuracy: {best['mean_accuracy']:.1%} ({best['name']})",
        flush=True,
    )
    print(f"Report saved to: {result['report_path']}", flush=True)
    return 0


def run_digits_training_stability(output_dir: str | Path) -> int:
    """Run and describe the training-randomness digit benchmark."""
    result = run_digits_training_robustness(_project_path(output_dir))
    best = max(result["models"], key=lambda model: model["mean_accuracy"])
    print("Handwritten-digit training robustness", flush=True)
    print("-------------------------------------", flush=True)
    print(f"Training seeds: {len(result['training_seeds'])}", flush=True)
    print(f"Models per seed: {len(result['models'])}", flush=True)
    print(
        f"Best mean accuracy: {best['mean_accuracy']:.1%} ({best['name']})",
        flush=True,
    )
    print(f"Report saved to: {result['report_path']}", flush=True)
    return 0


def run_digits_explanation_experiment(
    output_dir: str | Path,
    sample_limit: int,
    threshold: float,
) -> int:
    """Run and describe the close-version explanation comparison."""
    result = run_digits_explanation_stability(
        _project_path(output_dir),
        sample_limit=sample_limit,
        similarity_threshold=threshold,
    )
    print("Handwritten-digit explanation stability", flush=True)
    print("---------------------------------------", flush=True)
    print(f"Samples: {result['sample_limit']}", flush=True)
    print(f"Model pairs: {len(result['pairs'])}", flush=True)
    print(f"Inspection threshold: {result['similarity_threshold']:.2f}", flush=True)
    print(f"Report saved to: {result['report_path']}", flush=True)
    return 0


def run_digits_explanation_robustness_experiment(
    output_dir: str | Path,
    sample_limit: int,
    threshold: float,
) -> int:
    """Run and describe explanation comparisons across data splits."""
    result = run_digits_explanation_split_robustness(
        _project_path(output_dir),
        sample_limit=sample_limit,
        similarity_threshold=threshold,
    )
    hidden_drift = sum(
        pair["total_hidden_drift"] for pair in result["pair_summaries"]
    )
    print("Handwritten-digit explanation split robustness", flush=True)
    print("----------------------------------------------", flush=True)
    print(f"Splits: {len(result['split_seeds'])}", flush=True)
    print(f"Samples per split: {result['sample_limit_per_split']}", flush=True)
    print(f"Hidden-drift flags: {hidden_drift}", flush=True)
    print(f"Report saved to: {result['report_path']}", flush=True)
    return 0


def run_digits_explanation_threshold_experiment(
    output_dir: str | Path,
    sample_limit: int,
) -> int:
    """Run and describe explanation-threshold sensitivity."""
    result = run_digits_explanation_threshold_sensitivity(
        _project_path(output_dir),
        sample_limit=sample_limit,
    )
    print("Handwritten-digit explanation threshold sensitivity", flush=True)
    print("---------------------------------------------------", flush=True)
    for row in result["sensitivity"]:
        print(
            f"Threshold {row['threshold']:.2f}: "
            f"{row['hidden_drift_flags']} flags ({row['flag_rate']:.2%})",
            flush=True,
        )
    print(f"Report saved to: {result['report_path']}", flush=True)
    return 0


def run_mnist_calibration_experiment(
    input_path: str | Path,
    output_dir: str | Path,
    train_limit: int,
    sample_limit: int,
) -> int:
    """Run and describe MNIST benign-retraining calibration."""
    result = run_mnist_explanation_calibration(
        _project_path(input_path),
        _project_path(output_dir),
        train_limit=train_limit,
        sample_limit=sample_limit,
    )
    print("MNIST explanation calibration", flush=True)
    print("-----------------------------", flush=True)
    print(f"Benign pairs: {len(result['pairs'])}", flush=True)
    print(
        f"Same-prediction comparisons: {result['same_prediction_comparisons']}",
        flush=True,
    )
    print(f"5th-percentile threshold: {result['calibrated_threshold']:.3f}", flush=True)
    print(
        "Held-out benign flag rate: "
        f"{result['heldout_validation']['flag_rate']:.2%}",
        flush=True,
    )
    print(f"Report saved to: {result['report_path']}", flush=True)
    return 0


def run_cifar10_calibration_experiment(
    input_path: str | Path,
    output_dir: str | Path,
    train_limit: int,
    sample_limit: int,
) -> int:
    """Run and describe CIFAR-10 benign-retraining calibration."""
    result = run_cifar10_explanation_calibration(
        _project_path(input_path),
        _project_path(output_dir),
        train_limit=train_limit,
        sample_limit=sample_limit,
    )
    print("CIFAR-10 explanation calibration", flush=True)
    print("--------------------------------", flush=True)
    print(f"Benign pairs: {len(result['pairs'])}", flush=True)
    print(
        f"Same-prediction comparisons: {result['same_prediction_comparisons']}",
        flush=True,
    )
    print(f"5th-percentile threshold: {result['calibrated_threshold']:.3f}", flush=True)
    print(
        "Held-out benign flag rate: "
        f"{result['heldout_validation']['flag_rate']:.2%}",
        flush=True,
    )
    print(f"Report saved to: {result['report_path']}", flush=True)
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
        if args.command == "lime-heldout":
            return run_lime_heldout(args.config)
        if args.command == "lime-multiplicity":
            return run_lime_multiplicity(
                args.breast_report_dir,
                args.wine_report_dir,
                args.output_dir,
                alpha=args.alpha,
            )
        if args.command == "lime-split-robustness":
            return run_lime_split_robustness(args.config)
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
                robustness_report=args.robustness_report,
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
        if args.command == "model-family-summary":
            return run_family_summary(args)
        if args.command == "shap-split-robustness":
            run_shap_split_robustness(
                _project_path(args.config),
                project_root=PROJECT_ROOT,
            )
            return 0
        if args.command == "shap-multiplicity":
            run_shap_multiplicity_analysis(
                robustness_dir=_project_path(args.robustness_dir),
                output_dir=_project_path(args.output_dir),
                alpha=args.alpha,
            )
            return 0
        if args.command == "image-demo":
            return run_image_demo(args.output_dir)
        if args.command == "prepare-image-data":
            return run_prepare_image_data(args.input, args.output_dir, args.limit)
        if args.command == "image-model-demo":
            return run_image_model_demo(args.input, args.output_dir, args.limit)
        if args.command == "mnist-explanation-calibration":
            return run_mnist_calibration_experiment(
                args.input,
                args.output_dir,
                args.train_limit,
                args.sample_limit,
            )
        if args.command == "cifar10-explanation-calibration":
            return run_cifar10_calibration_experiment(
                args.input,
                args.output_dir,
                args.train_limit,
                args.sample_limit,
            )
        if args.command == "regression-demo":
            return run_regression_demo(args.output_dir)
        if args.command == "digits-close-benchmark":
            return run_digits_benchmark(args.output_dir)
        if args.command == "digits-split-robustness":
            return run_digits_robustness(args.output_dir)
        if args.command == "digits-training-robustness":
            return run_digits_training_stability(args.output_dir)
        if args.command == "digits-explanation-stability":
            return run_digits_explanation_experiment(
                args.output_dir,
                args.sample_limit,
                args.threshold,
            )
        if args.command == "digits-explanation-split-robustness":
            return run_digits_explanation_robustness_experiment(
                args.output_dir,
                args.sample_limit,
                args.threshold,
            )
        if args.command == "digits-explanation-threshold-sensitivity":
            return run_digits_explanation_threshold_experiment(
                args.output_dir,
                args.sample_limit,
            )
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    parser.error(f"Unsupported command: {args.command}")
    return 2
