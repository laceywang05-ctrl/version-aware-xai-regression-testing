import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from xai_regression import cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_run_uses_current_python_and_validated_config(self):
        with (
            patch("xai_regression.cli.run_validation", return_value={}) as run,
            redirect_stdout(StringIO()),
        ):
            exit_code = cli.run_experiment(PROJECT_ROOT / "config/default.yaml")

        self.assertEqual(exit_code, 0)
        config = run.call_args.args[0]
        self.assertEqual(config.experiment_id, "breast-cancer-shap-v1")
        self.assertTrue(
            str(run.call_args.kwargs["config_path"]).endswith(
                "config/default.yaml"
            )
        )
        self.assertEqual(run.call_args.kwargs["project_root"], cli.PROJECT_ROOT)

    def test_parser_requires_a_command(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            cli.main([])

    def test_lime_command_uses_validated_config(self):
        with (
            patch(
                "xai_regression.cli.run_lime_stability",
                return_value={},
            ) as run,
            redirect_stdout(StringIO()),
        ):
            exit_code = cli.run_lime_experiment(
                PROJECT_ROOT / "config/lime_stability.yaml"
            )

        self.assertEqual(exit_code, 0)
        config = run.call_args.args[0]
        self.assertEqual(
            config.experiment_id,
            "breast-cancer-lime-stability-v1",
        )
        self.assertEqual(run.call_args.kwargs["project_root"], cli.PROJECT_ROOT)

    def test_lime_compare_command_uses_validated_config(self):
        with (
            patch(
                "xai_regression.cli.run_lime_cross_version",
                return_value={},
            ) as run,
            redirect_stdout(StringIO()),
        ):
            exit_code = cli.run_lime_comparison(
                PROJECT_ROOT / "config/lime_cross_version.yaml"
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run.call_args.args[0].experiment_id,
            "breast-cancer-lime-cross-version-v1",
        )

    def test_lime_heldout_command_uses_validated_config(self):
        with (
            patch(
                "xai_regression.cli.run_lime_heldout_validation",
                return_value={},
            ) as run,
            redirect_stdout(StringIO()),
        ):
            exit_code = cli.run_lime_heldout(
                PROJECT_ROOT / "config/lime_heldout.yaml"
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run.call_args.args[0].experiment_id,
            "breast-cancer-lime-heldout-v1",
        )

    def test_lime_multiplicity_resolves_project_paths(self):
        with patch(
            "xai_regression.cli.run_multiplicity_analysis",
            return_value={},
        ) as run:
            exit_code = cli.run_lime_multiplicity(
                "breast",
                "wine",
                "output",
                alpha=0.05,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run.call_args.kwargs["breast_report_dir"],
            cli.PROJECT_ROOT / "breast",
        )
        self.assertEqual(
            run.call_args.kwargs["wine_report_dir"],
            cli.PROJECT_ROOT / "wine",
        )
        self.assertEqual(
            run.call_args.kwargs["output_dir"],
            cli.PROJECT_ROOT / "output",
        )
        self.assertEqual(run.call_args.kwargs["alpha"], 0.05)

    def test_lime_split_robustness_resolves_config_path(self):
        with patch(
            "xai_regression.cli.run_split_robustness",
            return_value={},
        ) as run:
            exit_code = cli.run_lime_split_robustness("split.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_args.args[0], cli.PROJECT_ROOT / "split.yaml")
        self.assertEqual(run.call_args.kwargs["project_root"], cli.PROJECT_ROOT)

    def test_report_command_resolves_project_paths(self):
        with patch(
            "xai_regression.cli.run_unified_report",
            return_value={},
        ) as run:
            exit_code = cli.run_report("shap", "lime", "output")

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_args.kwargs["shap_report_dir"], cli.PROJECT_ROOT / "shap")
        self.assertEqual(run.call_args.kwargs["lime_report_dir"], cli.PROJECT_ROOT / "lime")
        self.assertEqual(run.call_args.kwargs["output_dir"], cli.PROJECT_ROOT / "output")

    def test_workflow_runs_pipeline_without_serving_by_default(self):
        with (
            patch(
                "xai_regression.cli.run_pipeline",
                return_value={"report_path": Path("summary.json")},
            ) as pipeline,
            patch("xai_regression.cli.serve_dashboard") as serve,
        ):
            exit_code = cli.run_workflow(
                "shap.yaml",
                "lime.yaml",
                "output",
                serve=False,
                host="127.0.0.1",
                port=8000,
            )

        self.assertEqual(exit_code, 0)
        pipeline.assert_called_once()
        serve.assert_not_called()

    def test_workflow_can_serve_generated_report(self):
        report_path = PROJECT_ROOT / "summary.json"
        with (
            patch(
                "xai_regression.cli.run_pipeline",
                return_value={"report_path": report_path},
            ),
            patch("xai_regression.cli.serve_dashboard") as serve,
        ):
            exit_code = cli.run_workflow(
                "shap.yaml",
                "lime.yaml",
                "output",
                serve=True,
                host="127.0.0.1",
                port=8123,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(serve.call_args.kwargs["report_path"], report_path)
        self.assertEqual(serve.call_args.kwargs["port"], 8123)

    def test_dashboard_accepts_explicit_benchmark_reports(self):
        with patch("xai_regression.cli.serve_dashboard") as serve:
            exit_code = cli.run_dashboard(
                "summary.json",
                host="127.0.0.1",
                port=8000,
                benchmark_reports=["benchmark.json"],
                portfolio_report="portfolio.json",
                robustness_report="robustness.json",
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            serve.call_args.kwargs["benchmark_paths"],
            [cli.PROJECT_ROOT / "benchmark.json"],
        )
        self.assertEqual(
            serve.call_args.kwargs["portfolio_path"],
            cli.PROJECT_ROOT / "portfolio.json",
        )
        self.assertEqual(
            serve.call_args.kwargs["robustness_path"],
            cli.PROJECT_ROOT / "robustness.json",
        )

    def test_portfolio_command_resolves_all_paths(self):
        with patch(
            "xai_regression.cli.run_cross_dataset_summary", return_value={}
        ) as run:
            exit_code = cli.run_portfolio(
                "breast-shap.json",
                "wine-shap.json",
                "breast-lime.json",
                "wine-lime.json",
                "output",
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run.call_args.kwargs["wine_lime_report"],
            cli.PROJECT_ROOT / "wine-lime.json",
        )
        self.assertEqual(
            run.call_args.kwargs["output_dir"], cli.PROJECT_ROOT / "output"
        )

    def test_benchmark_command_resolves_both_configs(self):
        with patch(
            "xai_regression.cli.run_version_benchmark",
            return_value={},
        ) as run:
            exit_code = cli.run_benchmark(
                "config/default.yaml",
                "config/version_benchmark.yaml",
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run.call_args.kwargs["scenario_config_path"],
            cli.PROJECT_ROOT / "config/version_benchmark.yaml",
        )

    def test_model_family_summary_resolves_all_paths(self):
        args = Namespace(
            random_forest_breast_report="rf-breast.json",
            random_forest_wine_report="rf-wine.json",
            gradient_boosting_breast_report="gb-breast.json",
            gradient_boosting_wine_report="gb-wine.json",
            output_dir="output",
        )
        with patch(
            "xai_regression.cli.run_model_family_summary", return_value={}
        ) as run:
            exit_code = cli.run_family_summary(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run.call_args.kwargs["gradient_boosting_wine_report"],
            cli.PROJECT_ROOT / "gb-wine.json",
        )
        self.assertEqual(
            run.call_args.kwargs["output_dir"], cli.PROJECT_ROOT / "output"
        )

    def test_image_demo_resolves_output_path(self):
        with (
            patch(
                "xai_regression.cli.build_digits_image_demo",
                return_value={
                    "sample_count": 12,
                    "v1_accuracy": 0.5,
                    "v2_accuracy": 0.9,
                    "output_dir": cli.PROJECT_ROOT / "demo",
                },
            ) as build,
            redirect_stdout(StringIO()),
        ):
            exit_code = cli.run_image_demo("demo")

        self.assertEqual(exit_code, 0)
        build.assert_called_once_with(cli.PROJECT_ROOT / "demo")

    def test_regression_demo_resolves_output_path(self):
        with (
            patch(
                "xai_regression.cli.build_regression_demo",
                return_value={
                    "model_count": 4,
                    "sample_count": 20,
                    "best_mae": 12.5,
                    "output_dir": cli.PROJECT_ROOT / "regression-demo",
                },
            ) as build,
            redirect_stdout(StringIO()),
        ):
            exit_code = cli.run_regression_demo("regression-demo")
        self.assertEqual(exit_code, 0)
        build.assert_called_once_with(cli.PROJECT_ROOT / "regression-demo")


if __name__ == "__main__":
    unittest.main()
