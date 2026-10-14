import unittest
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


if __name__ == "__main__":
    unittest.main()
