import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from xai_regression import cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_run_uses_current_python_and_validated_config(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0)

        with (
            patch("xai_regression.cli.subprocess.run", return_value=completed) as run,
            redirect_stdout(StringIO()),
        ):
            exit_code = cli.run_experiment(PROJECT_ROOT / "config/default.yaml")

        self.assertEqual(exit_code, 0)
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(command[1].endswith("07_real_dataset_validation.py"))
        self.assertEqual(command[2], "--config")
        self.assertTrue(command[3].endswith("config/default.yaml"))
        self.assertEqual(run.call_args.kwargs["cwd"], cli.PROJECT_ROOT)

    def test_parser_requires_a_command(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            cli.main([])


if __name__ == "__main__":
    unittest.main()
