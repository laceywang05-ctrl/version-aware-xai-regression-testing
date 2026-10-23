import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xai_regression.server import (
    parse_multipart_form,
    serve_dashboard,
    validate_benchmark_files,
    validate_portfolio_file,
    validate_report_file,
    validate_robustness_file,
)


class DashboardServerTests(unittest.TestCase):
    def test_multipart_upload_fields_are_parsed_in_memory(self):
        boundary = "xai-boundary"
        payload = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="manifest"; filename="manifest.json"\r\n'
            "Content-Type: application/json\r\n\r\n"
            '{"task":"classification"}\r\n'
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        fields = parse_multipart_form(
            f"multipart/form-data; boundary={boundary}", payload
        )

        self.assertEqual(fields["manifest"][0], "manifest.json")
        self.assertEqual(
            json.loads(fields["manifest"][1]), {"task": "classification"}
        )

    def test_unified_report_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps({"methods": [], "metrics": []}),
                encoding="utf-8",
            )
            self.assertEqual(validate_report_file(path), path.resolve())

    def test_non_unified_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"status": "REVIEW"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "methods and metrics"):
                validate_report_file(path)

    def test_benchmark_report_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_results": [],
                        "paired_control_comparisons": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validate_benchmark_files([path]), [path.resolve()])

    def test_non_benchmark_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_text(json.dumps({"methods": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scenario_results"):
                validate_benchmark_files([path])

    def test_cross_dataset_report_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portfolio.json"
            path.write_text(
                json.dumps({"coverage": {}, "entries": []}), encoding="utf-8"
            )
            self.assertEqual(validate_portfolio_file(path), path.resolve())

    def test_non_cross_dataset_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portfolio.json"
            path.write_text(json.dumps({"methods": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "entries and coverage"):
                validate_portfolio_file(path)

    def test_split_robustness_report_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robustness.json"
            path.write_text(
                json.dumps({"dataset_summaries": [], "split_results": []}),
                encoding="utf-8",
            )
            self.assertEqual(validate_robustness_file(path), path.resolve())

    def test_non_split_robustness_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robustness.json"
            path.write_text(json.dumps({"entries": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset_summaries"):
                validate_robustness_file(path)

    def test_socket_error_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "summary.json"
            report.write_text(
                json.dumps({"methods": [], "metrics": []}),
                encoding="utf-8",
            )
            dashboard = root / "dashboard"
            dashboard.mkdir()
            (dashboard / "index.html").write_text("dashboard", encoding="utf-8")
            with (
                patch(
                    "xai_regression.server.ThreadingHTTPServer",
                    side_effect=OSError("address already in use"),
                ),
                self.assertRaisesRegex(RuntimeError, "Could not start dashboard"),
            ):
                serve_dashboard(report_path=report, dashboard_dir=dashboard)


if __name__ == "__main__":
    unittest.main()
