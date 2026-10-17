import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xai_regression.server import serve_dashboard, validate_report_file


class DashboardServerTests(unittest.TestCase):
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
