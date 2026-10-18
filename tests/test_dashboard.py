import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT_ROOT / "dashboard"


class DashboardTests(unittest.TestCase):
    def test_static_dashboard_files_exist(self):
        for relative_path in (
            "index.html",
            "styles.css",
            "app.js",
            "data/example-report.json",
        ):
            self.assertTrue((DASHBOARD / relative_path).is_file())

    def test_example_report_matches_unified_schema(self):
        report = json.loads(
            (DASHBOARD / "data/example-report.json").read_text(encoding="utf-8")
        )

        self.assertEqual(report["overall_status"], "REVIEW")
        self.assertEqual({row["method"] for row in report["methods"]}, {"SHAP", "LIME"})
        self.assertEqual(len(report["metrics"]), 8)
        self.assertEqual(
            {row["metric"] for row in report["metrics"]},
            {
                "cosine_similarity",
                "rank_correlation",
                "top_k_overlap",
                "direction_flip_rate",
            },
        )

    def test_html_contains_required_application_regions(self):
        html = (DASHBOARD / "index.html").read_text(encoding="utf-8")

        for element_id in (
            "overall-status",
            "method-cards",
            "method-tabs",
            "metric-grid",
            "report-upload",
            "benchmark-view",
            "benchmark-selector",
            "scenario-grid",
            "paired-results",
            "portfolio-view",
            "portfolio-grid",
            "portfolio-results",
        ):
            self.assertIn(f'id="{element_id}"', html)

    def test_dashboard_prefers_runtime_report_then_falls_back(self):
        script = (DASHBOARD / "app.js").read_text(encoding="utf-8")
        self.assertIn('fetch("data/report.json"', script)
        self.assertIn('fetch("data/example-report.json"', script)
        self.assertIn('fetch("data/benchmarks.json"', script)
        self.assertIn('fetch("data/portfolio.json"', script)

    def test_dashboard_supports_both_report_schemas(self):
        script = (DASHBOARD / "app.js").read_text(encoding="utf-8")
        self.assertIn("function validateReport", script)
        self.assertIn("function validateBenchmark", script)
        self.assertIn("function renderBenchmark", script)
        self.assertIn("function validatePortfolio", script)
        self.assertIn("function renderPortfolio", script)


if __name__ == "__main__":
    unittest.main()
