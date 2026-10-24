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
            "data/upload-manifest-example.json",
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
            "robustness-view",
            "robustness-status",
            "robustness-dataset-grid",
            "robustness-results",
            "upload-view",
            "classification-upload-form",
            "model-v1-upload",
            "model-v2-upload",
            "additional-models-upload",
            "baseline-model-selector",
            "candidate-model-selector",
            "model-collection-message",
            "dataset-upload",
            "manifest-upload",
            "manifest-builder",
            "manifest-label-column",
            "manifest-id-column",
            "manifest-image-column",
            "manifest-image-height",
            "manifest-image-width",
            "manifest-trust-models",
            "images-upload",
            "upload-results",
            "upload-validation",
            "upload-validation-grid",
            "model-overview-grid",
            "model-agreement-results",
            "class-diagnostics-results",
            "class-v1-heading",
            "class-v2-heading",
            "confusion-diagnostics-results",
            "confusion-v1-heading",
            "confusion-v2-heading",
            "confusion-empty-message",
            "download-comparison-json",
            "download-samples-csv",
            "confidence-risk-summary",
            "confidence-threshold-input",
            "explanation-threshold-input",
            "upload-sample-results",
            "explanation-dialog",
            "v1-heatmap",
            "v2-heatmap",
        ):
            self.assertIn(f'id="{element_id}"', html)

    def test_dashboard_prefers_runtime_report_then_falls_back(self):
        script = (DASHBOARD / "app.js").read_text(encoding="utf-8")
        self.assertIn('fetch("data/report.json"', script)
        self.assertIn('fetch("data/example-report.json"', script)
        self.assertIn('fetch("data/benchmarks.json"', script)
        self.assertIn('fetch("data/portfolio.json"', script)
        self.assertIn('fetch("data/robustness.json"', script)

    def test_dashboard_supports_both_report_schemas(self):
        script = (DASHBOARD / "app.js").read_text(encoding="utf-8")
        self.assertIn("function validateReport", script)
        self.assertIn("function validateBenchmark", script)
        self.assertIn("function renderBenchmark", script)
        self.assertIn("function validatePortfolio", script)
        self.assertIn("function renderPortfolio", script)
        self.assertIn("function validateRobustness", script)
        self.assertIn("function renderRobustness", script)
        self.assertIn("function renderUploadComparison", script)
        self.assertIn('fetch("/api/classification/compare"', script)
        self.assertIn('fetch("/api/classification/overview"', script)
        self.assertIn('fetch("/api/classification/validate"', script)
        self.assertIn("function renderModelOverview", script)
        self.assertIn('calibration.className = "calibration-summary"', script)
        self.assertIn("expected_calibration_error", script)
        self.assertIn("function renderUploadValidation", script)
        self.assertIn("function downloadComparisonJson", script)
        self.assertIn("function downloadSamplesCsv", script)
        self.assertIn("application/json", script)
        self.assertIn("text/csv;charset=utf-8", script)
        self.assertIn("sample.image_data_url", script)
        self.assertIn("function renderPixelHeatmap", script)
        self.assertIn("function openExplanation", script)
        self.assertIn("function uploadedModels", script)
        self.assertIn("function updateModelSelectors", script)
        self.assertIn("function parseCsvHeader", script)
        self.assertIn("function updateManifestBuilder", script)
        self.assertIn("function selectedManifestFile", script)
        self.assertIn('"generated-manifest.json"', script)
        self.assertIn("slice(0, 3)", script)
        self.assertIn('selected === "v2_high_confidence_wrong"', script)
        self.assertIn('selected === "high_confidence_regression"', script)
        self.assertIn('selected === "explanation_drift"', script)
        self.assertIn('selected === "hidden_explanation_drift"', script)
        self.assertIn("function formatSimilarity", script)
        self.assertIn("function triageThresholds", script)
        self.assertIn("function sampleTriage", script)
        self.assertIn("function renderTriageSummary", script)
        self.assertIn("function comparisonForExport", script)
        self.assertIn("triage_settings: triageThresholds()", script)
        self.assertIn('"v2_high_confidence_wrong", "high_confidence_regression"', script)

    def test_hidden_dashboard_views_are_not_displayed(self):
        styles = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".dashboard-view[hidden] { display: none; }", styles)


if __name__ == "__main__":
    unittest.main()
