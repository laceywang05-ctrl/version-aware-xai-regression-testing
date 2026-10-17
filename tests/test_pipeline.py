import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from xai_regression.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_pipeline_runs_three_stages_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "unified"

            def write_unified_report(**kwargs):
                reports = Path(kwargs["output_dir"]) / "reports"
                reports.mkdir(parents=True, exist_ok=True)
                (reports / "summary.json").write_text(
                    json.dumps({"methods": [], "metrics": []}),
                    encoding="utf-8",
                )
                return {"overall_status": "REVIEW"}

            with (
                patch(
                    "xai_regression.pipeline.run_validation",
                    return_value={"status": "WITHIN_BASELINE"},
                ) as shap,
                patch(
                    "xai_regression.pipeline.run_lime_cross_version",
                    return_value={"status": "REVIEW"},
                ) as lime,
                patch(
                    "xai_regression.pipeline.run_unified_report",
                    side_effect=write_unified_report,
                ) as report,
                redirect_stdout(StringIO()),
            ):
                result = run_pipeline(
                    shap_config_path=PROJECT_ROOT / "config/default.yaml",
                    lime_config_path=PROJECT_ROOT / "config/lime_cross_version.yaml",
                    output_dir=output,
                    project_root=root,
                )

            shap.assert_called_once()
            lime.assert_called_once()
            report.assert_called_once()
            self.assertEqual(result["report_path"], output / "reports/summary.json")
            manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["overall_status"], "REVIEW")
            self.assertEqual(manifest["shap_status"], "WITHIN_BASELINE")
            self.assertEqual(manifest["lime_status"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
