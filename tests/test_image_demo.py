import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from xai_regression.image_demo import build_digits_image_demo
from xai_regression.upload_comparison import compare_upload_payloads


class ImageDemoTests(unittest.TestCase):
    def test_demo_bundle_runs_through_upload_comparison(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            result = build_digits_image_demo(output)

            expected = {
                "model_v1.joblib",
                "model_v2.joblib",
                "model_v3_extra_trees.joblib",
                "model_v4_knn.joblib",
                "samples.csv",
                "manifest.json",
                "images.zip",
                "README.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertGreater(result["v2_accuracy"], result["v1_accuracy"])
            self.assertEqual(result["model_count"], 4)
            with ZipFile(output / "images.zip") as archive:
                self.assertEqual(len(archive.namelist()), result["sample_count"])

            comparison = compare_upload_payloads(
                model_v1=("model_v1.joblib", (output / "model_v1.joblib").read_bytes()),
                model_v2=("model_v2.joblib", (output / "model_v2.joblib").read_bytes()),
                dataset=("samples.csv", (output / "samples.csv").read_bytes()),
                manifest=(output / "manifest.json").read_bytes(),
                images=("images.zip", (output / "images.zip").read_bytes()),
            )

            self.assertEqual(
                comparison["summary"]["total_samples"], result["sample_count"]
            )
            self.assertGreater(
                comparison["summary"]["v1_wrong_v2_correct"], 0
            )
            self.assertGreater(
                comparison["summary"]["v1_correct_v2_wrong"], 0
            )
            self.assertTrue(
                comparison["samples"][0]["image_data_url"].startswith(
                    "data:image/png;base64,"
                )
            )
            self.assertEqual(len(comparison["samples"][0]["v1_explanation"]), 8)
            self.assertEqual(len(comparison["samples"][0]["v1_explanation"][0]), 8)
            self.assertEqual(
                comparison["explanations"]["v1"]["method"], "tree_shap"
            )
            self.assertEqual(
                comparison["explanations"]["v2"]["method"], "tree_shap"
            )
            self.assertEqual(
                comparison["summary"]["explanation_compared_count"],
                result["sample_count"],
            )
            self.assertIsInstance(
                comparison["samples"][0]["explanation_similarity"], float
            )
            summary = json.loads((output / "README.json").read_text())
            self.assertIn("DecisionTreeClassifier", summary["model_v1"])
            self.assertEqual(len(summary["models"]), 4)
            self.assertTrue(
                all("test_macro_f1" in model for model in summary["models"])
            )


if __name__ == "__main__":
    unittest.main()
