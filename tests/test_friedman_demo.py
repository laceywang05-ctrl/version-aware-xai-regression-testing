import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from xai_regression.friedman_demo import build_friedman_regression_demo
from xai_regression.regression_comparison import summarize_regression_upload_models


class FriedmanDemoTests(unittest.TestCase):
    def test_demo_bundle_contains_five_model_families(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            result = build_friedman_regression_demo(output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            samples = pd.read_csv(output / "samples.csv")
            model_paths = sorted(output.glob("*.joblib"))

            self.assertEqual(result["model_count"], 5)
            self.assertEqual(result["training_samples"], 700)
            self.assertEqual(result["sample_count"], 300)
            self.assertEqual(result["feature_count"], 10)
            self.assertEqual(len(model_paths), 5)
            self.assertEqual(manifest["task"], "regression")
            self.assertEqual(manifest["feature_columns"], [f"x{i}" for i in range(1, 11)])
            self.assertEqual(len(samples), 300)
            features = samples[manifest["feature_columns"]]
            self.assertTrue(
                all(len(joblib.load(path).predict(features)) == 300 for path in model_paths)
            )

    def test_all_five_models_enter_dashboard_overview(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            build_friedman_regression_demo(output)
            model_paths = sorted(output.glob("*.joblib"))

            overview = summarize_regression_upload_models(
                models=[(path.name, path.read_bytes()) for path in model_paths],
                dataset=("samples.csv", (output / "samples.csv").read_bytes()),
                manifest=(output / "manifest.json").read_bytes(),
            )

            self.assertEqual(len(overview["models"]), 5)
            self.assertEqual(len(overview["pairwise_error_changes"]), 10)
            self.assertEqual(overview["dataset"]["rows"], 300)
            self.assertEqual(overview["dataset"]["feature_count"], 10)
            self.assertEqual(overview["models"][overview["best_model_index"]]["rank"], 1)
            self.assertEqual(
                overview["models"][overview["best_model_index"]]["filename"],
                "model_3_svr.joblib",
            )


if __name__ == "__main__":
    unittest.main()
