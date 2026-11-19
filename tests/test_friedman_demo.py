import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from xai_regression.friedman_demo import build_friedman_regression_demo


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


if __name__ == "__main__":
    unittest.main()
