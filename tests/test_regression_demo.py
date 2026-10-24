import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from xai_regression.regression_demo import build_regression_demo


class RegressionDemoTests(unittest.TestCase):
    def test_demo_bundle_contains_four_models_and_upload_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            result = build_regression_demo(output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            samples = pd.read_csv(output / "samples.csv")
            model_paths = sorted(output.glob("*.joblib"))

            self.assertEqual(result["model_count"], 4)
            self.assertEqual(len(model_paths), 4)
            self.assertEqual(manifest["task"], "regression")
            self.assertEqual(manifest["label_column"], "target")
            self.assertEqual(len(samples), result["sample_count"])
            self.assertIn("sample_id", samples)
            self.assertTrue(all(hasattr(joblib.load(path), "predict") for path in model_paths))


if __name__ == "__main__":
    unittest.main()
