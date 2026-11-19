import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from xai_regression.upload_comparison import validate_upload_models
from xai_regression.wine_demo import build_wine_classification_demo


class WineDemoTests(unittest.TestCase):
    def test_demo_bundle_contains_four_compatible_multiclass_models(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            result = build_wine_classification_demo(output)
            manifest_path = output / "manifest.json"
            samples_path = output / "samples.csv"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            samples = pd.read_csv(samples_path)
            model_paths = sorted(output.glob("*.joblib"))
            validation = validate_upload_models(
                models=[(path.name, path.read_bytes()) for path in model_paths],
                dataset=(samples_path.name, samples_path.read_bytes()),
                manifest=manifest_path.read_bytes(),
            )

            self.assertEqual(result["model_count"], 4)
            self.assertEqual(result["training_samples"], 124)
            self.assertEqual(result["sample_count"], 54)
            self.assertEqual(result["feature_count"], 13)
            self.assertEqual(result["class_count"], 3)
            self.assertEqual(manifest["task"], "classification")
            self.assertEqual(len(manifest["class_names"]), 3)
            self.assertEqual(len(samples), 54)
            self.assertEqual(len(model_paths), 4)
            self.assertTrue(validation["ready"])
            self.assertTrue(all(hasattr(joblib.load(path), "predict") for path in model_paths))


if __name__ == "__main__":
    unittest.main()
