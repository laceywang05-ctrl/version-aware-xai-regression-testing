import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from xai_regression.iris_demo import build_iris_classification_demo
from xai_regression.upload_comparison import validate_upload_models


class IrisDemoTests(unittest.TestCase):
    def test_demo_bundle_contains_five_compatible_model_families(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            result = build_iris_classification_demo(output)
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

            self.assertEqual(result["model_count"], 5)
            self.assertEqual(result["training_samples"], 105)
            self.assertEqual(result["sample_count"], 45)
            self.assertEqual(result["feature_count"], 4)
            self.assertEqual(result["class_count"], 3)
            self.assertEqual(len(samples), 45)
            self.assertEqual(len(model_paths), 5)
            self.assertTrue(validation["ready"])
            model_types = {type(joblib.load(path)).__name__ for path in model_paths}
            self.assertEqual(
                model_types,
                {"Pipeline", "GaussianNB", "ExtraTreesClassifier"},
            )


if __name__ == "__main__":
    unittest.main()
