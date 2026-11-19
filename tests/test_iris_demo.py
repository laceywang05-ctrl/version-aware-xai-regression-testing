import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from xai_regression.iris_demo import build_iris_classification_demo
from xai_regression.upload_comparison import (
    summarize_upload_models,
    validate_upload_models,
)


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

            overview = summarize_upload_models(
                models=[(path.name, path.read_bytes()) for path in model_paths],
                dataset=(samples_path.name, samples_path.read_bytes()),
                manifest=manifest_path.read_bytes(),
            )
            self.assertEqual(len(overview["models"]), 5)
            self.assertEqual(len(overview["pairwise_agreements"]), 10)
            self.assertEqual(len(overview["class_comparison"]), 3)
            self.assertEqual(overview["sample_comparison_total"], 45)
            self.assertEqual(
                overview["models"][overview["best_model_index"]]["filename"],
                "model_3_svm.joblib",
            )
            self.assertEqual(overview["explanation_consensus"]["method"], "LIME")


if __name__ == "__main__":
    unittest.main()
