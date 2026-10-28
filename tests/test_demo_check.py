import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from xai_regression.demo_check import check_demo_bundle


class DemoCheckTests(unittest.TestCase):
    def test_classification_bundle_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                {"sample_id": ["a", "b", "c"], "x": [0.0, 1.0, 2.0], "target": [0, 1, 1]}
            )
            frame.to_csv(root / "samples.csv", index=False)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "task": "classification",
                        "label_column": "target",
                        "id_column": "sample_id",
                        "feature_columns": ["x"],
                        "trusted_model_files": True,
                    }
                ),
                encoding="utf-8",
            )
            for index in range(2):
                model = DummyClassifier(strategy="most_frequent").fit(frame[["x"]], frame["target"])
                joblib.dump(model, root / f"model_{index}.joblib")

            result = check_demo_bundle(root)

        self.assertTrue(result["ready"])
        self.assertEqual(result["task"], "classification")
        self.assertEqual(result["model_count"], 2)
        self.assertEqual(result["dataset"]["rows"], 3)

    def test_missing_dataset_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "samples.csv"):
                check_demo_bundle(root)
