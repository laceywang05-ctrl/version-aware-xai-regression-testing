import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from xai_regression.demo_check import check_demo_bundle, check_demo_matrix


class DemoCheckTests(unittest.TestCase):
    @staticmethod
    def _write_classification_bundle(root: Path) -> None:
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
            model = DummyClassifier(strategy="most_frequent").fit(
                frame[["x"]], frame["target"]
            )
            joblib.dump(model, root / f"model_{index}.joblib")

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

    def test_matrix_keeps_checking_after_one_bundle_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "ready"
            failed = root / "failed"
            ready.mkdir()
            failed.mkdir()
            self._write_classification_bundle(ready)
            (failed / "manifest.json").write_text("{}", encoding="utf-8")

            result = check_demo_matrix([ready, failed])

        self.assertFalse(result["ready"])
        self.assertEqual(result["ready_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertIsNone(result["bundles"][0]["error"])
        self.assertIn("samples.csv", result["bundles"][1]["error"])

    def test_explanation_preflight_reports_actual_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                {
                    "sample_id": ["a", "b", "c"],
                    "pixel": [0.0, 1.0, 2.0],
                    "target": [0, 1, 1],
                }
            )
            frame.to_csv(root / "samples.csv", index=False)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "task": "classification",
                        "label_column": "target",
                        "id_column": "sample_id",
                        "feature_columns": ["pixel"],
                        "image_shape": [1, 1],
                        "trusted_model_files": True,
                    }
                ),
                encoding="utf-8",
            )
            for index in range(2):
                model = DummyClassifier(strategy="prior").fit(
                    frame[["pixel"]], frame["target"]
                )
                joblib.dump(model, root / f"model_{index}.joblib")

            result = check_demo_bundle(root, check_explanations=True)

        self.assertEqual(len(result["explanations"]), 2)
        self.assertTrue(all(row["available"] for row in result["explanations"]))
        self.assertEqual(
            {row["method"] for row in result["explanations"]},
            {"pixel_occlusion"},
        )

    def test_tabular_explanation_preflight_reports_lime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                {
                    "sample_id": [f"sample-{index}" for index in range(8)],
                    "x1": [float(index) for index in range(8)],
                    "x2": [float(index % 3) for index in range(8)],
                    "target": [index % 2 for index in range(8)],
                }
            )
            frame.to_csv(root / "samples.csv", index=False)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "task": "classification",
                        "label_column": "target",
                        "id_column": "sample_id",
                        "feature_columns": ["x1", "x2"],
                        "trusted_model_files": True,
                    }
                ),
                encoding="utf-8",
            )
            for index in range(2):
                model = DummyClassifier(strategy="prior").fit(
                    frame[["x1", "x2"]], frame["target"]
                )
                joblib.dump(model, root / f"model_{index}.joblib")

            result = check_demo_bundle(root, check_explanations=True)

        self.assertTrue(result["ready"])
        self.assertEqual(len(result["explanations"]), 2)
        self.assertTrue(
            all("lime" in row["method"] for row in result["explanations"])
        )

    def test_regression_explanation_preflight_reports_permutation_importance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                {
                    "sample_id": [f"sample-{index}" for index in range(8)],
                    "x1": [float(index) for index in range(8)],
                    "x2": [float(index % 3) for index in range(8)],
                    "target": [float(index * 2) for index in range(8)],
                }
            )
            frame.to_csv(root / "samples.csv", index=False)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "task": "regression",
                        "label_column": "target",
                        "id_column": "sample_id",
                        "feature_columns": ["x1", "x2"],
                        "trusted_model_files": True,
                    }
                ),
                encoding="utf-8",
            )
            for index in range(2):
                model = DummyRegressor(strategy="mean").fit(
                    frame[["x1", "x2"]], frame["target"]
                )
                joblib.dump(model, root / f"model_{index}.joblib")

            result = check_demo_bundle(root, check_explanations=True)

        self.assertTrue(result["ready"])
        self.assertEqual(len(result["explanations"]), 2)
        self.assertTrue(all(row["available"] for row in result["explanations"]))
        self.assertEqual(
            {row["method"] for row in result["explanations"]},
            {"permutation_importance"},
        )
