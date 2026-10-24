from io import BytesIO
import json
import unittest
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from xai_regression.upload_comparison import (
    compare_classifiers,
    compare_upload_payloads,
    load_uploaded_images,
    parse_manifest,
    summarize_upload_models,
    validate_upload_models,
)


class StubClassifier:
    def __init__(self, predictions, probabilities=None):
        self.predictions = np.asarray(predictions)
        self.probabilities = probabilities

    def predict(self, features):
        return self.predictions

    def predict_proba(self, features):
        if self.probabilities is None:
            raise AttributeError
        return np.asarray(self.probabilities)


class UploadComparisonTests(unittest.TestCase):
    def test_manifest_requires_classification_and_trust_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, "classification only"):
            parse_manifest(
                {"task": "regression", "label_column": "target", "trusted_model_files": True}
            )
        with self.assertRaisesRegex(ValueError, "trusted_model_files"):
            parse_manifest({"task": "classification", "label_column": "target"})
        with self.assertRaisesRegex(ValueError, "image_shape"):
            parse_manifest(
                {
                    "task": "classification",
                    "label_column": "target",
                    "image_shape": [8],
                    "trusted_model_files": True,
                }
            )

    def test_comparison_reports_all_correctness_transitions(self):
        dataset = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "x": [0.0, 1.0, 2.0, 3.0],
                "target": [0, 1, 0, 1],
            }
        )
        v1 = StubClassifier(
            [0, 1, 1, 0],
            [[0.9, 0.1], [0.1, 0.9], [0.3, 0.7], [0.6, 0.4]],
        )
        v2 = StubClassifier(
            [0, 0, 0, 0],
            [[0.8, 0.2], [0.7, 0.3], [0.8, 0.2], [0.7, 0.3]],
        )
        manifest = parse_manifest(
            {
                "task": "classification",
                "label_column": "target",
                "id_column": "id",
                "feature_columns": ["x"],
                "class_names": {"0": "negative", "1": "positive"},
                "trusted_model_files": True,
            }
        )

        result = compare_classifiers(v1, v2, dataset, manifest)

        self.assertEqual(result["summary"]["both_correct"], 1)
        self.assertEqual(result["summary"]["v1_correct_v2_wrong"], 1)
        self.assertEqual(result["summary"]["v1_wrong_v2_correct"], 1)
        self.assertEqual(result["summary"]["both_wrong"], 1)
        self.assertEqual(result["samples"][1]["actual"], "positive")
        self.assertAlmostEqual(result["samples"][0]["v1_confidence"], 0.9)
        self.assertEqual(result["class_diagnostics"][0]["class_label"], "negative")
        self.assertAlmostEqual(result["class_diagnostics"][0]["accuracy_change"], 0.5)
        self.assertEqual(result["class_diagnostics"][1]["class_label"], "positive")
        self.assertAlmostEqual(result["class_diagnostics"][1]["accuracy_change"], -0.5)

    def test_missing_label_column_is_rejected(self):
        manifest = parse_manifest(
            {
                "task": "classification",
                "label_column": "target",
                "trusted_model_files": True,
            }
        )
        with self.assertRaisesRegex(ValueError, "label column"):
            compare_classifiers(
                StubClassifier([0]),
                StubClassifier([0]),
                pd.DataFrame({"x": [1]}),
                manifest,
            )

    def test_uploaded_joblib_models_and_csv_are_compared_in_memory(self):
        features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        labels = np.asarray([0, 0, 1, 1])
        model = DecisionTreeClassifier(max_depth=1, random_state=7).fit(
            features, labels
        )
        first = BytesIO()
        second = BytesIO()
        joblib.dump(model, first)
        joblib.dump(model, second)
        dataset = features.assign(target=labels).to_csv(index=False).encode("utf-8")
        manifest = json.dumps(
            {
                "task": "classification",
                "label_column": "target",
                "feature_columns": ["x"],
                "trusted_model_files": True,
            }
        ).encode("utf-8")

        result = compare_upload_payloads(
            model_v1=("v1.joblib", first.getvalue()),
            model_v2=("v2.joblib", second.getvalue()),
            dataset=("samples.csv", dataset),
            manifest=manifest,
        )

        self.assertEqual(result["summary"]["total_samples"], 4)
        self.assertEqual(result["summary"]["prediction_agreement"], 1.0)
        self.assertEqual(result["summary"]["both_correct"], 4)
        self.assertEqual(result["models"]["v1"]["filename"], "v1.joblib")
        self.assertEqual(result["models"]["v2"]["filename"], "v2.joblib")

    def test_uploaded_model_collection_is_ranked_and_compared(self):
        features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        labels = np.asarray([0, 0, 1, 1])
        strong = DecisionTreeClassifier(max_depth=1, random_state=7).fit(
            features, labels
        )
        weak = DummyClassifier(strategy="constant", constant=0).fit(features, labels)
        payloads = []
        for filename, model in (
            ("strong.joblib", strong),
            ("copy.joblib", strong),
            ("weak.joblib", weak),
        ):
            buffer = BytesIO()
            joblib.dump(model, buffer)
            payloads.append((filename, buffer.getvalue()))
        dataset = features.assign(target=labels).to_csv(index=False).encode("utf-8")
        manifest = json.dumps(
            {
                "task": "classification",
                "label_column": "target",
                "feature_columns": ["x"],
                "trusted_model_files": True,
            }
        ).encode("utf-8")

        result = summarize_upload_models(
            models=payloads,
            dataset=("samples.csv", dataset),
            manifest=manifest,
        )

        self.assertEqual(len(result["models"]), 3)
        self.assertEqual(result["models"][0]["accuracy"], 1.0)
        self.assertEqual(result["models"][2]["accuracy"], 0.5)
        self.assertEqual(result["best_model_index"], 0)
        self.assertEqual(len(result["pairwise_agreements"]), 3)
        self.assertEqual(result["pairwise_agreements"][0]["agreement"], 1.0)

    def test_upload_validation_identifies_an_incompatible_model(self):
        features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        labels = np.asarray([0, 0, 1, 1])
        compatible = DecisionTreeClassifier(max_depth=1, random_state=7).fit(
            features, labels
        )
        incompatible = DecisionTreeClassifier(max_depth=1, random_state=7).fit(
            pd.DataFrame({"x": features["x"], "y": features["x"]}), labels
        )
        payloads = []
        for filename, model in (
            ("compatible.joblib", compatible),
            ("incompatible.joblib", incompatible),
        ):
            buffer = BytesIO()
            joblib.dump(model, buffer)
            payloads.append((filename, buffer.getvalue()))
        manifest = json.dumps(
            {
                "task": "classification",
                "label_column": "target",
                "feature_columns": ["x"],
                "trusted_model_files": True,
            }
        ).encode("utf-8")

        result = validate_upload_models(
            models=payloads,
            dataset=(
                "samples.csv",
                features.assign(target=labels).to_csv(index=False).encode("utf-8"),
            ),
            manifest=manifest,
        )

        self.assertFalse(result["ready"])
        self.assertTrue(result["models"][0]["compatible"])
        self.assertFalse(result["models"][1]["compatible"])

    def test_uploaded_image_archive_is_attached_to_matching_samples(self):
        archive = BytesIO()
        with ZipFile(archive, "w") as output:
            output.writestr("digits/sample-1.png", b"\x89PNG\r\n\x1a\nexample")
        images = load_uploaded_images(archive.getvalue(), "samples.zip")
        dataset = pd.DataFrame(
            {"id": ["one"], "image": ["digits/sample-1.png"], "x": [1.0], "target": [1]}
        )
        manifest = parse_manifest(
            {
                "task": "classification",
                "label_column": "target",
                "id_column": "id",
                "image_column": "image",
                "feature_columns": ["x"],
                "trusted_model_files": True,
            }
        )

        result = compare_classifiers(
            StubClassifier([1], [[0.1, 0.9]]),
            StubClassifier([0], [[0.8, 0.2]]),
            dataset,
            manifest,
            images,
        )

        self.assertEqual(result["samples"][0]["image_name"], "digits/sample-1.png")
        self.assertTrue(
            result["samples"][0]["image_data_url"].startswith("data:image/png;base64,")
        )

    def test_image_reference_must_exist_in_archive(self):
        manifest = parse_manifest(
            {
                "task": "classification",
                "label_column": "target",
                "image_column": "image",
                "feature_columns": ["x"],
                "trusted_model_files": True,
            }
        )
        with self.assertRaisesRegex(ValueError, "was not found"):
            compare_classifiers(
                StubClassifier([1], [[0.1, 0.9]]),
                StubClassifier([1], [[0.1, 0.9]]),
                pd.DataFrame({"image": ["missing.png"], "x": [1.0], "target": [1]}),
                manifest,
                {},
            )

    def test_non_tree_classifier_uses_pixel_occlusion_fallback(self):
        dataset = pd.DataFrame(
            {
                "p0": [0.0, 0.0, 1.0, 1.0, 0.2, 0.8],
                "p1": [0.0, 1.0, 0.0, 1.0, 0.1, 0.9],
                "p2": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
                "p3": [0.2, 0.1, 0.9, 0.8, 0.4, 0.6],
                "target": [0, 0, 1, 1, 0, 1],
            }
        )
        features = dataset[["p0", "p1", "p2", "p3"]]
        model = LogisticRegression(random_state=3).fit(features, dataset["target"])
        manifest = parse_manifest(
            {
                "task": "classification",
                "label_column": "target",
                "feature_columns": ["p0", "p1", "p2", "p3"],
                "image_shape": [2, 2],
                "trusted_model_files": True,
            }
        )

        result = compare_classifiers(model, model, dataset, manifest)

        self.assertEqual(
            result["explanations"]["v1"]["method"], "pixel_occlusion"
        )
        self.assertEqual(len(result["samples"][0]["v1_explanation"]), 2)
        self.assertEqual(len(result["samples"][0]["v1_explanation"][0]), 2)


if __name__ == "__main__":
    unittest.main()
