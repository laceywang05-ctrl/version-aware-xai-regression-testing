from io import BytesIO
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from xai_regression.regression_comparison import (
    compare_regression_upload_payloads,
    compare_regressors,
    parse_regression_manifest,
)


class StubRegressor:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, features):
        return self.predictions


class RegressionComparisonTests(unittest.TestCase):
    def test_manifest_requires_regression_numeric_contract(self):
        with self.assertRaisesRegex(ValueError, "task set to regression"):
            parse_regression_manifest(
                {"task": "classification", "label_column": "target", "trusted_model_files": True}
            )
        with self.assertRaisesRegex(ValueError, "trusted_model_files"):
            parse_regression_manifest({"task": "regression", "label_column": "target"})

    def test_comparison_reports_error_improvements_and_regressions(self):
        dataset = pd.DataFrame(
            {"id": ["a", "b", "c"], "x": [0.0, 1.0, 2.0], "target": [1.0, 2.0, 3.0]}
        )
        manifest = parse_regression_manifest(
            {
                "task": "regression",
                "label_column": "target",
                "id_column": "id",
                "feature_columns": ["x"],
                "trusted_model_files": True,
            }
        )

        result = compare_regressors(
            StubRegressor([1.0, 1.0, 4.0]),
            StubRegressor([1.5, 2.0, 5.0]),
            dataset,
            manifest,
        )

        self.assertEqual(result["summary"]["improved_count"], 1)
        self.assertEqual(result["summary"]["worsened_count"], 2)
        self.assertEqual(result["summary"]["unchanged_count"], 0)
        self.assertAlmostEqual(result["summary"]["v1"]["mae"], 2 / 3)
        self.assertAlmostEqual(result["summary"]["v2"]["mae"], 5 / 6)
        self.assertIn("lower", result["summary"]["mae_change_ci_95"])
        self.assertIn("upper", result["summary"]["mae_change_ci_95"])
        self.assertEqual(result["summary"]["mae_change_ci_95"]["resamples"], 5000)
        self.assertEqual(result["samples"][1]["outcome"], "improved")
        self.assertEqual(result["samples"][2]["outcome"], "worsened")

    def test_bootstrap_interval_detects_consistent_mae_improvement(self):
        dataset = pd.DataFrame({"x": [0, 1, 2, 3, 4], "target": [0, 1, 2, 3, 4]})
        manifest = parse_regression_manifest(
            {
                "task": "regression",
                "label_column": "target",
                "feature_columns": ["x"],
                "trusted_model_files": True,
            }
        )
        result = compare_regressors(
            StubRegressor([2, 3, 4, 5, 6]),
            StubRegressor([0.5, 1.5, 2.5, 3.5, 4.5]),
            dataset,
            manifest,
        )
        interval = result["summary"]["mae_change_ci_95"]
        self.assertLess(interval["upper"], 0)
        self.assertTrue(result["summary"]["mae_change_significant"])

    def test_joblib_regressors_and_csv_are_compared_in_memory(self):
        features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        labels = np.asarray([1.0, 3.0, 5.0, 7.0])
        model = LinearRegression().fit(features, labels)
        payload = BytesIO()
        joblib.dump(model, payload)
        manifest = json.dumps(
            {
                "task": "regression",
                "label_column": "target",
                "feature_columns": ["x"],
                "trusted_model_files": True,
            }
        ).encode()

        result = compare_regression_upload_payloads(
            model_v1=("v1.joblib", payload.getvalue()),
            model_v2=("v2.joblib", payload.getvalue()),
            dataset=("samples.csv", features.assign(target=labels).to_csv(index=False).encode()),
            manifest=manifest,
        )

        self.assertEqual(result["task"], "regression")
        self.assertAlmostEqual(result["summary"]["v1"]["rmse"], 0.0)
        self.assertAlmostEqual(result["summary"]["v2"]["r2"], 1.0)
        self.assertEqual(result["models"]["v1"]["filename"], "v1.joblib")


if __name__ == "__main__":
    unittest.main()
