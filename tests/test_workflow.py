import unittest
from pathlib import Path

import numpy as np

from xai_regression.config import load_config
from xai_regression.workflow import (
    compare_explanations,
    load_dataset,
    positive_class_shap_values,
    train_reference_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT_ROOT / "config/default.yaml")

    def test_dataset_split_and_probe_are_reproducible(self):
        first = load_dataset(self.config)
        second = load_dataset(self.config)

        self.assertEqual(len(first.X_train) + len(first.X_test), 569)
        self.assertEqual(len(first.X_probe), 100)
        self.assertTrue(first.X_probe.equals(second.X_probe))
        self.assertTrue(first.y_probe.equals(second.y_probe))

    def test_wine_binary_dataset_is_reproducible(self):
        config = load_config(
            PROJECT_ROOT / "config/wine_benchmark_calibration.yaml"
        )
        first = load_dataset(config)
        second = load_dataset(config)

        self.assertEqual(len(first.X_train) + len(first.X_test), 130)
        self.assertEqual(first.X_train.shape[1], 13)
        self.assertEqual(len(first.X_probe), 30)
        self.assertEqual(set(first.y_train.unique()), {0, 1})
        self.assertTrue(first.X_probe.equals(second.X_probe))
        self.assertTrue(first.y_probe.equals(second.y_probe))

    def test_identical_explanations_produce_stable_metrics(self):
        values = np.array(
            [
                [1.0, -2.0, 0.5, 0.2, -0.1],
                [0.2, 0.1, -0.4, 0.7, -0.3],
            ]
        )
        changes = np.array([False, False])
        comparison = compare_explanations(
            self.config,
            values,
            values,
            changes,
        )

        self.assertTrue(
            np.allclose(comparison["cosine_similarity"], 1.0)
        )
        self.assertTrue(np.allclose(comparison["rank_correlation"], 1.0))
        self.assertTrue((comparison["direction_flip_rate"] == 0.0).all())

    def test_gradient_boosting_produces_binary_shap_matrix(self):
        config = load_config(
            PROJECT_ROOT / "config/gradient_boosting_breast_calibration.yaml"
        )
        dataset = load_dataset(config)
        model = train_reference_model(
            config,
            config.model.reference_seed,
            dataset.X_train,
            dataset.y_train,
        )
        values = positive_class_shap_values(model, dataset.X_probe.iloc[:5])

        self.assertEqual(values.shape, (5, dataset.X_train.shape[1]))
        self.assertEqual(model.__class__.__name__, "GradientBoostingClassifier")


if __name__ == "__main__":
    unittest.main()
