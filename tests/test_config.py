import tempfile
import unittest
from pathlib import Path

from xai_regression.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        config = load_config(PROJECT_ROOT / "config/default.yaml")

        self.assertEqual(config.dataset.name, "breast_cancer")
        self.assertEqual(
            config.output_dir,
            Path("artifacts/experiment_05_real_dataset"),
        )
        self.assertEqual(config.dataset.probe_size, 100)
        self.assertEqual(config.model.reference_seed, 42)
        self.assertEqual(len(config.model.benign_seeds), 10)
        self.assertEqual(config.explainer.method, "tree_shap")
        self.assertEqual(config.comparison.top_k, 5)

    def test_invalid_probe_size_is_rejected(self):
        invalid_yaml = """
project: {name: test}
experiment: {id: invalid-probe, output_dir: artifacts/test}
dataset: {name: breast_cancer, test_size: 0.25, split_seed: 42, probe_size: 0}
model:
  family: random_forest
  n_estimators: 10
  class_weight: balanced
  reference_seed: 42
  benign_seeds: [1]
  candidate: {random_seed: 2, max_depth: 4, min_samples_leaf: 1}
explainer: {method: tree_shap, target_class: malignant}
comparison: {top_k: 2, metrics: [cosine_similarity]}
policy: {lower_quantile: 0.05, upper_quantile: 0.95}
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.yaml"
            path.write_text(invalid_yaml, encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "probe_size"):
                load_config(path)

    def test_lime_stability_config_loads(self):
        config = load_config(PROJECT_ROOT / "config/lime_stability.yaml")

        self.assertEqual(config.explainer.method, "lime")
        self.assertIsNotNone(config.lime)
        self.assertEqual(config.lime.probe_size, 20)
        self.assertEqual(config.lime.num_samples, 1000)
        self.assertEqual(len(config.lime.repeated_seeds), 10)

    def test_lime_heldout_seeds_are_disjoint(self):
        config = load_config(PROJECT_ROOT / "config/lime_heldout.yaml")

        self.assertEqual(len(config.lime.repeated_seeds), 30)
        self.assertEqual(len(config.lime.evaluation_seeds), 30)
        self.assertTrue(
            set(config.lime.repeated_seeds).isdisjoint(
                config.lime.evaluation_seeds
            )
        )

    def test_confirmatory_lime_seeds_are_new_and_disjoint(self):
        earlier = load_config(PROJECT_ROOT / "config/lime_heldout.yaml")
        confirmatory = load_config(
            PROJECT_ROOT / "config/lime_confirmatory.yaml"
        )

        self.assertEqual(len(confirmatory.lime.evaluation_seeds), 30)
        self.assertTrue(
            set(confirmatory.lime.evaluation_seeds).isdisjoint(
                confirmatory.lime.repeated_seeds
            )
        )
        self.assertTrue(
            set(confirmatory.lime.evaluation_seeds).isdisjoint(
                earlier.lime.evaluation_seeds
            )
        )


if __name__ == "__main__":
    unittest.main()
