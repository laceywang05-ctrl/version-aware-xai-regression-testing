import tempfile
import unittest
from pathlib import Path

import pandas as pd

from xai_regression.benchmark import aggregate_scenario_runs, load_scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_default_scenarios_load(self):
        output_dir, scenarios = load_scenarios(
            PROJECT_ROOT / "config/version_benchmark.yaml"
        )

        self.assertEqual(output_dir.name, "experiment_09_version_benchmark")
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(scenarios[0].change_type, "control")
        self.assertIsNone(scenarios[0].max_depth)
        self.assertEqual(len(scenarios[0].random_seeds), 5)

    def test_duplicate_scenario_ids_are_rejected(self):
        content = """
benchmark:
  output_dir: artifacts/test
scenarios:
  - id: repeated
    change_type: control
    description: first
    random_seed: 1
    max_depth: null
    min_samples_leaf: 1
  - id: repeated
    change_type: control
    description: second
    random_seed: 2
    max_depth: null
    min_samples_leaf: 1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate scenario id"):
                load_scenarios(path)

    def test_repeated_runs_are_aggregated(self):
        runs = pd.DataFrame(
            [
                {
                    "scenario_id": "control",
                    "change_type": "control",
                    "description": "seed only",
                    "accuracy": 0.9,
                    "prediction_agreement": 1.0,
                    "mean_cosine_similarity": 0.98,
                    "review_flag": False,
                    "flagged_metrics": "",
                },
                {
                    "scenario_id": "control",
                    "change_type": "control",
                    "description": "seed only",
                    "accuracy": 0.8,
                    "prediction_agreement": 0.9,
                    "mean_cosine_similarity": 0.96,
                    "review_flag": True,
                    "flagged_metrics": "cosine_similarity",
                },
            ]
        )

        summary = aggregate_scenario_runs(runs).iloc[0]

        self.assertEqual(summary["runs"], 2)
        self.assertAlmostEqual(summary["review_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_prediction_agreement"], 0.95)
        self.assertEqual(
            summary["flagged_metric_frequency"],
            "cosine_similarity:1",
        )


if __name__ == "__main__":
    unittest.main()
