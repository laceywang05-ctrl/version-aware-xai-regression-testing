import tempfile
import unittest
from pathlib import Path

import pandas as pd

from xai_regression.benchmark import (
    aggregate_scenario_runs,
    build_paired_control_comparisons,
    exact_mcnemar_p,
    load_scenarios,
    wilson_interval,
)
from xai_regression.config import load_config


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
        self.assertEqual(len(scenarios[0].random_seeds), 30)

    def test_heldout_seeds_are_disjoint_from_calibration(self):
        config = load_config(PROJECT_ROOT / "config/benchmark_calibration.yaml")
        output_dir, scenarios = load_scenarios(
            PROJECT_ROOT / "config/version_benchmark_heldout.yaml"
        )
        heldout = set(scenarios[0].random_seeds)

        self.assertEqual(output_dir.name, "experiment_10_heldout_benchmark")
        self.assertEqual(len(config.model.benign_seeds), 30)
        self.assertEqual(len(heldout), 30)
        self.assertTrue(heldout.isdisjoint(config.model.benign_seeds))
        self.assertNotIn(config.model.reference_seed, heldout)

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
        self.assertLess(summary["review_rate_ci_lower"], 0.5)
        self.assertGreater(summary["review_rate_ci_upper"], 0.5)
        self.assertAlmostEqual(summary["mean_prediction_agreement"], 0.95)
        self.assertEqual(
            summary["flagged_metric_frequency"],
            "cosine_similarity:1",
        )

    def test_wilson_interval_for_known_rate(self):
        lower, upper = wilson_interval(24, 30)

        self.assertAlmostEqual(lower, 0.6269, places=4)
        self.assertAlmostEqual(upper, 0.9049, places=4)

    def test_wilson_interval_rejects_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "between zero and trials"):
            wilson_interval(2, 1)

    def test_exact_mcnemar_uses_discordant_pairs(self):
        control_clear, control_review, p_value = exact_mcnemar_p(
            [False, False, True, True],
            [True, True, True, False],
        )

        self.assertEqual(control_clear, 2)
        self.assertEqual(control_review, 1)
        self.assertEqual(p_value, 1.0)

    def test_paired_control_comparisons_apply_bonferroni(self):
        runs = pd.DataFrame(
            [
                {
                    "scenario_id": scenario,
                    "change_type": change_type,
                    "random_seed": seed,
                    "review_flag": flag,
                }
                for scenario, change_type, flags in [
                    ("control", "control", [False, False, False, True]),
                    ("changed_a", "model_configuration", [True, True, True, True]),
                    ("changed_b", "model_configuration", [False, False, False, True]),
                ]
                for seed, flag in enumerate(flags)
            ]
        )

        comparisons = build_paired_control_comparisons(runs)

        self.assertEqual(len(comparisons), 2)
        self.assertTrue(
            (comparisons["bonferroni_adjusted_p"] >= comparisons["exact_mcnemar_p"]).all()
        )


if __name__ == "__main__":
    unittest.main()
