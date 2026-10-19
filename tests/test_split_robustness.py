import unittest
from pathlib import Path

import pandas as pd

from xai_regression.config import load_config
from xai_regression.split_robustness import (
    config_for_split,
    load_split_robustness_config,
    summarize_split_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SplitRobustnessTests(unittest.TestCase):
    def test_default_scenario_loads(self):
        scenario = load_split_robustness_config(
            PROJECT_ROOT / "config/lime_split_robustness.yaml"
        )

        self.assertEqual(scenario.reference_split_seed, 42)
        self.assertEqual(scenario.new_split_seeds, (17, 73))
        self.assertEqual(scenario.family_alpha, 0.05)

    def test_config_for_split_changes_split_and_output_only(self):
        base = load_config(PROJECT_ROOT / "config/lime_confirmatory.yaml")

        changed = config_for_split(
            base,
            split_seed=17,
            output_dir=Path("artifacts/test-split"),
        )

        self.assertEqual(changed.dataset.split_seed, 17)
        self.assertEqual(changed.lime.output_dir, Path("artifacts/test-split"))
        self.assertEqual(changed.lime.evaluation_seeds, base.lime.evaluation_seeds)
        self.assertEqual(base.dataset.split_seed, 42)

    def test_summary_requires_every_split_to_separate(self):
        rows = []
        for dataset in ("breast_cancer", "wine_binary"):
            for split_seed in (17, 42, 73):
                rows.append(
                    {
                        "dataset": dataset,
                        "split_seed": split_seed,
                        "control_review_rate": 0.05,
                        "candidate_review_rate": 0.50,
                        "status": "SEPARATED",
                    }
                )

        summary = summarize_split_results(pd.DataFrame(rows))

        self.assertEqual(summary["status"], "ROBUST_ACROSS_SPLITS")
        self.assertTrue(
            all(row["all_splits_separated"] for row in summary["datasets"])
        )

    def test_summary_reports_split_sensitivity(self):
        results = pd.DataFrame(
            [
                {
                    "dataset": "breast_cancer",
                    "split_seed": 17,
                    "control_review_rate": 0.10,
                    "candidate_review_rate": 0.50,
                    "status": "SEPARATED",
                },
                {
                    "dataset": "breast_cancer",
                    "split_seed": 42,
                    "control_review_rate": 0.10,
                    "candidate_review_rate": 0.10,
                    "status": "NO_SEPARATION",
                },
            ]
        )

        summary = summarize_split_results(results)

        self.assertEqual(summary["status"], "SPLIT_SENSITIVE")


if __name__ == "__main__":
    unittest.main()
