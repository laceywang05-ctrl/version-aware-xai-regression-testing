import unittest
from pathlib import Path

import pandas as pd

from xai_regression.config import load_config
from xai_regression.runner import calibrate_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT_ROOT / "config/default.yaml")

    def test_policy_flags_candidate_beyond_benign_baseline(self):
        benign_rows = []
        for seed in self.config.model.benign_seeds:
            benign_rows.append(
                {
                    "seed": seed,
                    "cosine_similarity": 0.99,
                    "rank_correlation": 0.95,
                    "top_k_overlap": 1.0,
                    "direction_flip_rate": 0.0,
                }
            )
        benign = pd.DataFrame(benign_rows)
        candidate = pd.DataFrame(
            {
                "cosine_similarity": [0.80, 0.85],
                "rank_correlation": [0.70, 0.75],
                "top_k_overlap": [0.60, 0.60],
                "direction_flip_rate": [0.20, 0.30],
            }
        )

        policy, limits = calibrate_policy(self.config, benign, candidate)

        self.assertEqual(set(limits), set(self.config.comparison.metrics))
        self.assertTrue(policy["review_flag"].all())


if __name__ == "__main__":
    unittest.main()
