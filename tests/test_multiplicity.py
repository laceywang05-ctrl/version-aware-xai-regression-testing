import unittest

import pandas as pd

from xai_regression.multiplicity import (
    evaluate_multiplicity_rules,
    seed_breach_rates,
)


class MultiplicityPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = pd.DataFrame(
            [
                {
                    "metric": "cosine_similarity",
                    "sample_limit_direction": "lower",
                    "sample_limit": 0.8,
                },
                {
                    "metric": "direction_flip_rate",
                    "sample_limit_direction": "upper",
                    "sample_limit": 0.3,
                },
            ]
        )

    def test_seed_breach_rates_respect_metric_direction(self):
        samples = pd.DataFrame(
            {
                "seed": [1, 1, 2, 2],
                "cosine_similarity": [0.7, 0.9, 0.9, 0.9],
                "direction_flip_rate": [0.1, 0.4, 0.1, 0.1],
            }
        )

        result = seed_breach_rates(samples, self.policy)

        self.assertEqual(result["cosine_similarity"].tolist(), [0.5, 0.0])
        self.assertEqual(result["direction_flip_rate"].tolist(), [0.5, 0.0])

    def test_bonferroni_uses_a_higher_run_limit_quantile(self):
        calibration_rows = []
        for seed in range(1, 21):
            calibration_rows.extend(
                {
                    "seed": seed,
                    "cosine_similarity": 0.7 if sample < seed % 4 else 0.9,
                    "direction_flip_rate": 0.4 if sample < seed % 3 else 0.1,
                }
                for sample in range(10)
            )
        control = pd.DataFrame(
            {
                "seed": [101, 101, 102, 102],
                "cosine_similarity": [0.9, 0.9, 0.7, 0.9],
                "direction_flip_rate": [0.1, 0.1, 0.1, 0.1],
            }
        )
        candidate = control.copy()
        candidate["cosine_similarity"] = 0.1

        comparison, limits, decisions = evaluate_multiplicity_rules(
            pd.DataFrame(calibration_rows),
            control,
            candidate,
            self.policy,
            dataset="synthetic",
        )

        quantiles = comparison.set_index("rule")["run_limit_quantile"]
        self.assertGreater(quantiles["bonferroni"], quantiles["unadjusted"])
        self.assertEqual(len(limits), 4)
        self.assertEqual(len(decisions), 8)
        candidate_counts = comparison.set_index("rule")["candidate_review_count"]
        self.assertEqual(candidate_counts["unadjusted"], 2)
        self.assertEqual(candidate_counts["bonferroni"], 2)

    def test_invalid_alpha_is_rejected(self):
        samples = pd.DataFrame(
            {
                "seed": [1],
                "cosine_similarity": [0.9],
                "direction_flip_rate": [0.1],
            }
        )
        with self.assertRaisesRegex(ValueError, "alpha"):
            evaluate_multiplicity_rules(
                samples,
                samples,
                samples,
                self.policy,
                dataset="synthetic",
                alpha=0,
            )


if __name__ == "__main__":
    unittest.main()
