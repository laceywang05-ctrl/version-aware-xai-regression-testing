import unittest

import pandas as pd

from xai_regression.reporting import normalize_method_reports


class ReportingTests(unittest.TestCase):
    def test_shap_summary_is_normalized(self):
        summary = {
            "status": "REVIEW",
            "probe_samples": 10,
            "candidate_prediction_agreement": 0.9,
            "reference_performance": {"accuracy": 0.95},
            "candidate_performance": {"accuracy": 0.93},
            "flagged_metrics": ["cosine_similarity"],
            "candidate_explanation_metrics": {
                "mean_cosine_similarity": 0.8,
                "mean_rank_correlation": 0.7,
                "mean_top_k_overlap": 0.6,
                "mean_direction_flip_rate": 0.2,
            },
        }
        policy = pd.DataFrame(
            {
                "metric": [
                    "cosine_similarity",
                    "rank_correlation",
                    "top_k_overlap",
                    "direction_flip_rate",
                ],
                "sample_limit": [0.7, 0.5, 0.4, 0.3],
                "candidate_breach_rate": [0.2, 0.1, 0.1, 0.1],
                "run_level_limit_95_percent": [0.1, 0.1, 0.1, 0.1],
                "review_flag": [True, False, False, False],
            }
        )
        baseline = pd.DataFrame(
            {
                "cosine_similarity": [0.9],
                "rank_correlation": [0.8],
                "top_k_overlap": [0.7],
                "direction_flip_rate": [0.1],
            }
        )

        method, metrics = normalize_method_reports(
            "SHAP", summary, policy, baseline
        )

        self.assertEqual(method["baseline_type"], "benign model retraining")
        self.assertEqual(method["flagged_metrics"], "cosine_similarity")
        self.assertEqual(metrics[0]["candidate_mean"], 0.8)
        self.assertTrue(metrics[0]["review_flag"])

    def test_unknown_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            normalize_method_reports("IG", {}, pd.DataFrame(), pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
