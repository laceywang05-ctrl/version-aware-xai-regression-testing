import tempfile
import unittest
from pathlib import Path

import pandas as pd

from xai_regression.shap_multiplicity import (
    evaluate_shap_run_rules,
    run_shap_multiplicity_analysis,
)


METRICS = (
    "cosine_similarity",
    "rank_correlation",
    "top_k_overlap",
    "direction_flip_rate",
)


def policy_frame():
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "sample_limit_direction": "upper" if metric == "direction_flip_rate" else "lower",
                "sample_limit": 0.5,
            }
            for metric in METRICS
        ]
    )


def sample_frame(seeds, scenario_id=None, candidate=False):
    rows = []
    for seed in seeds:
        for sample_id in range(10):
            row = {
                "seed" if scenario_id is None else "random_seed": seed,
                "sample_id": sample_id,
                "cosine_similarity": 0.1 if candidate else 0.9,
                "rank_correlation": 0.1 if candidate else 0.9,
                "top_k_overlap": 0.1 if candidate else 0.9,
                "direction_flip_rate": 0.9 if candidate else 0.1,
            }
            if scenario_id is not None:
                row["scenario_id"] = scenario_id
            rows.append(row)
    return pd.DataFrame(rows)


class ShapMultiplicityTests(unittest.TestCase):
    def test_rules_retain_clear_candidate_separation(self):
        calibration = sample_frame(range(1, 31))
        scenarios = pd.concat(
            [
                sample_frame(range(101, 131), "seed_only_control"),
                sample_frame(
                    range(101, 131), "constrained_candidate", candidate=True
                ),
            ],
            ignore_index=True,
        )

        comparison, decisions = evaluate_shap_run_rules(
            calibration,
            scenarios,
            policy_frame(),
            dataset="breast_cancer",
            model_family="random_forest",
            split_seed=17,
        )

        self.assertEqual(set(comparison["rule"]), {"unadjusted", "bonferroni"})
        self.assertTrue(comparison["separated"].all())
        self.assertEqual(len(decisions), 120)

    def test_analysis_requires_saved_benign_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "runs/case/split_17/reports"
            reports.mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "benign_per_sample"):
                run_shap_multiplicity_analysis(
                    robustness_dir=root,
                    output_dir=root / "output",
                )


if __name__ == "__main__":
    unittest.main()
