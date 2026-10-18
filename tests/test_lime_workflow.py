import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from xai_regression.config import load_config
from xai_regression.lime_workflow import (
    apply_sample_policy_flags,
    evaluate_heldout_lime_policy,
    explanation_to_vector,
    lime_class_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeExplanation:
    def __init__(self):
        self.local_exp = {1: [(0, 0.5), (2, -0.25)]}


class LimeWorkflowTests(unittest.TestCase):
    def test_lime_class_names_match_dataset(self):
        breast = load_config(PROJECT_ROOT / "config/lime_cross_version.yaml")
        wine = load_config(PROJECT_ROOT / "config/wine_lime_cross_version.yaml")

        self.assertEqual(lime_class_names(breast), ["benign", "malignant"])
        self.assertEqual(
            lime_class_names(wine),
            ["wine_class_0", "wine_class_1"],
        )

    def test_sparse_explanation_is_mapped_to_feature_order(self):
        vector = explanation_to_vector(
            FakeExplanation(),
            label=1,
            n_features=4,
        )

        np.testing.assert_allclose(vector, [0.5, 0.0, -0.25, 0.0])

    def test_missing_class_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "class label 0"):
            explanation_to_vector(
                FakeExplanation(),
                label=0,
                n_features=4,
            )

    def test_policy_flags_respect_metric_direction(self):
        frame = pd.DataFrame(
            {
                "cosine_similarity": [0.7, 0.9],
                "direction_flip_rate": [0.1, 0.4],
            }
        )
        result = apply_sample_policy_flags(
            frame,
            {"cosine_similarity": 0.8, "direction_flip_rate": 0.3},
        )

        self.assertEqual(
            result["cosine_similarity_breach"].tolist(),
            [True, False],
        )
        self.assertEqual(
            result["direction_flip_rate_breach"].tolist(),
            [False, True],
        )
        self.assertEqual(result["any_metric_breach"].tolist(), [True, True])

    def test_heldout_policy_evaluates_control_and_candidate_by_seed(self):
        config = load_config(PROJECT_ROOT / "config/lime_heldout.yaml")
        calibration = pd.DataFrame(
            [
                {
                    "seed": seed,
                    "cosine_similarity": value,
                    "rank_correlation": value,
                    "top_k_overlap": value,
                    "direction_flip_rate": 1 - value,
                }
                for seed, value in ((1, 0.95), (1, 0.90), (2, 0.85), (2, 0.80))
            ]
        )
        control = pd.DataFrame(
            [
                {
                    "seed": seed,
                    "cosine_similarity": 0.90,
                    "rank_correlation": 0.90,
                    "top_k_overlap": 0.90,
                    "direction_flip_rate": 0.10,
                }
                for seed in (10, 11)
            ]
        )
        candidate = control.copy()
        candidate[[
            "cosine_similarity",
            "rank_correlation",
            "top_k_overlap",
        ]] = 0.10
        candidate["direction_flip_rate"] = 0.90

        policy, control_flagged, candidate_flagged, runs = (
            evaluate_heldout_lime_policy(
                config,
                calibration,
                control,
                candidate,
                candidate_prediction_agreement=0.95,
            )
        )

        self.assertEqual(len(policy), 4)
        self.assertFalse(control_flagged["any_metric_breach"].any())
        self.assertTrue(candidate_flagged["any_metric_breach"].all())
        self.assertEqual(len(runs), 4)
        self.assertEqual(
            runs.groupby("scenario")["review_flag"].sum().to_dict(),
            {
                "heldout_cross_version_candidate": 2,
                "heldout_fixed_model_control": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
