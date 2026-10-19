import json
import tempfile
import unittest
from pathlib import Path

from xai_regression.model_family import (
    normalize_benchmark,
    run_model_family_summary,
)


def benchmark_summary(dataset, family, separated=(False, True, True)):
    scenario_ids = ("shallow_depth", "larger_leaves", "constrained_candidate")
    return {
        "benchmark": "controlled-model-version-scenarios",
        "dataset": dataset,
        "model_family": family,
        "scenario_results": [
            {
                "scenario_id": "seed_only_control",
                "change_type": "control",
                "review_rate": 0.1,
            },
            *[
                {
                    "scenario_id": scenario_id,
                    "change_type": "model_configuration",
                    "review_rate": 0.8,
                    "mean_prediction_agreement": 0.99,
                    "mean_cosine_similarity": 0.95,
                }
                for scenario_id in scenario_ids
            ],
        ],
        "paired_control_comparisons": [
            {
                "candidate_scenario_id": scenario_id,
                "paired_runs": 30,
                "review_rate_difference": 0.7,
                "bonferroni_adjusted_p": 0.01 if is_separated else 0.2,
            }
            for scenario_id, is_separated in zip(scenario_ids, separated)
        ],
    }


class ModelFamilyTests(unittest.TestCase):
    def test_benchmark_normalization_preserves_model_and_significance(self):
        rows = normalize_benchmark(
            benchmark_summary("breast_cancer", "gradient_boosting"),
            expected_dataset="breast_cancer",
            model_family="gradient_boosting",
        )

        self.assertEqual(rows[0]["model"], "Gradient Boosting")
        self.assertFalse(rows[0]["separated"])
        self.assertTrue(rows[-1]["separated"])

    def test_legacy_random_forest_summary_can_omit_metadata(self):
        summary = benchmark_summary("breast_cancer", "random_forest")
        summary.pop("dataset")
        summary.pop("model_family")

        rows = normalize_benchmark(
            summary,
            expected_dataset="breast_cancer",
            model_family="random_forest",
        )

        self.assertEqual(rows[0]["dataset_id"], "breast_cancer")
        self.assertEqual(rows[0]["model_family"], "random_forest")

    def test_summary_identifies_universal_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for dataset, family in (
                ("breast_cancer", "random_forest"),
                ("wine_binary", "random_forest"),
                ("breast_cancer", "gradient_boosting"),
                ("wine_binary", "gradient_boosting"),
            ):
                path = root / f"{dataset}_{family}.json"
                path.write_text(
                    json.dumps(benchmark_summary(dataset, family)),
                    encoding="utf-8",
                )
                inputs.append(path)

            result = run_model_family_summary(
                random_forest_breast_report=inputs[0],
                random_forest_wine_report=inputs[1],
                gradient_boosting_breast_report=inputs[2],
                gradient_boosting_wine_report=inputs[3],
                output_dir=root / "output",
            )

            self.assertEqual(result["status"], "CROSS_FAMILY_SEPARATION")
            self.assertEqual(
                result["universally_separated_scenarios"],
                ["larger_leaves", "constrained_candidate"],
            )
            self.assertEqual(len(result["entries"]), 12)
            self.assertTrue(
                (root / "output/reports/model_family_comparison.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
