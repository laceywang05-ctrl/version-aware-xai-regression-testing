import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from xai_regression.digits_benchmark import (
    run_digits_close_version_benchmark,
    run_digits_explanation_stability,
    run_digits_split_robustness,
    run_digits_training_robustness,
)


class DigitsCloseVersionBenchmarkTests(unittest.TestCase):
    def test_benchmark_writes_four_models_and_six_pairs(self):
        with TemporaryDirectory() as temporary:
            result = run_digits_close_version_benchmark(temporary)
            report = json.loads(Path(result["report_path"]).read_text())
            self.assertEqual(report["split"]["test_samples"], 540)
            self.assertEqual(len(report["models"]), 4)
            self.assertEqual(len(report["pairwise_comparisons"]), 6)
            self.assertGreater(report["models"][0]["accuracy"], 0.9)
            self.assertAlmostEqual(
                report["models"][0]["accuracy"], report["models"][1]["accuracy"]
            )
            self.assertEqual(
                report["pairwise_comparisons"][0]["mcnemar_exact_p"], 1.0
            )

    def test_split_robustness_reports_intervals_and_pairwise_changes(self):
        with TemporaryDirectory() as temporary:
            result = run_digits_split_robustness(
                temporary,
                split_seeds=(11, 23),
                bootstrap_samples=100,
            )
            self.assertEqual(len(result["runs"]), 2)
            self.assertEqual(len(result["models"]), 4)
            self.assertEqual(len(result["pairwise_accuracy_changes"]), 6)
            self.assertEqual(len(result["models"][0]["accuracy_ci_95"]), 2)

    def test_training_robustness_holds_the_split_fixed(self):
        with TemporaryDirectory() as temporary:
            result = run_digits_training_robustness(
                temporary,
                training_seeds=(13, 31),
                bootstrap_samples=100,
            )
            self.assertEqual(result["fixed_split_seed"], 42)
            self.assertEqual(len(result["runs"]), 2)
            self.assertEqual(len(result["models"]), 4)
            self.assertEqual(len(result["pairwise_accuracy_changes"]), 6)

    def test_explanation_stability_compares_three_close_model_pairs(self):
        with TemporaryDirectory() as temporary:
            result = run_digits_explanation_stability(
                temporary,
                sample_limit=20,
            )
            self.assertEqual(len(result["pairs"]), 3)
            self.assertTrue(
                all(pair["samples_compared"] == 20 for pair in result["pairs"])
            )
            self.assertTrue(
                all(
                    -1.0 <= pair["mean_cosine_similarity"] <= 1.0
                    for pair in result["pairs"]
                )
            )


if __name__ == "__main__":
    unittest.main()
