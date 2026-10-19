import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xai_regression.shap_split_robustness import (
    load_robustness_plan,
    run_shap_split_robustness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def benchmark_result(separated=True):
    return {
        "scenario_results": [
            {
                "scenario_id": "seed_only_control",
                "change_type": "control",
                "review_rate": 0.1,
            },
            {
                "scenario_id": "constrained_candidate",
                "change_type": "model_configuration",
                "review_rate": 0.8,
                "mean_prediction_agreement": 0.99,
                "mean_cosine_similarity": 0.95,
            },
        ],
        "paired_control_comparisons": [
            {
                "paired_runs": 30,
                "review_rate_difference": 0.7,
                "bonferroni_adjusted_p": 0.01 if separated else 0.2,
            }
        ],
    }


class ShapSplitRobustnessTests(unittest.TestCase):
    def _temporary_plan(self, root):
        source = (
            PROJECT_ROOT / "config/shap_split_robustness.yaml"
        ).read_text(encoding="utf-8")
        path = root / "plan.yaml"
        path.write_text(
            source.replace(
                "artifacts/experiment_31_shap_split_robustness",
                str(root / "output"),
            ),
            encoding="utf-8",
        )
        return path

    def test_default_plan_covers_four_cases_and_two_new_splits(self):
        plan = load_robustness_plan(
            PROJECT_ROOT / "config/shap_split_robustness.yaml"
        )

        self.assertEqual(plan.split_seeds, (17, 73))
        self.assertEqual(len(plan.cases), 4)
        self.assertEqual(len(plan.heldout_model_seeds), 30)

    def test_all_separated_cells_produce_robust_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._temporary_plan(root)
            with patch(
                "xai_regression.shap_split_robustness.run_version_benchmark",
                return_value=benchmark_result(),
            ) as run:
                result = run_shap_split_robustness(
                    plan,
                    project_root=PROJECT_ROOT,
                )

            self.assertEqual(result["status"], "ROBUST")
            self.assertEqual(result["coverage"]["comparisons"], 8)
            self.assertEqual(run.call_count, 8)

    def test_one_failed_cell_produces_split_sensitive_status(self):
        results = [benchmark_result(False), *[benchmark_result()] * 7]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._temporary_plan(root)
            with patch(
                "xai_regression.shap_split_robustness.run_version_benchmark",
                side_effect=results,
            ):
                result = run_shap_split_robustness(
                    plan,
                    project_root=PROJECT_ROOT,
                )

        self.assertEqual(result["status"], "SPLIT_SENSITIVE")


if __name__ == "__main__":
    unittest.main()
