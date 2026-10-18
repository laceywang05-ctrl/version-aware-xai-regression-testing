import json
import tempfile
import unittest
from pathlib import Path

from xai_regression.portfolio import (
    normalize_lime_comparison,
    normalize_shap_benchmark,
    run_cross_dataset_summary,
)


def shap_summary(review_rate=0.5, adjusted_p=0.01):
    return {
        "candidate_run_count": 60,
        "scenario_results": [
            {
                "scenario_id": "seed_only_control",
                "change_type": "control",
                "review_rate": 0.1,
            },
            {
                "scenario_id": "changed",
                "change_type": "model_configuration",
                "review_rate": review_rate,
                "mean_prediction_agreement": 0.99,
                "mean_cosine_similarity": 0.97,
            },
        ],
        "paired_control_comparisons": [
            {"bonferroni_adjusted_p": adjusted_p}
        ],
    }


def lime_summary():
    return {
        "status": "REVIEW",
        "matched_cross_version_comparisons": 200,
        "probe_prediction_agreement": 1.0,
        "noise_matched_cross_version_means": {"cosine_similarity": 0.96},
        "paired_seed_cross_version_means": {"cosine_similarity": 0.99},
        "any_metric_breach_rate": 0.2,
        "flagged_metrics": ["cosine_similarity"],
    }


def heldout_lime_summary():
    return {
        "status": "SEPARATED",
        "matched_candidate_comparisons": 600,
        "probe_prediction_agreement": 1.0,
        "heldout_candidate_means": {"cosine_similarity": 0.97},
        "paired_seed_cross_version_means": {"cosine_similarity": 0.99},
        "heldout_control": {"review_rate": 0.1},
        "candidate": {"review_rate": 0.6},
        "flagged_metrics": ["cosine_similarity"],
    }


class PortfolioTests(unittest.TestCase):
    def test_shap_benchmark_preserves_control_semantics(self):
        row = normalize_shap_benchmark(
            shap_summary(), dataset_id="breast_cancer"
        )

        self.assertEqual(row["status"], "SEPARATED")
        self.assertEqual(row["control_review_rate"], 0.1)
        self.assertEqual(row["headline_value"], 0.5)

    def test_lime_summary_preserves_noise_and_paired_cosine(self):
        row = normalize_lime_comparison(
            lime_summary(), dataset_id="wine_binary"
        )

        self.assertEqual(row["cosine_similarity"], 0.96)
        self.assertEqual(row["paired_seed_cosine_similarity"], 0.99)
        self.assertEqual(row["flagged_metrics"], ["cosine_similarity"])

    def test_heldout_lime_summary_preserves_control_rate(self):
        row = normalize_lime_comparison(
            heldout_lime_summary(), dataset_id="wine_binary"
        )

        self.assertEqual(row["evidence_type"], "held-out paired-seed validation")
        self.assertEqual(row["headline_metric"], "candidate_review_rate")
        self.assertEqual(row["headline_value"], 0.6)
        self.assertEqual(row["control_review_rate"], 0.1)

    def test_cross_dataset_summary_writes_complete_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, payload in (
                ("breast_shap", shap_summary()),
                ("wine_shap", shap_summary(review_rate=0.7)),
                ("breast_lime", lime_summary()),
                ("wine_lime", lime_summary()),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path

            result = run_cross_dataset_summary(
                breast_shap_report=paths["breast_shap"],
                wine_shap_report=paths["wine_shap"],
                breast_lime_report=paths["breast_lime"],
                wine_lime_report=paths["wine_lime"],
                output_dir=root / "output",
            )

            self.assertEqual(result["coverage"]["completed_cells"], 4)
            self.assertEqual(
                {(row["dataset_id"], row["method"]) for row in result["entries"]},
                {
                    ("breast_cancer", "SHAP"),
                    ("breast_cancer", "LIME"),
                    ("wine_binary", "SHAP"),
                    ("wine_binary", "LIME"),
                },
            )
            for filename in (
                "summary.json",
                "report.md",
                "cross_dataset_comparison.csv",
            ):
                self.assertTrue((root / "output/reports" / filename).is_file())


if __name__ == "__main__":
    unittest.main()
