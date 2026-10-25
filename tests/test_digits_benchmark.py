import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from xai_regression.digits_benchmark import run_digits_close_version_benchmark


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


if __name__ == "__main__":
    unittest.main()
