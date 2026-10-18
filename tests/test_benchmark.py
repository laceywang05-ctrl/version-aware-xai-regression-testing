import tempfile
import unittest
from pathlib import Path

from xai_regression.benchmark import load_scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_default_scenarios_load(self):
        output_dir, scenarios = load_scenarios(
            PROJECT_ROOT / "config/version_benchmark.yaml"
        )

        self.assertEqual(output_dir.name, "experiment_09_version_benchmark")
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(scenarios[0].change_type, "control")
        self.assertIsNone(scenarios[0].max_depth)

    def test_duplicate_scenario_ids_are_rejected(self):
        content = """
benchmark:
  output_dir: artifacts/test
scenarios:
  - id: repeated
    change_type: control
    description: first
    random_seed: 1
    max_depth: null
    min_samples_leaf: 1
  - id: repeated
    change_type: control
    description: second
    random_seed: 2
    max_depth: null
    min_samples_leaf: 1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate scenario id"):
                load_scenarios(path)


if __name__ == "__main__":
    unittest.main()
