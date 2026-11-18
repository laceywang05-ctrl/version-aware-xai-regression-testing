"""Check that cross-seed overlap uses matched image identities and outcomes."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xai_regression.cifar10_width_consistency import analyze


class WidthConsistencyTests(unittest.TestCase):
    def test_counts_shared_fixes_and_reversed_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            # Image 0 is fixed twice; image 1 is fixed under one seed and
            # regressed under the other, so it cannot be called a stable fix.
            for seed, first, second in ((42, {0: 1, 1: 1}, {0: 0, 1: 0}),
                                        (43, {0: 1}, {0: 0, 1: 1})):
                runs = []
                for name, widths, changes in (("v1", [32, 64, 128], first),
                                              ("v2", [40, 80, 160], second)):
                    run = root / f"{seed}-{name}"
                    run.mkdir()
                    (run / "summary.json").write_text(json.dumps({"config": {
                        "seed": seed, "widths": widths, "archive_sha256": "same-archive",
                    }}))
                    np.savez(run / "split_indices.npz", train=np.array([1, 2]), validation=np.array([3]))
                    with (run / "test_predictions.csv").open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("official_test_index", "actual", "v2_prediction"))
                        for index in range(10000):
                            writer.writerow((index, 0, changes.get(index, 0)))
                    runs.append(run)
                pairs.append((seed, *runs))
            report = analyze(pairs)
            self.assertEqual(report["categories"]["changed"]["all_seeds"], 2)
            self.assertEqual(report["categories"]["fixed"]["all_seeds"], 1)
            self.assertEqual(report["categories"]["fixed"]["any_seed"], 2)
            self.assertEqual(report["categories"]["regressed"]["any_seed"], 1)
            self.assertEqual(report["both_fixed_and_regressed_across_seeds"], 1)


if __name__ == "__main__":
    unittest.main()
