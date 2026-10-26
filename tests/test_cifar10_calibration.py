import io
import pickle
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from xai_regression.cifar10_calibration import run_cifar10_explanation_calibration
from xai_regression.image_bundle import CIFAR10_CLASS_NAMES


def _write_batch(archive, name, data, labels):
    payload = pickle.dumps({b"data": data, b"labels": labels}, protocol=2)
    info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


class Cifar10CalibrationTests(unittest.TestCase):
    def _archive(self, path: Path) -> None:
        rng = np.random.default_rng(42)
        with tarfile.open(path, "w:gz") as archive:
            for batch in range(1, 6):
                labels = [index % 2 for index in range(30)]
                data = rng.integers(0, 256, size=(30, 3_072), dtype=np.uint8)
                data[np.asarray(labels) == 1, :200] = 255
                _write_batch(archive, f"data_batch_{batch}", data, labels)
            labels = [index % 2 for index in range(30)]
            data = rng.integers(0, 256, size=(30, 3_072), dtype=np.uint8)
            data[np.asarray(labels) == 1, :200] = 255
            _write_batch(archive, "test_batch", data, labels)
            meta = pickle.dumps(
                {b"label_names": [name.encode() for name in CIFAR10_CLASS_NAMES]},
                protocol=2,
            )
            info = tarfile.TarInfo("cifar-10-batches-py/batches.meta")
            info.size = len(meta)
            archive.addfile(info, io.BytesIO(meta))

    def test_calibration_separates_training_and_validation_seeds(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "cifar-10-python.tar.gz"
            self._archive(source)
            result = run_cifar10_explanation_calibration(
                source,
                root / "output",
                training_seeds=(11, 29, 47),
                validation_seeds=(59, 71),
                train_limit=120,
                sample_limit=20,
                n_estimators=4,
            )
            self.assertEqual(len(result["pairs"]), 3)
            self.assertGreater(result["same_prediction_comparisons"], 0)
            self.assertGreaterEqual(result["calibrated_threshold"], -1.0)
            self.assertLessEqual(result["calibrated_threshold"], 1.0)
            self.assertEqual(len(result["heldout_validation"]["pairs"]), 1)
            self.assertTrue((root / "output" / "summary.json").is_file())

    def test_calibration_rejects_overlapping_seed_sets(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run_cifar10_explanation_calibration(
                "unused.tar.gz",
                "unused",
                training_seeds=(11, 29, 47),
                validation_seeds=(47, 59),
            )


if __name__ == "__main__":
    unittest.main()
