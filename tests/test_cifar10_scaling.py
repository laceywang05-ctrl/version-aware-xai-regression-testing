import io
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import unittest

import numpy as np

from xai_regression.image_bundle import CIFAR10_CLASS_NAMES
from xai_regression.cifar10_scaling import run_cifar10_training_scale


def _write_batch(archive, name, data, labels):
    payload = pickle.dumps({b"data": data, b"labels": labels}, protocol=2)
    info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_cifar_archive(path: Path, train_rows: int, test_rows: int) -> None:
    rng = np.random.default_rng(42)
    with tarfile.open(path, "w:gz") as archive:
        for batch in range(1, 6):
            labels = [index % 2 for index in range(train_rows)]
            data = rng.integers(0, 256, size=(train_rows, 3_072), dtype=np.uint8)
            data[np.asarray(labels) == 1, :200] = 255
            _write_batch(archive, f"data_batch_{batch}", data, labels)
        labels = [index % 2 for index in range(test_rows)]
        data = rng.integers(0, 256, size=(test_rows, 3_072), dtype=np.uint8)
        data[np.asarray(labels) == 1, :200] = 255
        _write_batch(archive, "test_batch", data, labels)
        meta = pickle.dumps(
            {b"label_names": [name.encode() for name in CIFAR10_CLASS_NAMES]},
            protocol=2,
        )
        info = tarfile.TarInfo("cifar-10-batches-py/batches.meta")
        info.size = len(meta)
        archive.addfile(info, io.BytesIO(meta))


class Cifar10ScalingTests(unittest.TestCase):
    def test_scaling_uses_fixed_test_sample_and_writes_summary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "cifar-10-python.tar.gz"
            _write_cifar_archive(archive, train_rows=60, test_rows=30)

            result = run_cifar10_training_scale(
                archive,
                root / "output",
                train_limits=(20, 40),
                test_limit=20,
                n_estimators=2,
            )

            self.assertEqual(result["train_limits"], [20, 40])
            self.assertEqual(result["test_samples"], 20)
            self.assertEqual(len(result["runs"]), 2)
            self.assertTrue(all(len(run["models"]) == 4 for run in result["runs"]))
            self.assertTrue((root / "output" / "summary.json").is_file())

    def test_scaling_rejects_unsorted_training_limits(self):
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            run_cifar10_training_scale("unused", "unused", train_limits=(40, 20))


if __name__ == "__main__":
    unittest.main()
