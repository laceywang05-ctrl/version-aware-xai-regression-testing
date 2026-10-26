import json
import io
import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from xai_regression.image_bundle import (
    CIFAR10_CLASS_NAMES,
    build_image_model_demo,
    prepare_image_npz_bundle,
)


class ImageBundleTests(unittest.TestCase):
    def _write_cifar_archive(self, path: Path) -> None:
        def add_pickle(archive, name, value):
            payload = pickle.dumps(value, protocol=2)
            info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        with tarfile.open(path, "w:gz") as archive:
            for batch_index in range(1, 6):
                labels = np.asarray([(batch_index + index) % 2 for index in range(6)])
                pixels = np.arange(6 * 3_072, dtype=np.uint8).reshape(6, 3_072)
                add_pickle(
                    archive,
                    f"data_batch_{batch_index}",
                    {b"data": pixels, b"labels": labels.tolist()},
                )
            add_pickle(
                archive,
                "test_batch",
                {
                    b"data": np.arange(12 * 3_072, dtype=np.uint8).reshape(12, 3_072),
                    b"labels": [index % 2 for index in range(12)],
                },
            )
            add_pickle(
                archive,
                "batches.meta",
                {b"label_names": [name.encode("utf-8") for name in CIFAR10_CLASS_NAMES]},
            )

    def test_rgb_npz_is_prepared_for_dashboard_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cifar-like.npz"
            images = np.zeros((3, 4, 4, 3), dtype=np.uint8)
            images[1] = 127
            np.savez(source, images=images, labels=np.asarray([0, 1, 1]))
            result = prepare_image_npz_bundle(source, root / "output", limit=2)
            output = root / "output"
            frame = pd.read_csv(output / "samples.csv")
            manifest = json.loads((output / "manifest.json").read_text())
            with ZipFile(output / "images.zip") as archive:
                names = archive.namelist()
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["feature_count"], 48)
            self.assertEqual(len(frame), 2)
            self.assertEqual(len(manifest["feature_columns"]), 48)
            self.assertEqual(manifest["image_shape"], [4, 4, 3])
            self.assertEqual(len(names), 2)

    def test_npz_requires_matching_images_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.npz"
            np.savez(source, images=np.zeros((2, 4, 4)), labels=np.asarray([1]))
            with self.assertRaisesRegex(ValueError, "one value per image"):
                prepare_image_npz_bundle(source, Path(directory) / "output")

    def test_keras_test_split_and_column_labels_are_recognised(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cifar10.npz"
            np.savez(
                source,
                x_train=np.zeros((4, 4, 4, 3), dtype=np.uint8),
                y_train=np.asarray([[0], [1], [0], [1]]),
                x_test=np.ones((3, 4, 4, 3), dtype=np.uint8),
                y_test=np.asarray([[2], [3], [4]]),
            )
            result = prepare_image_npz_bundle(source, root / "output")
            frame = pd.read_csv(root / "output" / "samples.csv")
            manifest = json.loads((root / "output" / "manifest.json").read_text())
            self.assertEqual(result["array_source"], "x_test/y_test")
            self.assertEqual(frame["target"].tolist(), [2, 3, 4])
            self.assertIn("x_test/y_test", manifest["dataset_note"])
            self.assertEqual(manifest["class_names"]["2"], "bird")
            self.assertEqual(result["class_name_source"], "CIFAR-10")

    def test_train_split_is_used_when_test_split_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mnist-train.npz"
            np.savez(
                source,
                x_train=np.zeros((3, 4, 4), dtype=np.uint8),
                y_train=np.asarray([5, 6, 7]),
            )
            result = prepare_image_npz_bundle(source, root / "output")
            self.assertEqual(result["array_source"], "x_train/y_train")

    def test_explicit_npz_class_names_are_written_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "animals.npz"
            np.savez(
                source,
                images=np.zeros((3, 4, 4, 3), dtype=np.uint8),
                labels=np.asarray([0, 1, 1]),
                class_names=np.asarray(["cat", "dog"]),
            )
            result = prepare_image_npz_bundle(source, root / "output")
            manifest = json.loads((root / "output" / "manifest.json").read_text())
            self.assertEqual(manifest["class_names"], {"0": "cat", "1": "dog"})
            self.assertEqual(result["class_name_source"], "NPZ class_names")

    def test_four_model_demo_is_upload_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mnist-like.npz"
            images = np.arange(40 * 4 * 4, dtype=float).reshape(40, 4, 4) % 16
            labels = np.asarray([index % 2 for index in range(40)])
            np.savez(source, images=images, labels=labels)
            result = build_image_model_demo(source, root / "demo", limit=40)
            output = root / "demo"
            self.assertEqual(result["model_count"], 4)
            self.assertTrue((output / "model_4_extra_trees.joblib").is_file())
            self.assertTrue((output / "samples.csv").is_file())
            self.assertTrue((output / "images.zip").is_file())
            summary = json.loads((output / "README.json").read_text())
            self.assertEqual(len(summary["models"]), 4)
            self.assertIn("stratified split", result["split_strategy"])

    def test_four_model_demo_preserves_official_keras_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mnist.npz"
            train_labels = np.asarray([index % 2 for index in range(30)])
            test_labels = np.asarray([index % 2 for index in range(12)])
            np.savez(
                source,
                x_train=np.arange(30 * 4 * 4).reshape(30, 4, 4) % 16,
                y_train=train_labels,
                x_test=np.arange(12 * 4 * 4).reshape(12, 4, 4) % 16,
                y_test=test_labels,
            )
            result = build_image_model_demo(source, root / "demo", limit=100)
            summary = json.loads((root / "demo" / "README.json").read_text())
            self.assertEqual(result["training_samples"], 30)
            self.assertEqual(result["test_samples"], 12)
            self.assertEqual(result["split_strategy"], "official x_train/y_train and x_test/y_test")
            self.assertEqual(summary["split_strategy"], result["split_strategy"])

    def test_four_model_demo_reads_official_cifar10_python_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cifar-10-python.tar.gz"
            self._write_cifar_archive(source)
            result = build_image_model_demo(source, root / "demo", limit=20)
            manifest = json.loads((root / "demo" / "manifest.json").read_text())
            summary = json.loads((root / "demo" / "README.json").read_text())
            frame = pd.read_csv(root / "demo" / "samples.csv")
            self.assertEqual(result["training_samples"], 20)
            self.assertEqual(result["test_samples"], 12)
            self.assertEqual(
                result["split_strategy"], "official CIFAR-10 Python train/test batches"
            )
            self.assertEqual(manifest["class_names"]["0"], "airplane")
            self.assertEqual(manifest["class_names"]["1"], "automobile")
            self.assertEqual(len(manifest["feature_columns"]), 3_072)
            self.assertEqual(manifest["image_shape"], [32, 32, 3])
            self.assertEqual(len(frame), 12)
            self.assertEqual(summary["class_name_source"], "CIFAR-10 batches.meta")
