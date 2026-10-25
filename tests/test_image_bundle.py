import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from xai_regression.image_bundle import build_image_model_demo, prepare_image_npz_bundle


class ImageBundleTests(unittest.TestCase):
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
            self.assertNotIn("image_shape", manifest)
            self.assertEqual(len(names), 2)

    def test_npz_requires_matching_images_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.npz"
            np.savez(source, images=np.zeros((2, 4, 4)), labels=np.asarray([1]))
            with self.assertRaisesRegex(ValueError, "one value per image"):
                prepare_image_npz_bundle(source, Path(directory) / "output")

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
