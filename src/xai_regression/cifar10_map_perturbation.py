"""Exploratory pixel-occlusion sensitivity check for CIFAR-10 CNN maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _masked(image: np.ndarray, pixel_indices: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    result = image.reshape(32, 32, 3).copy()
    pixels = result.reshape(-1, 3)
    values = np.asarray(replacement, dtype=np.float32)
    if values.shape == (3,):
        pixels[pixel_indices] = values
    elif values.shape == (32, 32, 3):
        pixels[pixel_indices] = values.reshape(-1, 3)[pixel_indices]
    else:
        raise ValueError("Replacement must be RGB mean or a 32x32 RGB image.")
    return result.reshape(-1)


def _replacement(image: np.ndarray, mean: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return mean
    if mode == "blur":
        return gaussian_filter(image.reshape(32, 32, 3), sigma=(2.0, 2.0, 0.0), mode="reflect")
    raise ValueError("Replacement mode must be 'mean' or 'blur'.")


def _selected_indices(labels: np.ndarray, first: np.ndarray, second: np.ndarray,
                      per_class: int, seed: int) -> list[int]:
    if per_class < 1:
        raise ValueError("per_class must be positive.")
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(10):
        eligible = np.flatnonzero((labels == label) & (first == label) & (second == label))
        if len(eligible) < per_class:
            raise ValueError(f"Class {label} has only {len(eligible)} jointly correct images.")
        selected.extend(int(value) for value in rng.choice(eligible, per_class, replace=False))
    return sorted(selected)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _official_window(run_path: Path, start: int, stop: int, labels: np.ndarray) -> np.ndarray:
    """Verify official test indices and labels before comparing model predictions."""
    with (run_path / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != 10000 or
            [int(row["official_test_index"]) for row in rows] != list(range(10000)) or
            [int(row["actual"]) for row in rows[start:stop]] != labels.tolist()):
        raise ValueError(f"Official-test rows do not align with {run_path}.")
    return np.array([int(row["v2_prediction"]) for row in rows[start:stop]], dtype=int)


def analyze(bundle: Path, baseline_run: Path, candidate_run: Path, *, per_class: int = 3,
            fractions: tuple[float, ...] = (0.1, 0.2), random_repeats: int = 3,
            seed: int = 20261031, replacement_mode: str = "mean",
            archive: Path | None = None, test_start: int = 0, test_stop: int = 1000) -> dict:
    if random_repeats < 1 or any(not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("Require positive random repeats and fractions between 0 and 1.")
    if replacement_mode not in {"mean", "blur"}:
        raise ValueError("Replacement mode must be 'mean' or 'blur'.")
    from xai_regression.neural_explanations import TorchImageClassifier, _torch

    torch = _torch()
    torch.set_num_threads(4)
    runs = [json.loads((path / "summary.json").read_text()) for path in (baseline_run, candidate_run)]
    configs = [run["config"] for run in runs]
    if (configs[0]["seed"] != configs[1]["seed"] or
            configs[0].get("widths", [32, 64, 128]) != [32, 64, 128] or
            configs[1]["widths"] != [40, 80, 160] or
            configs[0]["archive_sha256"] != configs[1]["archive_sha256"] or
            configs[0]["mean"] != configs[1]["mean"]):
        raise ValueError("Expected a matched original-width/wider-width pair.")
    if archive is None:
        if (test_start, test_stop) != (0, 1000):
            raise ValueError("An official test window beyond the upload bundle requires --archive.")
        manifest = json.loads((bundle / "manifest.json").read_text())
        frame = pd.read_csv(bundle / "samples.csv")
        if len(frame) != 1000 or len(manifest["feature_columns"]) != 3072:
            raise ValueError("Expected the 1,000-image RGB CIFAR-10 upload bundle.")
        labels = frame[manifest["label_column"]].to_numpy(dtype=int)
        images = frame[manifest["feature_columns"]].to_numpy(dtype=np.float32)
    else:
        if not 1000 <= test_start < test_stop <= 10000:
            raise ValueError("Archive audit must use official test indices 1000..9999.")
        if _sha256(archive) != configs[0]["archive_sha256"]:
            raise ValueError("CIFAR-10 archive hash differs from the training runs.")
        from xai_regression.image_bundle import _cifar10_batch_arrays, _read_cifar10_member
        with tarfile.open(archive, mode="r:gz") as stream:
            test_images, test_labels = _cifar10_batch_arrays(
                _read_cifar10_member(stream, "test_batch"), "test_batch"
            )
        if len(test_images) != 10000 or len(test_labels) != 10000:
            raise ValueError("Expected all 10,000 official CIFAR-10 test images.")
        images = test_images[test_start:test_stop].reshape(-1, 3072).astype(np.float32)
        labels = test_labels[test_start:test_stop].astype(int)
    if not np.isfinite(images).all():
        raise ValueError("Image pixels must be finite.")
    expected_predictions = [_official_window(path, test_start, test_stop, labels)
                            for path in (baseline_run, candidate_run)]
    model_paths = [bundle / "cifar10_v1_original_widths.joblib",
                   bundle / "cifar10_v2_wider_widths.joblib"]
    models = [joblib.load(path) for path in model_paths]
    if any(not isinstance(model, TorchImageClassifier) or model.image_shape != (32, 32, 3)
           for model in models):
        raise ValueError("Expected trusted TorchImageClassifier RGB models.")
    predictions = [np.argmax(model.predict_proba(images), axis=1) for model in models]
    for run_path, predicted, expected in zip((baseline_run, candidate_run), predictions, expected_predictions):
        if not np.array_equal(predicted, expected):
            raise ValueError(f"Uploaded model predictions do not match {run_path}.")
    selected_local = _selected_indices(labels, *predictions, per_class, seed)
    selected = [index + test_start for index in selected_local]
    mean_pixels = np.asarray(configs[0]["mean"], dtype=np.float32) * 255.0
    rng = np.random.default_rng(seed + 1)
    rows = []
    for index in selected_local:
        image = images[index]
        replacement = _replacement(image, mean_pixels, replacement_mode)
        target = int(labels[index])
        random_masks = {
            str(fraction): [rng.choice(1024, int(np.ceil(1024 * fraction)), replace=False)
                            for _ in range(random_repeats)] for fraction in fractions
        }
        for version, model in zip(("V1", "V2"), models):
            original = float(model.predict_proba(image[None, :])[0, target])
            for method in ("integrated_gradients", "grad_cam"):
                attribution = np.asarray(getattr(model, method)(image[None, :], np.array([target]))[0], dtype=float)
                positive = attribution.reshape(-1)
                for fraction in fractions:
                    count = int(np.ceil(1024 * fraction))
                    # Only positive evidence for the explained class is used.
                    informative = int(np.sum(positive > 1e-8)) >= count
                    top_drop = None
                    random_drop = None
                    if informative:
                        top = np.argsort(-positive, kind="stable")[:count]
                        top_score = float(model.predict_proba(_masked(image, top, replacement)[None, :])[0, target])
                        random_scores = model.predict_proba(np.stack([
                            _masked(image, mask, replacement) for mask in random_masks[str(fraction)]
                        ]))[:, target]
                        top_drop = original - top_score
                        random_drop = original - float(np.mean(random_scores))
                    rows.append({
                        "official_test_index": int(index + test_start), "class": target, "version": version,
                        "method": method, "fraction": fraction, "informative": informative,
                        "original_probability": original, "top_drop": top_drop,
                        "random_drop": random_drop,
                        "top_minus_random": top_drop - random_drop if informative else None,
                    })
    summary = []
    for version in ("V1", "V2"):
        for method in ("integrated_gradients", "grad_cam"):
            for fraction in fractions:
                group = [row for row in rows if row["version"] == version and row["method"] == method
                         and row["fraction"] == fraction]
                informative_rows = [row for row in group if row["informative"]]
                differences = [row["top_minus_random"] for row in informative_rows]
                summary.append({
                    "version": version, "method": method, "fraction": fraction,
                    "informative": len(informative_rows), "selected": len(group),
                    "mean_top_probability_drop": _mean([row["top_drop"] for row in informative_rows]),
                    "mean_random_probability_drop": _mean([row["random_drop"] for row in informative_rows]),
                    "mean_top_minus_random": _mean(differences),
                    "positive_top_minus_random": sum(value > 0 for value in differences),
                })
    return {
        "protocol": "Prediction-conditioned, jointly correct CIFAR-10 images; target true-class probability; fixed-size pixel masking",
        "seed": configs[0]["seed"], "selected_indices": selected,
        "test_window": [test_start, test_stop],
        "per_class": per_class, "fractions": list(fractions), "random_repeats": random_repeats,
        "random_seed": seed, "replacement_mode": replacement_mode,
        "model_sha256": {name: _sha256(path) for name, path in zip(("V1", "V2"), model_paths)},
        "source_archive_sha256": configs[0]["archive_sha256"],
        "summary": summary, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--replacement", choices=("mean", "blur"), default="mean")
    parser.add_argument("--archive", type=Path, help="Use official test images outside the 1,000-image upload bundle")
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--test-stop", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.bundle, args.baseline_run, args.candidate_run,
                     per_class=args.per_class, random_repeats=args.random_repeats,
                     replacement_mode=args.replacement, archive=args.archive,
                     test_start=args.test_start, test_stop=args.test_stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for item in result["summary"]:
        print(item, flush=True)
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
