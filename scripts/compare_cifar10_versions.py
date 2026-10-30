"""Compare two validation-selected CIFAR-10 runs on the same official test set.

This keeps a genuine model change separate from the early-checkpoint demo. The
output folder also contains a ready-to-upload dashboard bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xai_regression.upload_comparison import _exact_mcnemar_p, _explanation_similarity


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def compare(baseline_run: Path, candidate_run: Path, output: Path) -> dict:
    baseline = json.loads((baseline_run / "summary.json").read_text())
    candidate = json.loads((candidate_run / "summary.json").read_text())
    a, b = baseline["config"], candidate["config"]
    if tuple(a.get("widths", (32, 64, 128))) != (32, 64, 128) or tuple(b.get("widths", ())) != (40, 80, 160):
        raise ValueError("Expected original 32/64/128 and candidate 40/80/160 channel widths.")
    allowed_changes = {"architecture", "widths", "device", "torch_version"}
    for key in (set(a) & set(b)) - allowed_changes:
        if a[key] != b[key]:
            raise ValueError(f"Runs differ beyond the intended model change: {key}")
    with np.load(baseline_run / "split_indices.npz") as first, np.load(candidate_run / "split_indices.npz") as second:
        for key in ("train", "validation"):
            if not np.array_equal(first[key], second[key]):
                raise ValueError(f"Training split changed: {key}")
    first_rows = _rows(baseline_run / "test_predictions.csv")
    second_rows = _rows(candidate_run / "test_predictions.csv")
    if len(first_rows) != 10000 or len(second_rows) != 10000:
        raise ValueError("Both official-test prediction files must have 10,000 rows.")
    expected_indices = list(range(10000))
    if ([int(row["official_test_index"]) for row in first_rows] != expected_indices
            or [int(row["official_test_index"]) for row in second_rows] != expected_indices):
        raise ValueError("Official-test row order does not match.")
    actual = np.array([int(row["actual"]) for row in first_rows])
    if not np.array_equal(actual, [int(row["actual"]) for row in second_rows]):
        raise ValueError("Official-test row labels do not match.")
    v1 = np.array([int(row["v2_prediction"]) for row in first_rows])
    v2 = np.array([int(row["v2_prediction"]) for row in second_rows])
    right1, right2 = v1 == actual, v2 == actual
    changed = v1 != v2
    fixed = ~right1 & right2
    regressed = right1 & ~right2
    other_change = changed & ~fixed & ~regressed
    names = json.loads((baseline_run / "upload_bundle" / "manifest.json").read_text())["class_names"]
    paths = Counter(zip(v1[changed], v2[changed]))
    flow = []
    for (start, end), count in paths.most_common():
        selected = (v1 == start) & (v2 == end)
        flow.append({
            "from": names[str(start)], "to": names[str(end)], "count": count,
            "fixed": int(np.sum(selected & fixed)),
            "regressed": int(np.sum(selected & regressed)),
            "wrong_class_changed": int(np.sum(selected & other_change)),
        })
    classes = []
    for index in range(10):
        mask = actual == index
        classes.append({
            "class": names[str(index)], "samples": int(mask.sum()),
            "v1_recall": float(np.mean(right1[mask])),
            "v2_recall": float(np.mean(right2[mask])),
            "change": float(np.mean(right2[mask]) - np.mean(right1[mask])),
        })
    counts = np.array([int(np.sum(right1 & right2)), int(np.sum(~right1 & right2)),
                       int(np.sum(right1 & ~right2)), int(np.sum(~right1 & ~right2))])
    draws = np.random.default_rng(20261030).multinomial(10000, counts / 10000, size=20000)
    interval = np.quantile((draws[:, 1] - draws[:, 2]) / 10000, [0.025, 0.975])
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "upload_bundle"
    bundle.mkdir(exist_ok=True)
    first_name = "cifar10_v1_original_widths.joblib"
    second_name = "cifar10_v2_wider_widths.joblib"
    shutil.copy2(baseline_run / "upload_bundle" / baseline["models"]["v2"]["filename"], bundle / first_name)
    shutil.copy2(candidate_run / "upload_bundle" / candidate["models"]["v2"]["filename"], bundle / second_name)
    for name in ("samples.csv", "images.zip"):
        shutil.copy2(baseline_run / "upload_bundle" / name, bundle / name)
    manifest = json.loads((baseline_run / "upload_bundle" / "manifest.json").read_text())
    manifest["dataset_note"] = (
        "CIFAR-10 official test images 0..999. V1 uses 32/64/128 channels; V2 uses "
        "40/80/160 channels. Both checkpoints were selected on the same validation "
        "split. Full-test metrics are in directional_comparison.json; explanation "
        "similarity is exploratory."
    )
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Compare maps for a fixed target (the true class), so both models explain
    # the same logit even when their predicted classes differ.
    sample_indices = list(np.flatnonzero(changed[:1000])[:20]) + list(np.flatnonzero(~changed[:1000])[:20])
    frame = pd.read_csv(bundle / "samples.csv")
    features = frame[manifest["feature_columns"]]
    model1, model2 = joblib.load(bundle / first_name), joblib.load(bundle / second_name)
    explanation_rows = []
    for index in sample_indices:
        sample = features.iloc[[int(index)]]
        target = np.array([int(actual[index])])
        map1 = model1.grad_cam(sample, target)[0]
        map2 = model2.grad_cam(sample, target)[0]
        explanation_rows.append({
            "test_index": int(index), "actual": names[str(actual[index])],
            "v1_prediction": names[str(v1[index])], "v2_prediction": names[str(v2[index])],
            "prediction_changed": bool(changed[index]),
            "same_target_gradcam_cosine": _explanation_similarity(map1, map2),
        })
    result = {
        "protocol": "Same 45k/5k stratified split, nominal seed 42, training settings and official 10k test set; convolution widths are the sole intentional design change. The original run resumed after epoch 50 without restoring augmentation RNG, and architecture-dependent initialization can also change RNG streams, so observed differences cannot be attributed solely to width.",
        "v1": {"widths": [32, 64, 128], "parameters": baseline["parameters"], "best_epoch": baseline["best_epoch"], "validation_accuracy": baseline["best_validation_accuracy"], "test": baseline["models"]["v2"]["test"]},
        "v2": {"widths": [40, 80, 160], "parameters": candidate["parameters"], "best_epoch": candidate["best_epoch"], "validation_accuracy": candidate["best_validation_accuracy"], "test": candidate["models"]["v2"]["test"]},
        "prediction_changes": {"changed_class": int(changed.sum()), "agreement": float(np.mean(~changed)),
            "both_correct": int(counts[0]), "wrong_to_correct": int(counts[1]),
            "correct_to_wrong": int(counts[2]), "both_wrong": int(counts[3]),
            "both_wrong_different_class": int(other_change.sum()),
            "accuracy_change": float(np.mean(right2) - np.mean(right1)),
            "paired_bootstrap_95ci": interval.tolist(),
            "exact_mcnemar_p": _exact_mcnemar_p(int(counts[2]), int(counts[1]))},
        "class_recall": classes, "prediction_flows": flow,
        "explanations": {"method": "Grad-CAM", "target": "same true-class logit in both models",
            "scope": "first 20 changed and first 20 unchanged images among dashboard's first 1,000; exploratory, no calibrated threshold",
            "samples": explanation_rows},
        "upload_bundle": str(bundle.resolve()),
    }
    (output / "directional_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline_run, args.candidate_run, args.output_dir)
    print(json.dumps({"prediction_changes": result["prediction_changes"], "upload_bundle": result["upload_bundle"]}, indent=2))
