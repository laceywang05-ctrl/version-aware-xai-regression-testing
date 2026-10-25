"""Evaluate closely matched random-forest versions on handwritten digits."""

from __future__ import annotations

from math import comb
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


MODEL_CONFIGS = (
    {"name": "V1", "n_estimators": 40, "max_depth": 8, "random_state": 17},
    {"name": "V2", "n_estimators": 80, "max_depth": 10, "random_state": 29},
    {"name": "V3", "n_estimators": 120, "max_depth": None, "random_state": 41},
    {"name": "V4", "n_estimators": 80, "max_depth": 6, "random_state": 53},
)


def _mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    cumulative = sum(comb(discordant, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * cumulative / (2**discordant))


def run_digits_close_version_benchmark(output_dir: str | Path) -> dict:
    """Run a deterministic four-version benchmark and save its JSON report."""
    digits = load_digits()
    indices = np.arange(len(digits.data))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=digits.target,
    )
    train_features = pd.DataFrame(digits.data[train_indices])
    test_features = pd.DataFrame(digits.data[test_indices])
    train_labels = digits.target[train_indices]
    test_labels = digits.target[test_indices]
    models = []
    predictions = []
    for config in MODEL_CONFIGS:
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=config["random_state"],
            n_jobs=1,
        ).fit(train_features, train_labels)
        prediction = model.predict(test_features)
        predictions.append(prediction)
        models.append(
            {
                **config,
                "accuracy": float(np.mean(prediction == test_labels)),
                "macro_f1": float(
                    f1_score(test_labels, prediction, average="macro")
                ),
                "correct_count": int(np.count_nonzero(prediction == test_labels)),
            }
        )
    pairs = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            left_correct = predictions[left] == test_labels
            right_correct = predictions[right] == test_labels
            left_only = int(np.count_nonzero(left_correct & ~right_correct))
            right_only = int(np.count_nonzero(~left_correct & right_correct))
            pairs.append(
                {
                    "left": MODEL_CONFIGS[left]["name"],
                    "right": MODEL_CONFIGS[right]["name"],
                    "left_only_correct": left_only,
                    "right_only_correct": right_only,
                    "prediction_agreement": float(
                        np.mean(predictions[left] == predictions[right])
                    ),
                    "mcnemar_exact_p": _mcnemar_p(left_only, right_only),
                }
            )
    report = {
        "dataset": "scikit-learn handwritten digits (8x8)",
        "split": {"random_state": 42, "training_samples": 1257, "test_samples": 540},
        "models": models,
        "pairwise_comparisons": pairs,
        "interpretation": (
            "Exact McNemar p-values test paired correctness differences. "
            "Non-significance does not prove model equivalence."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": destination, "report_path": report_path, **report}
