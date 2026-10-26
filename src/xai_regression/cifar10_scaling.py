"""Measure how CIFAR-10 demo performance changes with training-set size."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from xai_regression.image_bundle import _load_cifar10_python_splits


def run_cifar10_training_scale(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    train_limits: tuple[int, ...] = (1_000, 2_500, 5_000),
    test_limit: int = 1_000,
    n_estimators: int = 60,
) -> dict:
    """Benchmark the four demo models across increasing training sizes."""
    if not train_limits or tuple(sorted(set(train_limits))) != train_limits:
        raise ValueError("Training limits must be unique and increasing.")
    if train_limits[0] < 20 or train_limits[-1] > 5_000:
        raise ValueError("Training limits must be between 20 and 5,000.")
    if test_limit < 20 or test_limit > 5_000:
        raise ValueError("Test limit must be between 20 and 5,000.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive.")

    load_limit = max(train_limits[-1], test_limit)
    train_images, train_labels, test_images, test_labels, _ = (
        _load_cifar10_python_splits(input_path, limit=load_limit)
    )
    test_features = test_images[:test_limit].reshape(test_limit, -1)
    heldout_labels = test_labels[:test_limit]
    model_factories = (
        ("dummy", lambda: DummyClassifier(strategy="most_frequent")),
        ("decision_tree", lambda: DecisionTreeClassifier(max_depth=5, random_state=11)),
        (
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=n_estimators, random_state=29, n_jobs=1
            ),
        ),
        (
            "extra_trees",
            lambda: ExtraTreesClassifier(
                n_estimators=n_estimators, random_state=37, n_jobs=1
            ),
        ),
    )
    runs = []
    for train_limit in train_limits:
        features = train_images[:train_limit].reshape(train_limit, -1)
        labels = train_labels[:train_limit]
        models = []
        for name, factory in model_factories:
            model = factory()
            started = perf_counter()
            model.fit(features, labels)
            fit_seconds = perf_counter() - started
            prediction = model.predict(test_features)
            models.append(
                {
                    "name": name,
                    "accuracy": float(np.mean(prediction == heldout_labels)),
                    "macro_f1": float(
                        f1_score(
                            heldout_labels,
                            prediction,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "fit_seconds": fit_seconds,
                }
            )
        runs.append({"training_samples": train_limit, "models": models})

    best_by_size = [
        {
            "training_samples": run["training_samples"],
            **max(run["models"], key=lambda item: item["accuracy"]),
        }
        for run in runs
    ]
    result = {
        "dataset": "CIFAR-10 official Python batches",
        "test_samples": test_limit,
        "train_limits": list(train_limits),
        "n_estimators": n_estimators,
        "runs": runs,
        "best_by_size": best_by_size,
        "interpretation": (
            "Every row uses the same official test subset. Differences therefore "
            "describe training-data scaling for these lightweight demo models."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(summary_path)
    return result
