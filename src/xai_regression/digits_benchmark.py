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

from xai_regression.upload_comparison import compare_classifiers, parse_manifest


MODEL_CONFIGS = (
    {"name": "V1", "n_estimators": 40, "max_depth": 8, "random_state": 17},
    {"name": "V2", "n_estimators": 80, "max_depth": 10, "random_state": 29},
    {"name": "V3", "n_estimators": 120, "max_depth": None, "random_state": 41},
    {"name": "V4", "n_estimators": 80, "max_depth": 6, "random_state": 53},
)
DEFAULT_SPLIT_SEEDS = (11, 23, 37, 49, 61, 73, 89, 101, 127, 149)
DEFAULT_TRAINING_SEEDS = (13, 31, 47, 59, 71, 83, 97, 109, 131, 157)


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


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def run_digits_split_robustness(
    output_dir: str | Path,
    *,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    bootstrap_samples: int = 2_000,
) -> dict:
    """Repeat the close-version benchmark across deterministic data splits."""
    if len(split_seeds) < 2:
        raise ValueError("Split robustness requires at least two split seeds.")
    if bootstrap_samples < 100:
        raise ValueError("Bootstrap samples must be at least 100.")
    digits = load_digits()
    indices = np.arange(len(digits.data))
    runs = []
    for split_seed in split_seeds:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=0.30,
            random_state=split_seed,
            stratify=digits.target,
        )
        train_features = pd.DataFrame(digits.data[train_indices])
        test_features = pd.DataFrame(digits.data[test_indices])
        train_labels = digits.target[train_indices]
        test_labels = digits.target[test_indices]
        metrics = []
        for config in MODEL_CONFIGS:
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=config["random_state"],
                n_jobs=1,
            ).fit(train_features, train_labels)
            prediction = model.predict(test_features)
            metrics.append(
                {
                    "name": config["name"],
                    "accuracy": float(np.mean(prediction == test_labels)),
                    "macro_f1": float(
                        f1_score(test_labels, prediction, average="macro")
                    ),
                }
            )
        runs.append({"split_seed": split_seed, "models": metrics})

    rng = np.random.default_rng(2026)
    summaries = []
    for model_index, config in enumerate(MODEL_CONFIGS):
        accuracies = np.asarray(
            [run["models"][model_index]["accuracy"] for run in runs]
        )
        macro_f1 = np.asarray(
            [run["models"][model_index]["macro_f1"] for run in runs]
        )
        accuracy_ci = _bootstrap_mean_interval(
            accuracies, samples=bootstrap_samples, rng=rng
        )
        f1_ci = _bootstrap_mean_interval(
            macro_f1, samples=bootstrap_samples, rng=rng
        )
        summaries.append(
            {
                "name": config["name"],
                "mean_accuracy": float(accuracies.mean()),
                "accuracy_ci_95": list(accuracy_ci),
                "mean_macro_f1": float(macro_f1.mean()),
                "macro_f1_ci_95": list(f1_ci),
                "best_accuracy": float(accuracies.max()),
                "worst_accuracy": float(accuracies.min()),
            }
        )

    pairwise = []
    for left in range(len(MODEL_CONFIGS)):
        for right in range(left + 1, len(MODEL_CONFIGS)):
            differences = np.asarray(
                [
                    run["models"][right]["accuracy"]
                    - run["models"][left]["accuracy"]
                    for run in runs
                ]
            )
            interval = _bootstrap_mean_interval(
                differences, samples=bootstrap_samples, rng=rng
            )
            pairwise.append(
                {
                    "left": MODEL_CONFIGS[left]["name"],
                    "right": MODEL_CONFIGS[right]["name"],
                    "mean_accuracy_change": float(differences.mean()),
                    "accuracy_change_ci_95": list(interval),
                    "right_wins": int(np.count_nonzero(differences > 0)),
                    "ties": int(np.count_nonzero(differences == 0)),
                    "left_wins": int(np.count_nonzero(differences < 0)),
                }
            )

    report = {
        "dataset": "scikit-learn handwritten digits (8x8)",
        "split_seeds": list(split_seeds),
        "bootstrap_samples": bootstrap_samples,
        "models": summaries,
        "pairwise_accuracy_changes": pairwise,
        "runs": runs,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": destination, "report_path": report_path, **report}


def run_digits_training_robustness(
    output_dir: str | Path,
    *,
    training_seeds: tuple[int, ...] = DEFAULT_TRAINING_SEEDS,
    bootstrap_samples: int = 2_000,
) -> dict:
    """Repeat the four versions on one split with varying forest randomness."""
    if len(training_seeds) < 2:
        raise ValueError("Training robustness requires at least two random seeds.")
    if bootstrap_samples < 100:
        raise ValueError("Bootstrap samples must be at least 100.")
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
    runs = []
    for training_seed in training_seeds:
        metrics = []
        for config in MODEL_CONFIGS:
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=training_seed,
                n_jobs=1,
            ).fit(train_features, train_labels)
            prediction = model.predict(test_features)
            metrics.append(
                {
                    "name": config["name"],
                    "accuracy": float(np.mean(prediction == test_labels)),
                    "macro_f1": float(
                        f1_score(test_labels, prediction, average="macro")
                    ),
                }
            )
        runs.append({"training_seed": training_seed, "models": metrics})

    rng = np.random.default_rng(2027)
    summaries = []
    for model_index, config in enumerate(MODEL_CONFIGS):
        accuracies = np.asarray(
            [run["models"][model_index]["accuracy"] for run in runs]
        )
        macro_f1 = np.asarray(
            [run["models"][model_index]["macro_f1"] for run in runs]
        )
        summaries.append(
            {
                "name": config["name"],
                "mean_accuracy": float(accuracies.mean()),
                "accuracy_ci_95": list(
                    _bootstrap_mean_interval(
                        accuracies, samples=bootstrap_samples, rng=rng
                    )
                ),
                "mean_macro_f1": float(macro_f1.mean()),
                "macro_f1_ci_95": list(
                    _bootstrap_mean_interval(
                        macro_f1, samples=bootstrap_samples, rng=rng
                    )
                ),
                "best_accuracy": float(accuracies.max()),
                "worst_accuracy": float(accuracies.min()),
            }
        )

    pairwise = []
    for left in range(len(MODEL_CONFIGS)):
        for right in range(left + 1, len(MODEL_CONFIGS)):
            differences = np.asarray(
                [
                    run["models"][right]["accuracy"]
                    - run["models"][left]["accuracy"]
                    for run in runs
                ]
            )
            pairwise.append(
                {
                    "left": MODEL_CONFIGS[left]["name"],
                    "right": MODEL_CONFIGS[right]["name"],
                    "mean_accuracy_change": float(differences.mean()),
                    "accuracy_change_ci_95": list(
                        _bootstrap_mean_interval(
                            differences, samples=bootstrap_samples, rng=rng
                        )
                    ),
                    "right_wins": int(np.count_nonzero(differences > 0)),
                    "ties": int(np.count_nonzero(differences == 0)),
                    "left_wins": int(np.count_nonzero(differences < 0)),
                }
            )

    report = {
        "dataset": "scikit-learn handwritten digits (8x8)",
        "fixed_split_seed": 42,
        "training_seeds": list(training_seeds),
        "bootstrap_samples": bootstrap_samples,
        "models": summaries,
        "pairwise_accuracy_changes": pairwise,
        "runs": runs,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": destination, "report_path": report_path, **report}


def run_digits_explanation_stability(
    output_dir: str | Path,
    *,
    sample_limit: int = 120,
    similarity_threshold: float = 0.80,
) -> dict:
    """Compare Tree SHAP maps for three closely performing forest versions."""
    if not 20 <= sample_limit <= 500:
        raise ValueError("Explanation stability sample limit must be 20 to 500.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("Explanation similarity threshold must be between 0 and 1.")
    digits = load_digits()
    columns = [f"pixel_{index}" for index in range(digits.data.shape[1])]
    indices = np.arange(len(digits.data))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=digits.target,
    )
    train_features = pd.DataFrame(digits.data[train_indices], columns=columns)
    test_features = pd.DataFrame(
        digits.data[test_indices][:sample_limit], columns=columns
    )
    train_labels = digits.target[train_indices]
    test_labels = digits.target[test_indices][:sample_limit]
    dataset = test_features.copy()
    dataset["target"] = test_labels
    manifest = parse_manifest(
        {
            "task": "classification",
            "label_column": "target",
            "image_shape": [8, 8],
            "feature_columns": columns,
            "class_names": {str(value): f"digit {value}" for value in range(10)},
            "trusted_model_files": True,
            "dataset_note": "Fixed seed-42 held-out handwritten-digit subset.",
        }
    )
    models = []
    for config in MODEL_CONFIGS[:3]:
        models.append(
            RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=config["random_state"],
                n_jobs=1,
            ).fit(train_features, train_labels)
        )

    pairs = []
    for left in range(len(models)):
        for right in range(left + 1, len(models)):
            comparison = compare_classifiers(
                models[left], models[right], dataset, manifest
            )
            comparable = [
                sample
                for sample in comparison["samples"]
                if sample["explanation_similarity"] is not None
            ]
            similarities = np.asarray(
                [sample["explanation_similarity"] for sample in comparable]
            )
            same_prediction = [
                sample for sample in comparable if not sample["prediction_changed"]
            ]
            hidden_drift = [
                sample
                for sample in same_prediction
                if sample["explanation_similarity"] < similarity_threshold
            ]
            pairs.append(
                {
                    "left": MODEL_CONFIGS[left]["name"],
                    "right": MODEL_CONFIGS[right]["name"],
                    "samples_compared": len(comparable),
                    "mean_cosine_similarity": float(similarities.mean()),
                    "median_cosine_similarity": float(np.median(similarities)),
                    "p10_cosine_similarity": float(np.quantile(similarities, 0.10)),
                    "below_threshold_count": int(
                        np.count_nonzero(similarities < similarity_threshold)
                    ),
                    "same_prediction_count": len(same_prediction),
                    "hidden_drift_count": len(hidden_drift),
                    "prediction_agreement": comparison["summary"][
                        "prediction_agreement"
                    ],
                    "accuracy_change": comparison["summary"]["accuracy_change"],
                }
            )

    report = {
        "dataset": "scikit-learn handwritten digits (8x8)",
        "fixed_split_seed": 42,
        "sample_limit": sample_limit,
        "similarity_threshold": similarity_threshold,
        "explanation_method": "Tree SHAP for predicted-class contributions",
        "pairs": pairs,
        "interpretation": (
            "Cosine similarity compares normalized contribution-map direction. "
            "Values below the inspection threshold are flags, not calibrated failures."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": destination, "report_path": report_path, **report}
