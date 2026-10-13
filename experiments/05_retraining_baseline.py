"""Estimate normal explanation variation caused by stochastic retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_regression.metrics import (  # noqa: E402
    absolute_rank_correlation,
    cosine_similarity,
    direction_flip_rate,
    top_k_overlap,
)


DATA_SEED = 42
REFERENCE_SEED = 42
RETRAINING_SEEDS = [7, 21, 55, 84, 126, 168, 210, 252, 294, 336]
TOP_K = 3

OUTPUT_DIR = Path("artifacts/experiment_03_retraining_baseline")
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def make_fixed_data():
    """Recreate the same dataset and split used by experiment 02."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=DATA_SEED,
    )
    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=DATA_SEED,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, X_test.iloc[:100].copy()


def train_model(seed, X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    return model


def positive_class_shap_values(model, X_probe):
    values = np.asarray(shap.TreeExplainer(model)(X_probe).values)

    if values.ndim == 3:
        if values.shape[2] != 2:
            raise RuntimeError(f"Expected binary-class SHAP values, got {values.shape}.")
        values = values[:, :, 1]
    elif values.ndim != 2:
        raise RuntimeError(f"Expected a 2D explanation array, got {values.shape}.")

    return values


X_train, X_test, y_train, y_test, X_probe = make_fixed_data()

reference_model = train_model(REFERENCE_SEED, X_train, y_train)
reference_predictions = reference_model.predict(X_probe)
reference_values = positive_class_shap_values(reference_model, X_probe)
reference_accuracy = accuracy_score(y_test, reference_model.predict(X_test))

per_sample_rows = []
run_rows = []

for seed in RETRAINING_SEEDS:
    candidate_model = train_model(seed, X_train, y_train)
    candidate_predictions = candidate_model.predict(X_probe)
    candidate_values = positive_class_shap_values(candidate_model, X_probe)
    candidate_accuracy = accuracy_score(y_test, candidate_model.predict(X_test))

    sample_metrics = []
    for sample_id, (left, right) in enumerate(zip(reference_values, candidate_values)):
        row = {
            "seed": seed,
            "sample_id": sample_id,
            "prediction_changed": bool(
                reference_predictions[sample_id] != candidate_predictions[sample_id]
            ),
            "cosine_similarity": cosine_similarity(left, right),
            "rank_correlation": absolute_rank_correlation(left, right),
            "top_k_overlap": top_k_overlap(left, right, TOP_K),
            "direction_flip_rate": direction_flip_rate(left, right),
        }
        per_sample_rows.append(row)
        sample_metrics.append(row)

    sample_frame = pd.DataFrame(sample_metrics)
    run_rows.append(
        {
            "seed": seed,
            "reference_accuracy": reference_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "accuracy_delta": candidate_accuracy - reference_accuracy,
            "prediction_agreement": 1.0 - sample_frame["prediction_changed"].mean(),
            "mean_cosine_similarity": sample_frame["cosine_similarity"].mean(),
            "mean_rank_correlation": sample_frame["rank_correlation"].mean(),
            "mean_top_k_overlap": sample_frame["top_k_overlap"].mean(),
            "mean_direction_flip_rate": sample_frame["direction_flip_rate"].mean(),
        }
    )

per_sample = pd.DataFrame(per_sample_rows)
runs = pd.DataFrame(run_rows)

per_sample.to_csv(REPORT_DIR / "per_sample_retraining_comparison.csv", index=False)
runs.to_csv(REPORT_DIR / "retraining_run_summary.csv", index=False)

# These are provisional empirical limits, not universal pass/fail thresholds.
provisional_limits = {
    "basis": "Benign retraining with unchanged data and model hyperparameters",
    "retraining_seeds": RETRAINING_SEEDS,
    "sample_comparisons": int(len(per_sample)),
    "cosine_similarity_lower_5_percent": float(
        per_sample["cosine_similarity"].quantile(0.05)
    ),
    "rank_correlation_lower_5_percent": float(
        per_sample["rank_correlation"].quantile(0.05)
    ),
    "top_k_overlap_lower_5_percent": float(
        per_sample["top_k_overlap"].quantile(0.05)
    ),
    "direction_flip_rate_upper_95_percent": float(
        per_sample["direction_flip_rate"].quantile(0.95)
    ),
}

with (REPORT_DIR / "provisional_limits.json").open("w", encoding="utf-8") as handle:
    json.dump(provisional_limits, handle, indent=2)

figure, axes = plt.subplots(2, 2, figsize=(10, 7))
plot_specs = [
    ("cosine_similarity", "Cosine similarity"),
    ("rank_correlation", "Absolute-rank correlation"),
    ("top_k_overlap", f"Top-{TOP_K} overlap"),
    ("direction_flip_rate", "Direction-flip rate"),
]

for axis, (column, title) in zip(axes.flat, plot_specs):
    axis.hist(per_sample[column], bins=20, color="#2C7FB8", edgecolor="white")
    axis.set_title(title)
    axis.set_ylabel("Matched sample comparisons")

figure.suptitle("Explanation Variation Under Benign Random-Forest Retraining")
figure.tight_layout()
figure.savefig(FIGURE_DIR / "retraining_metric_distributions.png", dpi=180)
plt.close(figure)

print("Benign retraining baseline")
print("--------------------------")
print(f"Reference accuracy: {reference_accuracy:.3f}")
print(f"Retrained models: {len(runs)}")
print(f"Matched sample comparisons: {len(per_sample)}")
print(f"Mean prediction agreement: {runs['prediction_agreement'].mean():.3f}")
print(f"Mean cosine similarity: {per_sample['cosine_similarity'].mean():.3f}")
print(f"Mean rank correlation: {per_sample['rank_correlation'].mean():.3f}")
print(f"Mean top-{TOP_K} overlap: {per_sample['top_k_overlap'].mean():.3f}")
print(f"Mean direction-flip rate: {per_sample['direction_flip_rate'].mean():.3f}")
print("\nProvisional empirical limits (5th/95th percentiles)")
print(json.dumps(provisional_limits, indent=2))
print(f"\nReports saved to: {REPORT_DIR}")
print(f"Figure saved to: {FIGURE_DIR}")
