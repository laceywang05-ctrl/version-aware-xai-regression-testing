"""Validate the explanation-regression workflow on a real tabular dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_regression.config import load_config  # noqa: E402
from xai_regression.workflow import (  # noqa: E402
    compare_explanations,
    load_dataset,
    positive_class_shap_values,
    predictive_metrics,
    train_candidate_model,
    train_reference_model,
    validate_supported_workflow,
)


parser = argparse.ArgumentParser(
    description="Validate explanation regression on a real tabular dataset."
)
parser.add_argument(
    "--config",
    default="config/default.yaml",
    help="Path to the experiment YAML file.",
)
args = parser.parse_args()
CONFIG = load_config(args.config)

validate_supported_workflow(CONFIG)

OUTPUT_DIR = CONFIG.output_dir
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
for directory in (REPORT_DIR, FIGURE_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


dataset = load_dataset(CONFIG)
X_train = dataset.X_train
X_test = dataset.X_test
y_train = dataset.y_train
y_test = dataset.y_test
X_probe = dataset.X_probe
y_probe = dataset.y_probe

probe_to_save = X_probe.reset_index(names="source_row")
probe_to_save.insert(0, "sample_id", range(len(probe_to_save)))
probe_to_save["target_malignant"] = y_probe.to_numpy()
probe_to_save.to_csv(DATA_DIR / "fixed_probe.csv", index=False)

reference_model = train_reference_model(
    CONFIG,
    CONFIG.model.reference_seed,
    X_train,
    y_train,
)
reference_values = positive_class_shap_values(reference_model, X_probe)
reference_predictions = reference_model.predict(X_probe)
reference_performance = predictive_metrics(reference_model, X_test, y_test)

benign_frames = []
benign_run_rows = []

for seed in CONFIG.model.benign_seeds:
    benign_model = train_reference_model(CONFIG, seed, X_train, y_train)
    benign_values = positive_class_shap_values(benign_model, X_probe)
    benign_predictions = benign_model.predict(X_probe)
    comparison = compare_explanations(
        CONFIG,
        reference_values,
        benign_values,
        reference_predictions != benign_predictions,
    )
    comparison.insert(0, "seed", seed)
    benign_frames.append(comparison)

    performance = predictive_metrics(benign_model, X_test, y_test)
    benign_run_rows.append(
        {
            "seed": seed,
            **performance,
            "prediction_agreement": 1.0 - comparison["prediction_changed"].mean(),
            "mean_cosine_similarity": comparison["cosine_similarity"].mean(),
            "mean_rank_correlation": comparison["rank_correlation"].mean(),
            "mean_top_k_overlap": comparison["top_k_overlap"].mean(),
            "mean_direction_flip_rate": comparison["direction_flip_rate"].mean(),
        }
    )

benign_samples = pd.concat(benign_frames, ignore_index=True)
benign_runs = pd.DataFrame(benign_run_rows)

candidate_model = train_candidate_model(CONFIG, X_train, y_train)
candidate_values = positive_class_shap_values(candidate_model, X_probe)
candidate_predictions = candidate_model.predict(X_probe)
candidate_performance = predictive_metrics(candidate_model, X_test, y_test)
candidate_comparison = compare_explanations(
    CONFIG,
    reference_values,
    candidate_values,
    reference_predictions != candidate_predictions,
)

sample_limits = {
    "cosine_similarity": float(
        benign_samples["cosine_similarity"].quantile(
            CONFIG.policy.lower_quantile
        )
    ),
    "rank_correlation": float(
        benign_samples["rank_correlation"].quantile(
            CONFIG.policy.lower_quantile
        )
    ),
    "top_k_overlap": float(
        benign_samples["top_k_overlap"].quantile(
            CONFIG.policy.lower_quantile
        )
    ),
    "direction_flip_rate": float(
        benign_samples["direction_flip_rate"].quantile(
            CONFIG.policy.upper_quantile
        )
    ),
}

rules = [
    ("cosine_similarity", "lower"),
    ("rank_correlation", "lower"),
    ("top_k_overlap", "lower"),
    ("direction_flip_rate", "upper"),
]


def breach_mask(frame, metric, direction):
    if direction == "lower":
        return frame[metric] < sample_limits[metric]
    return frame[metric] > sample_limits[metric]


policy_rows = []
for metric, direction in rules:
    baseline_rates = (
        benign_samples.assign(
            breach=breach_mask(benign_samples, metric, direction)
        )
        .groupby("seed")["breach"]
        .mean()
    )
    run_level_limit = float(
        baseline_rates.quantile(CONFIG.policy.upper_quantile)
    )
    candidate_rate = float(
        breach_mask(candidate_comparison, metric, direction).mean()
    )
    policy_rows.append(
        {
            "metric": metric,
            "sample_limit_direction": direction,
            "sample_limit": sample_limits[metric],
            "benign_mean_breach_rate": float(baseline_rates.mean()),
            "benign_max_breach_rate": float(baseline_rates.max()),
            "run_level_limit_95_percent": run_level_limit,
            "candidate_breach_rate": candidate_rate,
            "review_flag": bool(candidate_rate > run_level_limit),
        }
    )

policy = pd.DataFrame(policy_rows)
overall_status = "REVIEW" if policy["review_flag"].any() else "WITHIN_BASELINE"

mean_abs_reference = np.abs(reference_values).mean(axis=0)
mean_abs_candidate = np.abs(candidate_values).mean(axis=0)
global_features = pd.DataFrame(
    {
        "feature": X_probe.columns,
        "mean_abs_reference": mean_abs_reference,
        "mean_abs_candidate": mean_abs_candidate,
    }
)
global_features["rank_reference"] = global_features["mean_abs_reference"].rank(
    ascending=False,
    method="min",
)
global_features["rank_candidate"] = global_features["mean_abs_candidate"].rank(
    ascending=False,
    method="min",
)
global_features["rank_change"] = (
    global_features["rank_candidate"] - global_features["rank_reference"]
)
global_features = global_features.sort_values("rank_reference")

benign_samples.to_csv(REPORT_DIR / "benign_per_sample.csv", index=False)
benign_runs.to_csv(REPORT_DIR / "benign_run_summary.csv", index=False)
candidate_comparison.to_csv(
    REPORT_DIR / "candidate_per_sample.csv",
    index=False,
)
policy.to_csv(REPORT_DIR / "policy_evaluation.csv", index=False)
global_features.to_csv(REPORT_DIR / "global_feature_comparison.csv", index=False)

top_features = global_features.head(10).iloc[::-1]
positions = np.arange(len(top_features))
height = 0.38
figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].barh(
    positions - height / 2,
    top_features["mean_abs_reference"],
    height,
    label="reference",
)
axes[0].barh(
    positions + height / 2,
    top_features["mean_abs_candidate"],
    height,
    label="candidate",
)
axes[0].set_yticks(positions, top_features["feature"])
axes[0].set_xlabel("Mean absolute SHAP value")
axes[0].set_title("Top Global Features")
axes[0].legend()

axes[1].bar(
    policy["metric"],
    policy["candidate_breach_rate"],
    label="candidate breach rate",
)
axes[1].scatter(
    policy["metric"],
    policy["run_level_limit_95_percent"],
    color="#D95F02",
    marker="_",
    s=500,
    linewidths=3,
    label="benign run-level limit",
)
axes[1].tick_params(axis="x", rotation=25)
axes[1].set_ylabel("Fraction of probe samples")
axes[1].set_title("Provisional Policy Evaluation")
axes[1].legend()

figure.suptitle("Real-Dataset Explanation Regression Validation")
figure.tight_layout()
figure.savefig(FIGURE_DIR / "real_dataset_validation.png", dpi=180)
plt.close(figure)

summary = {
    "experiment_id": CONFIG.experiment_id,
    "config_path": str(Path(args.config)),
    "dataset": "scikit-learn Breast Cancer Wisconsin Diagnostic",
    "samples": int(len(X_train) + len(X_test)),
    "features": int(X_train.shape[1]),
    "probe_samples": int(len(X_probe)),
    "positive_class": "malignant",
    "reference_performance": reference_performance,
    "candidate_performance": candidate_performance,
    "candidate_prediction_agreement": float(
        1.0 - candidate_comparison["prediction_changed"].mean()
    ),
    "candidate_explanation_metrics": {
        "mean_cosine_similarity": float(
            candidate_comparison["cosine_similarity"].mean()
        ),
        "mean_rank_correlation": float(
            candidate_comparison["rank_correlation"].mean()
        ),
        "mean_top_k_overlap": float(
            candidate_comparison["top_k_overlap"].mean()
        ),
        "mean_direction_flip_rate": float(
            candidate_comparison["direction_flip_rate"].mean()
        ),
    },
    "sample_limits": sample_limits,
    "status": overall_status,
    "flagged_metrics": policy.loc[policy["review_flag"], "metric"].tolist(),
    "interpretation": (
        "A review flag indicates explanation changes beyond the provisional "
        "benign-retraining baseline; it is not proof of a harmful regression."
    ),
}

(REPORT_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("Real-dataset validation")
print("-----------------------")
print(f"Dataset samples/features: {summary['samples']}/{summary['features']}")
print(f"Fixed probe samples: {summary['probe_samples']}")
print(f"Reference accuracy: {reference_performance['accuracy']:.3f}")
print(f"Candidate accuracy: {candidate_performance['accuracy']:.3f}")
print(f"Prediction agreement: {summary['candidate_prediction_agreement']:.3f}")
print(
    "Mean explanation cosine similarity: "
    f"{summary['candidate_explanation_metrics']['mean_cosine_similarity']:.3f}"
)
print(f"Overall status: {overall_status}")
print(
    "Flagged metrics: "
    f"{', '.join(summary['flagged_metrics']) if summary['flagged_metrics'] else 'none'}"
)
print(f"Reports saved to: {REPORT_DIR}")
print(f"Figure saved to: {FIGURE_DIR}")
