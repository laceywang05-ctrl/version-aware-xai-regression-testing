"""End-to-end orchestration for the real-dataset validation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from xai_regression.config import ExperimentConfig
from xai_regression.workflow import (
    compare_explanations,
    load_dataset,
    positive_class_shap_values,
    predictive_metrics,
    train_candidate_model,
    train_reference_model,
    validate_supported_workflow,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_DIRECTIONS = {
    "cosine_similarity": "lower",
    "rank_correlation": "lower",
    "top_k_overlap": "lower",
    "direction_flip_rate": "upper",
}


def _breach_mask(frame, metric, direction, limit):
    if direction == "lower":
        return frame[metric] < limit
    return frame[metric] > limit


def calibrate_policy(
    config: ExperimentConfig,
    benign_samples: pd.DataFrame,
    candidate_comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calibrate sample and run-level limits from benign retraining."""
    sample_limits = {}
    policy_rows = []

    for metric in config.comparison.metrics:
        direction = METRIC_DIRECTIONS.get(metric)
        if direction is None:
            raise ValueError(f"Unsupported comparison metric: {metric}")
        quantile = (
            config.policy.lower_quantile
            if direction == "lower"
            else config.policy.upper_quantile
        )
        sample_limit = float(benign_samples[metric].quantile(quantile))
        sample_limits[metric] = sample_limit
        benign_rates = (
            benign_samples.assign(
                breach=_breach_mask(
                    benign_samples,
                    metric,
                    direction,
                    sample_limit,
                )
            )
            .groupby("seed")["breach"]
            .mean()
        )
        run_level_limit = float(
            benign_rates.quantile(config.policy.upper_quantile)
        )
        candidate_rate = float(
            _breach_mask(
                candidate_comparison,
                metric,
                direction,
                sample_limit,
            ).mean()
        )
        policy_rows.append(
            {
                "metric": metric,
                "sample_limit_direction": direction,
                "sample_limit": sample_limit,
                "benign_mean_breach_rate": float(benign_rates.mean()),
                "benign_max_breach_rate": float(benign_rates.max()),
                "run_level_limit_95_percent": run_level_limit,
                "candidate_breach_rate": candidate_rate,
                "review_flag": bool(candidate_rate > run_level_limit),
            }
        )
    return pd.DataFrame(policy_rows), sample_limits


def build_global_feature_comparison(
    feature_names,
    reference_values,
    candidate_values,
):
    """Build a global mean-absolute-attribution ranking comparison."""
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_reference": np.abs(reference_values).mean(axis=0),
            "mean_abs_candidate": np.abs(candidate_values).mean(axis=0),
        }
    )
    frame["rank_reference"] = frame["mean_abs_reference"].rank(
        ascending=False,
        method="min",
    )
    frame["rank_candidate"] = frame["mean_abs_candidate"].rank(
        ascending=False,
        method="min",
    )
    frame["rank_change"] = frame["rank_candidate"] - frame["rank_reference"]
    return frame.sort_values("rank_reference")


def save_validation_figure(global_features, policy, path):
    """Save the global-feature and provisional-policy summary figure."""
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
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_validation(
    config: ExperimentConfig,
    *,
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Run the complete configured validation and write reproducible artifacts."""
    validate_supported_workflow(config)
    root = Path(project_root)
    output_dir = config.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    report_dir = output_dir / "reports"
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    for directory in (report_dir, figure_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config)
    probe = dataset.X_probe.reset_index(names="source_row")
    probe.insert(0, "sample_id", range(len(probe)))
    probe["target_malignant"] = dataset.y_probe.to_numpy()
    probe.to_csv(data_dir / "fixed_probe.csv", index=False)

    reference_model = train_reference_model(
        config,
        config.model.reference_seed,
        dataset.X_train,
        dataset.y_train,
    )
    reference_values = positive_class_shap_values(
        reference_model,
        dataset.X_probe,
    )
    reference_predictions = reference_model.predict(dataset.X_probe)
    reference_performance = predictive_metrics(
        reference_model,
        dataset.X_test,
        dataset.y_test,
    )

    benign_frames = []
    benign_run_rows = []
    for seed in config.model.benign_seeds:
        benign_model = train_reference_model(
            config,
            seed,
            dataset.X_train,
            dataset.y_train,
        )
        benign_values = positive_class_shap_values(
            benign_model,
            dataset.X_probe,
        )
        benign_predictions = benign_model.predict(dataset.X_probe)
        comparison = compare_explanations(
            config,
            reference_values,
            benign_values,
            reference_predictions != benign_predictions,
        )
        comparison.insert(0, "seed", seed)
        benign_frames.append(comparison)
        performance = predictive_metrics(
            benign_model,
            dataset.X_test,
            dataset.y_test,
        )
        benign_run_rows.append(
            {
                "seed": seed,
                **performance,
                "prediction_agreement": (
                    1.0 - comparison["prediction_changed"].mean()
                ),
                "mean_cosine_similarity": comparison[
                    "cosine_similarity"
                ].mean(),
                "mean_rank_correlation": comparison["rank_correlation"].mean(),
                "mean_top_k_overlap": comparison["top_k_overlap"].mean(),
                "mean_direction_flip_rate": comparison[
                    "direction_flip_rate"
                ].mean(),
            }
        )
    benign_samples = pd.concat(benign_frames, ignore_index=True)
    benign_runs = pd.DataFrame(benign_run_rows)

    candidate_model = train_candidate_model(
        config,
        dataset.X_train,
        dataset.y_train,
    )
    candidate_values = positive_class_shap_values(
        candidate_model,
        dataset.X_probe,
    )
    candidate_predictions = candidate_model.predict(dataset.X_probe)
    candidate_performance = predictive_metrics(
        candidate_model,
        dataset.X_test,
        dataset.y_test,
    )
    candidate_comparison = compare_explanations(
        config,
        reference_values,
        candidate_values,
        reference_predictions != candidate_predictions,
    )

    policy, sample_limits = calibrate_policy(
        config,
        benign_samples,
        candidate_comparison,
    )
    overall_status = "REVIEW" if policy["review_flag"].any() else "WITHIN_BASELINE"
    global_features = build_global_feature_comparison(
        dataset.X_probe.columns,
        reference_values,
        candidate_values,
    )

    benign_samples.to_csv(report_dir / "benign_per_sample.csv", index=False)
    benign_runs.to_csv(report_dir / "benign_run_summary.csv", index=False)
    candidate_comparison.to_csv(
        report_dir / "candidate_per_sample.csv",
        index=False,
    )
    policy.to_csv(report_dir / "policy_evaluation.csv", index=False)
    global_features.to_csv(
        report_dir / "global_feature_comparison.csv",
        index=False,
    )
    save_validation_figure(
        global_features,
        policy,
        figure_dir / "real_dataset_validation.png",
    )

    summary = {
        "experiment_id": config.experiment_id,
        "config_path": str(Path(config_path)),
        "dataset": "scikit-learn Breast Cancer Wisconsin Diagnostic",
        "samples": int(len(dataset.X_train) + len(dataset.X_test)),
        "features": int(dataset.X_train.shape[1]),
        "probe_samples": int(len(dataset.X_probe)),
        "positive_class": config.explainer.target_class,
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
        "flagged_metrics": policy.loc[
            policy["review_flag"],
            "metric",
        ].tolist(),
        "interpretation": (
            "A review flag indicates explanation changes beyond the provisional "
            "benign-retraining baseline; it is not proof of a harmful regression."
        ),
    }
    (report_dir / "summary.json").write_text(
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
    print(f"Flagged metrics: {', '.join(summary['flagged_metrics']) or 'none'}")
    print(f"Reports saved to: {report_dir}")
    print(f"Figure saved to: {figure_dir}")
    return summary
