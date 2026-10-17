"""Repeated-seed LIME stability experiment for a fixed model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from xai_regression.config import ExperimentConfig
from xai_regression.runner import METRIC_DIRECTIONS, calibrate_policy
from xai_regression.workflow import (
    compare_explanations,
    load_dataset,
    predictive_metrics,
    train_candidate_model,
    train_reference_model,
    validate_supported_data_model,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_COLUMNS = (
    "cosine_similarity",
    "rank_correlation",
    "top_k_overlap",
    "direction_flip_rate",
)


def explanation_to_vector(
    explanation: Any,
    *,
    label: int,
    n_features: int,
) -> np.ndarray:
    """Map LIME's sparse local explanation back to feature-index order."""
    try:
        weighted_features = explanation.local_exp[label]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            f"LIME explanation does not contain requested class label {label}."
        ) from error

    vector = np.zeros(n_features, dtype=float)
    for feature_index, weight in weighted_features:
        if not 0 <= feature_index < n_features:
            raise ValueError(
                f"LIME returned invalid feature index {feature_index}."
            )
        vector[feature_index] = float(weight)
    return vector


def generate_lime_matrix(
    config: ExperimentConfig,
    model: Any,
    X_train: pd.DataFrame,
    X_probe: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Generate one fixed-order LIME attribution matrix for a random seed."""
    if config.lime is None:
        raise ValueError("The LIME experiment requires a lime configuration.")

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as error:
        raise RuntimeError(
            "LIME is not installed. Run: python -m pip install "
            "'lime>=0.2.0.1,<0.3'"
        ) from error

    explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=list(X_train.columns),
        class_names=["benign", "malignant"],
        mode="classification",
        discretize_continuous=config.lime.discretize_continuous,
        kernel_width=config.lime.kernel_width,
        random_state=seed,
    )

    def predict_proba(values: np.ndarray) -> np.ndarray:
        """Preserve training feature names for scikit-learn validation."""
        frame = pd.DataFrame(values, columns=X_train.columns)
        return model.predict_proba(frame)

    vectors = []
    for row in X_probe.to_numpy():
        explanation = explainer.explain_instance(
            row,
            predict_proba,
            labels=(1,),
            num_features=X_train.shape[1],
            num_samples=config.lime.num_samples,
        )
        vectors.append(
            explanation_to_vector(
                explanation,
                label=1,
                n_features=X_train.shape[1],
            )
        )
    return np.vstack(vectors)


def _save_distribution_figure(results: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric in zip(axes.flat, METRIC_COLUMNS):
        axis.hist(results[metric], bins=15, color="#2a78b8", alpha=0.85)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Score")
        axis.set_ylabel("Matched comparisons")
    fig.suptitle("LIME Stability Across Random Seeds (Fixed Model)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def apply_sample_policy_flags(
    frame: pd.DataFrame,
    sample_limits: dict[str, float],
) -> pd.DataFrame:
    """Add metric-level and combined advisory breach flags."""
    flagged = frame.copy()
    columns = []
    for metric, limit in sample_limits.items():
        direction = METRIC_DIRECTIONS[metric]
        column = f"{metric}_breach"
        flagged[column] = (
            flagged[metric] < limit
            if direction == "lower"
            else flagged[metric] > limit
        )
        columns.append(column)
    flagged["any_metric_breach"] = flagged[columns].any(axis=1)
    return flagged


def _save_cross_version_figure(
    benign: pd.DataFrame,
    candidate: pd.DataFrame,
    policy: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    positions = np.arange(len(METRIC_COLUMNS))
    benign_data = [benign[metric].to_numpy() for metric in METRIC_COLUMNS]
    candidate_data = [candidate[metric].to_numpy() for metric in METRIC_COLUMNS]
    axes[0].boxplot(
        benign_data,
        positions=positions - 0.17,
        widths=0.28,
        patch_artist=True,
        boxprops={"facecolor": "#6baed6"},
        medianprops={"color": "black"},
    )
    axes[0].boxplot(
        candidate_data,
        positions=positions + 0.17,
        widths=0.28,
        patch_artist=True,
        boxprops={"facecolor": "#fd8d3c"},
        medianprops={"color": "black"},
    )
    axes[0].set_xticks(
        positions,
        [metric.replace("_", "\n") for metric in METRIC_COLUMNS],
    )
    axes[0].set_title("Fixed-Model Variation vs Cross-Version Change")
    axes[0].set_ylabel("Metric value")
    axes[0].plot([], [], color="#6baed6", linewidth=8, label="fixed model")
    axes[0].plot([], [], color="#fd8d3c", linewidth=8, label="cross version")
    axes[0].legend()

    axes[1].bar(
        policy["metric"],
        policy["candidate_breach_rate"],
        color="#fd8d3c",
        label="cross-version breach rate",
    )
    axes[1].scatter(
        policy["metric"],
        policy["run_level_limit_95_percent"],
        color="#2b8cbe",
        marker="_",
        s=500,
        linewidths=3,
        label="fixed-model run limit",
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Fraction of matched samples")
    axes[1].set_title("Advisory Policy Evaluation")
    axes[1].legend()
    figure.suptitle("LIME Cross-Version Explanation Comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_lime_stability(
    config: ExperimentConfig,
    *,
    config_path: str | Path,
    project_root: Path,
) -> dict[str, Any]:
    """Compare repeated LIME runs while holding the model and data fixed."""
    validate_supported_data_model(config)
    if config.explainer.method != "lime" or config.lime is None:
        raise ValueError("This command requires explainer.method=lime.")

    split = load_dataset(config)
    X_probe = split.X_probe.iloc[: config.lime.probe_size].copy()
    model = train_reference_model(
        config,
        config.model.reference_seed,
        split.X_train,
        split.y_train,
    )
    baseline = generate_lime_matrix(
        config,
        model,
        split.X_train,
        X_probe,
        seed=config.lime.baseline_seed,
    )

    per_sample_frames = []
    run_rows = []
    unchanged_predictions = np.zeros(len(X_probe), dtype=bool)
    for seed in config.lime.repeated_seeds:
        candidate = generate_lime_matrix(
            config,
            model,
            split.X_train,
            X_probe,
            seed=seed,
        )
        comparison = compare_explanations(
            config,
            baseline,
            candidate,
            unchanged_predictions,
        )
        comparison.insert(0, "seed", seed)
        per_sample_frames.append(comparison)
        row = {"seed": seed}
        row.update(
            {
                f"mean_{metric}": float(comparison[metric].mean())
                for metric in METRIC_COLUMNS
            }
        )
        run_rows.append(row)

    per_sample = pd.concat(per_sample_frames, ignore_index=True)
    run_summary = pd.DataFrame(run_rows)

    output_dir = project_root / config.lime.output_dir
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = reports_dir / "per_sample_lime_stability.csv"
    run_summary_path = reports_dir / "run_summary.csv"
    summary_path = reports_dir / "summary.json"
    figure_path = figures_dir / "lime_stability_distributions.png"
    per_sample.to_csv(per_sample_path, index=False)
    run_summary.to_csv(run_summary_path, index=False)
    _save_distribution_figure(per_sample, figure_path)

    summary: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "config_path": str(Path(config_path)),
        "interpretation": (
            "Same model and fixed probe data; observed differences estimate "
            "LIME explainer variance and are not model drift."
        ),
        "model_seed": config.model.reference_seed,
        "baseline_lime_seed": config.lime.baseline_seed,
        "repeated_lime_seeds": list(config.lime.repeated_seeds),
        "probe_samples": len(X_probe),
        "num_samples_per_explanation": config.lime.num_samples,
        "kernel_width": config.lime.kernel_width,
        "discretize_continuous": config.lime.discretize_continuous,
        "matched_comparisons": len(per_sample),
        "overall_means": {
            metric: float(per_sample[metric].mean())
            for metric in METRIC_COLUMNS
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("LIME fixed-model stability results")
    print("----------------------------------")
    print(f"Probe samples: {len(X_probe)}")
    print(f"Repeated seeds: {len(config.lime.repeated_seeds)}")
    print(f"Matched comparisons: {len(per_sample)}")
    for metric, value in summary["overall_means"].items():
        print(f"Mean {metric}: {value:.6f}")
    print(f"Reports saved to: {reports_dir}")
    print(f"Figure saved to: {figure_path}")
    return summary


def run_lime_cross_version(
    config: ExperimentConfig,
    *,
    config_path: str | Path,
    project_root: Path,
) -> dict[str, Any]:
    """Compare reference and candidate LIME explanations against noise."""
    validate_supported_data_model(config)
    if config.explainer.method != "lime" or config.lime is None:
        raise ValueError("This command requires explainer.method=lime.")

    split = load_dataset(config)
    X_probe = split.X_probe.iloc[: config.lime.probe_size].copy()
    reference_model = train_reference_model(
        config,
        config.model.reference_seed,
        split.X_train,
        split.y_train,
    )
    candidate_model = train_candidate_model(
        config,
        split.X_train,
        split.y_train,
    )
    reference_predictions = reference_model.predict(X_probe)
    candidate_predictions = candidate_model.predict(X_probe)
    prediction_changes = reference_predictions != candidate_predictions

    baseline_reference = generate_lime_matrix(
        config,
        reference_model,
        split.X_train,
        X_probe,
        seed=config.lime.baseline_seed,
    )
    benign_frames = []
    candidate_frames = []
    paired_candidate_frames = []
    for seed in config.lime.repeated_seeds:
        reference_values = generate_lime_matrix(
            config,
            reference_model,
            split.X_train,
            X_probe,
            seed=seed,
        )
        benign = compare_explanations(
            config,
            baseline_reference,
            reference_values,
            np.zeros(len(X_probe), dtype=bool),
        )
        benign.insert(0, "seed", seed)
        benign_frames.append(benign)

        candidate_values = generate_lime_matrix(
            config,
            candidate_model,
            split.X_train,
            X_probe,
            seed=seed,
        )
        candidate = compare_explanations(
            config,
            baseline_reference,
            candidate_values,
            prediction_changes,
        )
        candidate.insert(0, "seed", seed)
        candidate_frames.append(candidate)

        paired_candidate = compare_explanations(
            config,
            reference_values,
            candidate_values,
            prediction_changes,
        )
        paired_candidate.insert(0, "seed", seed)
        paired_candidate_frames.append(paired_candidate)

    benign_samples = pd.concat(benign_frames, ignore_index=True)
    candidate_samples = pd.concat(candidate_frames, ignore_index=True)
    paired_candidate_samples = pd.concat(
        paired_candidate_frames,
        ignore_index=True,
    )
    policy, sample_limits = calibrate_policy(
        config,
        benign_samples,
        candidate_samples,
    )
    candidate_flagged = apply_sample_policy_flags(
        candidate_samples,
        sample_limits,
    )
    overall_status = (
        "REVIEW" if policy["review_flag"].any() else "WITHIN_BASELINE"
    )

    output_dir = project_root / config.lime.output_dir
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    for directory in (reports_dir, figures_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    probe = X_probe.reset_index(names="source_row")
    probe.insert(0, "sample_id", range(len(probe)))
    probe["target_malignant"] = split.y_probe.iloc[: len(X_probe)].to_numpy()
    probe.to_csv(data_dir / "fixed_probe.csv", index=False)
    benign_samples.to_csv(
        reports_dir / "fixed_model_baseline.csv",
        index=False,
    )
    candidate_flagged.to_csv(
        reports_dir / "cross_version_per_sample.csv",
        index=False,
    )
    paired_candidate_samples.to_csv(
        reports_dir / "paired_seed_cross_version.csv",
        index=False,
    )
    policy.to_csv(reports_dir / "policy_evaluation.csv", index=False)
    figure_path = figures_dir / "lime_cross_version_comparison.png"
    _save_cross_version_figure(
        benign_samples,
        candidate_samples,
        policy,
        figure_path,
    )

    summary: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "config_path": str(Path(config_path)),
        "status": overall_status,
        "interpretation": (
            "Policy comparison uses the same independent-seed structure for "
            "fixed-model and cross-version runs. The paired-seed result "
            "controls random perturbations and is descriptive. REVIEW is not "
            "proof of a defective model."
        ),
        "probe_samples": len(X_probe),
        "repeated_seeds": len(config.lime.repeated_seeds),
        "matched_cross_version_comparisons": len(candidate_samples),
        "probe_prediction_agreement": float(1.0 - prediction_changes.mean()),
        "reference_performance": predictive_metrics(
            reference_model,
            split.X_test,
            split.y_test,
        ),
        "candidate_performance": predictive_metrics(
            candidate_model,
            split.X_test,
            split.y_test,
        ),
        "fixed_model_means": {
            metric: float(benign_samples[metric].mean())
            for metric in METRIC_COLUMNS
        },
        "noise_matched_cross_version_means": {
            metric: float(candidate_samples[metric].mean())
            for metric in METRIC_COLUMNS
        },
        "paired_seed_cross_version_means": {
            metric: float(paired_candidate_samples[metric].mean())
            for metric in METRIC_COLUMNS
        },
        "sample_limits": sample_limits,
        "flagged_metrics": policy.loc[policy["review_flag"], "metric"].tolist(),
        "any_metric_breach_rate": float(
            candidate_flagged["any_metric_breach"].mean()
        ),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("LIME cross-version comparison")
    print("-----------------------------")
    print(f"Status: {overall_status}")
    print(
        "Probe prediction agreement: "
        f"{summary['probe_prediction_agreement']:.3f}"
    )
    for metric in METRIC_COLUMNS:
        baseline = summary["fixed_model_means"][metric]
        cross_version = summary["noise_matched_cross_version_means"][metric]
        paired = summary["paired_seed_cross_version_means"][metric]
        print(
            f"{metric}: baseline={baseline:.6f}, "
            f"noise-matched cross={cross_version:.6f}, "
            f"paired cross={paired:.6f}"
        )
    print(f"Flagged metrics: {summary['flagged_metrics']}")
    print(f"Reports saved to: {reports_dir}")
    print(f"Figure saved to: {figure_path}")
    return summary
