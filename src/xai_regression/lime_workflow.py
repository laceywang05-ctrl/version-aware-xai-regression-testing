"""Repeated-seed LIME stability experiment for a fixed model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xai_regression.config import ExperimentConfig
from xai_regression.workflow import (
    compare_explanations,
    load_dataset,
    train_reference_model,
    validate_supported_data_model,
)


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
