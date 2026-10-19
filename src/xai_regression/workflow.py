"""Reusable components for the tabular explanation-regression workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from xai_regression.config import ExperimentConfig
from xai_regression.metrics import (
    absolute_rank_correlation,
    cosine_similarity,
    direction_flip_rate,
    top_k_overlap,
)


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_probe: pd.DataFrame
    y_probe: pd.Series


def validate_supported_data_model(config: ExperimentConfig) -> None:
    """Reject dataset and model choices not implemented by the prototype."""
    supported_targets = {
        "breast_cancer": "malignant",
        "wine_binary": "wine_class_1",
    }
    if config.dataset.name not in supported_targets:
        raise ValueError(
            "The current workflow supports dataset.name=breast_cancer or "
            "dataset.name=wine_binary."
        )
    supported_families = {"random_forest", "gradient_boosting"}
    if config.model.family not in supported_families:
        raise ValueError(
            "The current workflow supports model.family=random_forest or "
            "model.family=gradient_boosting."
        )
    expected_target = supported_targets[config.dataset.name]
    if config.explainer.target_class != expected_target:
        raise ValueError(
            f"The {config.dataset.name} workflow requires "
            f"target_class={expected_target}."
        )


def validate_supported_workflow(config: ExperimentConfig) -> None:
    """Validate the current Tree SHAP workflow configuration."""
    validate_supported_data_model(config)
    if config.explainer.method != "tree_shap":
        raise ValueError(
            "The current workflow supports explainer.method=tree_shap only."
        )


def load_dataset(config: ExperimentConfig) -> DatasetSplit:
    """Load and split the configured dataset, including a fixed probe subset."""
    validate_supported_data_model(config)
    if config.dataset.name == "breast_cancer":
        dataset = load_breast_cancer(as_frame=True)
        X = dataset.data.copy()

        # The source dataset uses malignant=0 and benign=1. Recode malignant as
        # the positive class so explanations describe cancer risk.
        y = (dataset.target == 0).astype(int)
    else:
        dataset = load_wine(as_frame=True)
        binary = dataset.target.isin([0, 1])
        X = dataset.data.loc[binary].reset_index(drop=True)
        # Use class 1 as the positive class and exclude class 2 to preserve the
        # binary-class assumptions used by Tree SHAP and the policy workflow.
        y = (dataset.target.loc[binary].reset_index(drop=True) == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.dataset.test_size,
        random_state=config.dataset.split_seed,
        stratify=y,
    )
    X_probe = X_test.iloc[: config.dataset.probe_size].copy()
    y_probe = y_test.iloc[: config.dataset.probe_size].copy()
    return DatasetSplit(X_train, X_test, y_train, y_test, X_probe, y_probe)


def train_configured_model(
    config: ExperimentConfig,
    *,
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int | None,
    min_samples_leaf: int,
) -> Any:
    """Train a supported tree model using comparable structural controls."""
    if config.model.family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config.model.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=config.model.class_weight,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    if config.model.family == "gradient_boosting":
        # ``None`` denotes the family default in shared scenario files.  The
        # sklearn default for gradient boosting trees is depth 3.
        effective_max_depth = 3 if max_depth is None else max_depth
        model = GradientBoostingClassifier(
            n_estimators=config.model.n_estimators,
            max_depth=effective_max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        sample_weight = None
        if config.model.class_weight == "balanced":
            sample_weight = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    raise ValueError(f"Unsupported model family: {config.model.family}")


def train_reference_model(
    config: ExperimentConfig,
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train one model with the accepted reference configuration."""
    return train_configured_model(
        config,
        seed=seed,
        X_train=X_train,
        y_train=y_train,
        max_depth=None,
        min_samples_leaf=1,
    )


def train_candidate_model(
    config: ExperimentConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train the controlled candidate model version."""
    return train_configured_model(
        config,
        seed=config.model.candidate_random_seed,
        X_train=X_train,
        y_train=y_train,
        max_depth=config.model.candidate_max_depth,
        min_samples_leaf=config.model.candidate_min_samples_leaf,
    )


def positive_class_shap_values(
    model: Any,
    X_probe: pd.DataFrame,
) -> np.ndarray:
    """Return a two-dimensional positive-class Tree SHAP matrix."""
    values = np.asarray(shap.TreeExplainer(model)(X_probe).values)

    if values.ndim == 3:
        if values.shape[2] != 2:
            raise RuntimeError(f"Expected binary-class SHAP values, got {values.shape}.")
        values = values[:, :, 1]
    elif values.ndim != 2:
        raise RuntimeError(f"Expected a 2D explanation array, got {values.shape}.")
    return values


def predictive_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calculate predictive performance for the configured positive class."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1_positive": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def compare_explanations(
    config: ExperimentConfig,
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    prediction_changes: np.ndarray,
) -> pd.DataFrame:
    """Compare matched attribution vectors for each fixed-probe sample."""
    rows = []
    for sample_id, (left, right) in enumerate(
        zip(reference_values, candidate_values)
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "prediction_changed": bool(prediction_changes[sample_id]),
                "cosine_similarity": cosine_similarity(left, right),
                "rank_correlation": absolute_rank_correlation(left, right),
                "top_k_overlap": top_k_overlap(
                    left,
                    right,
                    config.comparison.top_k,
                ),
                "direction_flip_rate": direction_flip_rate(left, right),
            }
        )
    return pd.DataFrame(rows)
