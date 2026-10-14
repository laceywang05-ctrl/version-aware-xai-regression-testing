"""Reusable components for the tabular explanation-regression workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

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


def validate_supported_workflow(config: ExperimentConfig) -> None:
    """Reject configuration choices not implemented by the current prototype."""
    if config.dataset.name != "breast_cancer":
        raise ValueError(
            "The current workflow supports dataset.name=breast_cancer only."
        )
    if config.model.family != "random_forest":
        raise ValueError(
            "The current workflow supports model.family=random_forest only."
        )
    if config.explainer.method != "tree_shap":
        raise ValueError(
            "The current workflow supports explainer.method=tree_shap only."
        )
    if config.explainer.target_class != "malignant":
        raise ValueError(
            "The breast-cancer workflow currently explains target_class=malignant."
        )


def load_dataset(config: ExperimentConfig) -> DatasetSplit:
    """Load and split the configured dataset, including a fixed probe subset."""
    validate_supported_workflow(config)
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()

    # The source dataset uses malignant=0 and benign=1. Recode malignant as the
    # positive class so positive-class explanations describe cancer risk.
    y = (dataset.target == 0).astype(int)

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


def train_reference_model(
    config: ExperimentConfig,
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train one model with the accepted reference configuration."""
    model = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        class_weight=config.model.class_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_candidate_model(
    config: ExperimentConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train the controlled candidate model version."""
    model = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.candidate_max_depth,
        min_samples_leaf=config.model.candidate_min_samples_leaf,
        class_weight=config.model.class_weight,
        random_state=config.model.candidate_random_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def positive_class_shap_values(
    model: RandomForestClassifier,
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
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calculate predictive performance for the positive malignant class."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1_malignant": float(f1_score(y_test, predictions)),
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
