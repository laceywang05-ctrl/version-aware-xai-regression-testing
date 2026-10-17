"""Metrics for comparing feature-attribution explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity, with explicit handling for zero vectors."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)

    if denominator == 0:
        return 1.0 if np.allclose(left, right) else 0.0

    return float(np.dot(left, right) / denominator)


def absolute_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman correlation between absolute attribution rankings."""
    left_ranks = pd.Series(np.abs(left)).rank(method="average")
    right_ranks = pd.Series(np.abs(right)).rank(method="average")
    correlation = left_ranks.corr(right_ranks)

    if pd.isna(correlation):
        return 1.0 if np.allclose(left, right) else 0.0

    return float(correlation)


def top_k_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Return the proportion of shared features in two absolute top-k sets."""
    left = np.asarray(left)
    right = np.asarray(right)

    if left.shape != right.shape:
        raise ValueError("Attribution vectors must have the same shape.")
    if left.ndim != 1:
        raise ValueError("Attribution vectors must be one-dimensional.")
    if not 1 <= k <= left.size:
        raise ValueError("k must be between 1 and the number of features.")

    left_top = set(np.argsort(np.abs(left))[-k:])
    right_top = set(np.argsort(np.abs(right))[-k:])
    return len(left_top.intersection(right_top)) / k


def direction_flip_rate(
    left: np.ndarray,
    right: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Return the sign-change rate among non-negligible attributions."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    active = (np.abs(left) > epsilon) | (np.abs(right) > epsilon)

    if not active.any():
        return 0.0

    return float((np.sign(left[active]) != np.sign(right[active])).mean())

