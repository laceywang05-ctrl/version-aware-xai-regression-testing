"""Metrics for comparing feature-attribution vectors across model versions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _pair(baseline: ArrayLike, candidate: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(baseline, dtype=float).reshape(-1)
    right = np.asarray(candidate, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("Attribution vectors must have the same shape.")
    if left.size == 0:
        raise ValueError("Attribution vectors must not be empty.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Attribution vectors must contain only finite values.")
    return left, right


def cosine_similarity(baseline: ArrayLike, candidate: ArrayLike) -> float:
    """Return cosine similarity in [-1, 1]. Two zero vectors are treated as equal."""
    left, right = _pair(baseline, candidate)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_rank_correlation(baseline: ArrayLike, candidate: ArrayLike) -> float:
    """Compare feature rankings using average ranks for ties."""
    left, right = _pair(baseline, candidate)
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.all(left_ranks == left_ranks[0]) and np.all(right_ranks == right_ranks[0]):
        return 1.0
    if np.all(left_ranks == left_ranks[0]) or np.all(right_ranks == right_ranks[0]):
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def top_k_overlap(baseline: ArrayLike, candidate: ArrayLike, k: int) -> float:
    """Return the proportion of shared top-k features by absolute attribution."""
    left, right = _pair(baseline, candidate)
    if not 1 <= k <= left.size:
        raise ValueError("k must be between 1 and the number of features.")
    left_top = set(np.argsort(np.abs(left), kind="mergesort")[-k:])
    right_top = set(np.argsort(np.abs(right), kind="mergesort")[-k:])
    return len(left_top & right_top) / k


def direction_flip_rate(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> float:
    """Return the share of active features whose attribution sign changes."""
    left, right = _pair(baseline, candidate)
    active = (np.abs(left) > tolerance) | (np.abs(right) > tolerance)
    if not np.any(active):
        return 0.0
    flipped = np.sign(left[active]) != np.sign(right[active])
    return float(np.mean(flipped))

