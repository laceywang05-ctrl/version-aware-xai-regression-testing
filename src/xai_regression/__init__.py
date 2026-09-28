"""Utilities for version-aware explanation regression testing."""

from .metrics import (
    cosine_similarity,
    direction_flip_rate,
    spearman_rank_correlation,
    top_k_overlap,
)

__all__ = [
    "cosine_similarity",
    "direction_flip_rate",
    "spearman_rank_correlation",
    "top_k_overlap",
]

