"""Create a compact comparison record for one explanation pair."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from numpy.typing import ArrayLike

from .metrics import (
    cosine_similarity,
    direction_flip_rate,
    spearman_rank_correlation,
    top_k_overlap,
)


@dataclass(frozen=True)
class ComparisonReport:
    baseline_version: str
    candidate_version: str
    cosine_similarity: float
    rank_correlation: float
    top_k_overlap: float
    direction_flip_rate: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


def compare_attributions(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    baseline_version: str,
    candidate_version: str,
    top_k: int = 5,
) -> ComparisonReport:
    return ComparisonReport(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        cosine_similarity=cosine_similarity(baseline, candidate),
        rank_correlation=spearman_rank_correlation(baseline, candidate),
        top_k_overlap=top_k_overlap(baseline, candidate, top_k),
        direction_flip_rate=direction_flip_rate(baseline, candidate),
    )

