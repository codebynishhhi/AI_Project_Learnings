
from numbers import Real
import math


def is_rejected_by_threshold(
    top_score: float | None,
    threshold: float,
) -> bool:
    """Return whether a query is rejected by the relevance threshold."""
    if top_score is None:
        return True

    if (
        isinstance(top_score, bool)
        or not isinstance(top_score, Real)
        or not math.isfinite(top_score)
    ):
        raise ValueError("top_score must be a finite number or None")

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    return top_score < threshold


def calculate_rejection_rate(
    rejected_count: int,
    total_count: int,
) -> float:
    """Calculate the fraction of unanswerable queries rejected."""
    if total_count < 0 or rejected_count < 0:
        raise ValueError("Counts cannot be negative")

    if rejected_count > total_count:
        raise ValueError("Rejected count cannot exceed total count")

    if total_count == 0:
        return 0.0

    return rejected_count / total_count
