
import pytest

from retrieval_metrics import (
    calculate_rejection_rate,
    is_rejected_by_threshold,
)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.20, True),
        (0.2999, True),
        (0.30, False),
        (0.65, False),
        (None, True),
    ],
)
def test_threshold_rejection(score, expected):
    assert is_rejected_by_threshold(score, 0.30) is expected


def test_rejection_rate():
    assert calculate_rejection_rate(2, 3) == pytest.approx(2 / 3)


def test_rejection_rate_with_no_queries():
    assert calculate_rejection_rate(0, 0) == 0.0


@pytest.mark.parametrize(
    ("rejected", "total"),
    [
        (-1, 3),
        (4, 3),
    ],
)
def test_rejection_rate_rejects_invalid_counts(rejected, total):
    with pytest.raises(ValueError):
        calculate_rejection_rate(rejected, total)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), True])
def test_threshold_rejection_rejects_invalid_scores(score):
    with pytest.raises(ValueError):
        is_rejected_by_threshold(score, 0.30)
