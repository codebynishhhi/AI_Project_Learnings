
import pytest

from answer_evaluator import score_answer


def test_fully_correct_answer_scores_one():
    result = score_answer(
        fact_results=[True, True, True],
        answerable=True,
    )

    assert result == 1.0


def test_partially_correct_answer_scores_half():
    result = score_answer(
        fact_results=[True, False, True],
        answerable=True,
    )

    assert result == 0.5


def test_incorrect_answer_scores_zero():
    result = score_answer(
        fact_results=[False, False, False],
        answerable=True,
    )

    assert result == 0.0


def test_answer_with_one_required_fact_can_be_correct():
    result = score_answer(
        fact_results=[True],
        answerable=True,
    )

    assert result == 1.0


def test_answerable_question_requires_fact_results():
    with pytest.raises(ValueError):
        score_answer(
            fact_results=[],
            answerable=True,
        )


def test_abstention_on_unanswerable_question_scores_one():
    result = score_answer(
        fact_results=[],
        answerable=False,
        abstained=True,
    )

    assert result == 1.0


def test_answering_unanswerable_question_scores_zero():
    result = score_answer(
        fact_results=[],
        answerable=False,
        abstained=False,
    )

    assert result == 0.0


def test_partial_correctness_requires_boolean_judgments():
    with pytest.raises(ValueError):
        score_answer(
            fact_results=[True, "maybe"],
            answerable=True,
        )


def test_answerable_question_cannot_be_marked_abstained():
    with pytest.raises(ValueError):
        score_answer(
            fact_results=[True, False],
            answerable=True,
            abstained=True,
        )
