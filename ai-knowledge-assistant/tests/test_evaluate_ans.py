from unittest.mock import Mock

from evaluate_answers import run_answer_evaluation

def test_runs_each_evaluation_question_through_rag():
    dataset = [
        {
            "id": "q1",
            "question": "What is an artificial neural network?",
            "reference_answer": "A computing system.",
            "required_facts": ["It is a computing system."],
            "answerable": True,
        },
        {
            "id": "q2",
            "question": "What exact formula is in the document?",
            "reference_answer": None,
            "required_facts": [],
            "answerable": False,
        },
    ]

    store = Mock()
    client = Mock()

    generated_answers = {
        "What is an artificial neural network?": "A computing system.",
        "What exact formula is in the document?": (
            "I don't have enough information to provide that."
        ),
    }

    def fake_answer_question(question, store, client):
        return generated_answers[question]

    results = run_answer_evaluation(
        dataset,
        store,
        client,
        answer_function=fake_answer_question,
    )

    assert len(results) == 2
    assert results[0]["id"] == "q1"
    assert results[0]["generated_answer"] == "A computing system."
    assert results[1]["id"] == "q2"
    assert results[1]["answerable"] is False
    assert "enough information" in results[1]["generated_answer"]


def test_preserves_reference_answers_and_required_facts():
    dataset = [
        {
            "id": "q1",
            "question": "What is deep learning?",
            "reference_answer": "A subset of machine learning.",
            "required_facts": ["It is a subset of machine learning."],
            "answerable": True,
        }
    ]

    results = run_answer_evaluation(
        dataset,
        Mock(),
        Mock(),
        answer_function=lambda question, store, client: "A subset of ML.",
    )

    assert results[0]["reference_answer"] == (
        "A subset of machine learning."
    )
    assert results[0]["required_facts"] == [
        "It is a subset of machine learning."
    ]


def test_empty_dataset_returns_no_results():
    results = run_answer_evaluation(
        [],
        Mock(),
        Mock(),
        answer_function=Mock(),
    )

    assert results == []