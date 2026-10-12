
from types import SimpleNamespace
from unittest.mock import Mock

import rag_chain


def make_client(answer: str) -> Mock:
    """Create a fake Groq client with a predefined response."""
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=answer)
            )
        ]
    )
    return client


def make_store(score: float = 0.8) -> Mock:
    """Create a fake vector store with one retrieved chunk."""
    store = Mock()
    store.search.return_value = [
        (
            "Q5: What is backpropagation used for?\n"
            "A5: It calculates the gradient of the error function.",
            score,
            {
                "source": "neural_networks_sample.pdf",
                "chunk_id": "Q5",
            },
        )
    ]
    return store


def test_relevant_question_returns_answer(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "generate_embedding",
        lambda question: [0.1, 0.2],
    )

    store = make_store(score=0.78)
    client = make_client(
        "Backpropagation calculates the gradient of the error function."
    )

    answer = rag_chain.answer_question(
        "What is backpropagation used for?",
        store,
        client,
    )

    assert "gradient" in answer.lower()
    client.chat.completions.create.assert_called_once()


def test_unrelated_question_abstains_without_calling_llm(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "generate_embedding",
        lambda question: [0.1, 0.2],
    )

    store = make_store(score=0.02)
    client = make_client("This response should never be used.")

    answer = rag_chain.answer_question(
        "What is the capital of France?",
        store,
        client,
    )

    assert "couldn't find relevant information" in answer.lower()
    client.chat.completions.create.assert_not_called()


def test_missing_detail_is_handled_by_answer_generation(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "generate_embedding",
        lambda question: [0.1, 0.2],
    )

    store = make_store(score=0.65)
    client = make_client(
        "I don't have enough information to provide that."
    )

    answer = rag_chain.answer_question(
        "What exact formula does the document give?",
        store,
        client,
    )

    assert "enough information" in answer.lower()

    # Confirm that the LLM received the question and retrieved context.
    prompt = client.chat.completions.create.call_args.kwargs[
        "messages"
    ][0]["content"]

    assert "What exact formula does the document give?" in prompt
    assert "gradient of the error function" in prompt
