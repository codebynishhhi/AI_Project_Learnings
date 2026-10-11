import pytest
from schema import SummaryResponse, SentimentResponse
from pydantic import ValidationError

# ==================================== TESTING DATA VALIDATION =========================================

# the core idea of unit testing Pydantic schemas now:

# Valid input → object is created
# Invalid input → ValidationError
# Assertions → verify the returned values
# pytest.raises() → verify expected failures

# ================================== Summarizer schema test cases ==================================
def test_summary_response_valid():
    response = SummaryResponse(
        summary="AI is transforming software development.",
        # summary=123,
        key_points=[
            "AI can automate repetitive tasks",
            "LLMs can generate code",
            "AI requires careful evaluation"
        ]
    )

    assert response.summary == "AI is transforming software development."
    assert len(response.key_points) == 3

def test_invalid_key_points():
    with pytest.raises(ValidationError):
        SummaryResponse(
            summary="AI is transforming software development.",
            key_points="not a list"
        )

def test_invalid_summary():
    with pytest.raises(ValidationError):
        SummaryResponse(
            summary=123, 
            key_points=[
                "one", 
                "two"
            ]
        )

# ================================== Sentiment schema test cases ==================================
 
def test_sentiment_response_valid():
    response = SentimentResponse(
        sentiment="Positive",
        confidence_score=0.92,
        explanation="The text expresses strong satisfaction."
    )

    assert response.sentiment == "Positive"
    assert response.confidence_score == 0.92
    assert response.explanation == "The text expresses strong satisfaction."

def test_invalid_sentiment():
    with pytest.raises(ValidationError):
        SentimentResponse(
        sentiment=123,
        confidence_score=0.92,
        explanation="The text expresses strong satisfaction."
        )

def test_invalid_confidence_score():
    with pytest.raises(ValidationError):
        SentimentResponse(
        sentiment=123,
        confidence_score='0.92',
        explanation="The text expresses strong satisfaction."
        )
