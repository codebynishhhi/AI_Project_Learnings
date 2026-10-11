from unittest.mock import patch

from app import summarize_text
from schemas import SummaryResponse

def test_summarize_text():

    fake_response = SummaryResponse(
        summary="AI is transforming software development.",
        key_points=[
            "AI can automate repetitive tasks",
            "LLMs can generate code",
            "AI requires careful evaluation"
        ]
    )

    with patch("app.ask_llm_structured") as mock_llm:

        mock_llm.return_value = fake_response

        response = summarize_text("AI is transforming software development.")

        assert response == fake_response