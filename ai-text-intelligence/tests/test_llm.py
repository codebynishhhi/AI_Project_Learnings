from unittest.mock import patch

from schemas import SummaryResponse
from llm import ask_llm_structured


def test_llm_returns_valid_schema():

    with patch("llm.client.chat.completions.create") as mock_create:

        mock_create.return_value.choices[0].message.content = """
        {
            "summary": "AI is transforming software development.",
            "key_points": [
                "AI can automate repetitive tasks",
                "LLMs can generate code",
                "AI requires careful evaluation"
            ]
        }
        """

        messages = [
            {
                "role": "user",
                "content": "Summarize AI."
            }
        ]

        response = ask_llm_structured(
            messages=messages,
            response_schema=SummaryResponse,
            schema_name="summary_test_schema",
            error_message="Test failed"
        )

        assert isinstance(response, SummaryResponse)
        assert response.summary == "AI is transforming software development."
        assert len(response.key_points) == 3
        assert response.key_points[0] == "AI can automate repetitive tasks"
        mock_create.assert_called_once_with(
            messages=messages,
            model="openai/gpt-oss-20b",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "summary_test_schema",
                    "schema": SummaryResponse.model_json_schema(),
                    "strict": True
                }
            }
)


def test_llm_returns_none_for_invalid_json():

    with patch("llm.client.chat.completions.create") as mock_create:

        mock_create.return_value.choices[0].message.content = """
        {
            "summary": "AI is transforming software development.",
            "key_points": [
        """

        response = ask_llm_structured(
            messages=[
                {
                    "role": "user",
                    "content": "Summarize AI."
                }
            ],
            response_schema=SummaryResponse,
            schema_name="summary_test_schema",
            error_message="Test failed"
        )

        assert response is None


def test_llm_returns_none_for_invalid_schema():

    with patch("llm.client.chat.completions.create") as mock_create:

        mock_create.return_value.choices[0].message.content = """
        {
            "summary": "AI is transforming software development.",
            "key_points": "This should be a list"
        }
        """

        response = ask_llm_structured(
            messages=[
                {
                    "role": "user",
                    "content": "Summarize AI."
                }
            ],
            response_schema=SummaryResponse,
            schema_name="summary_test_schema",
            error_message="Test failed"
        )
        assert response is None