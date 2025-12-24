import logging
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from result_evaluator.runtime.config import LLMConfig
from result_evaluator.runtime.llm import call_llm


class MockResponse(BaseModel):
    message: str


@pytest.fixture
def dummy_config() -> LLMConfig:
    return LLMConfig(
        api_key="test",
        model="test",
        endpoint="http://test",
        timeout=30,
        max_tokens=1000,
    )


def test_call_llm_logging_success(
    caplog: pytest.LogCaptureFixture, dummy_config: LLMConfig
):
    """Test successful call logs correct info."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_parsed = MockResponse(message="success")

    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    with patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client):
        with caplog.at_level(logging.DEBUG):
            call_llm("sys", "user", MockResponse, config=dummy_config)

            assert "LLM call completed" in caplog.text

            # Find the record with our message
            record = next(
                r for r in caplog.records if "LLM call completed" in r.message
            )
            assert hasattr(record, "llm_call")
            llm_call = record.llm_call
            assert llm_call["model"] == "test"
            assert llm_call["latency_ms"] >= 0
            assert llm_call["system_prompt_length"] == 3
            assert llm_call["user_prompt_length"] == 4
            assert llm_call["response_type"] == "MockResponse"

            assert dummy_config.api_key not in caplog.text


def test_call_llm_logging_error(
    caplog: pytest.LogCaptureFixture, dummy_config: LLMConfig
):
    """Test error call logs exception."""
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = Exception("Boom")

        with caplog.at_level(logging.ERROR):
            call_llm("sys", "user", MockResponse, config=dummy_config)

            assert "Unexpected error" in caplog.text
            # Verify exc_info was logged (at least one record has exc_info)
            assert any(record.exc_info for record in caplog.records)


def test_call_llm_logging_large_prompt(
    caplog: pytest.LogCaptureFixture, dummy_config: LLMConfig
):
    """Test warning for large prompts."""
    large_prompt = "x" * 100_001
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = MockResponse(message="ok")
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    with patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client):
        with caplog.at_level(logging.WARNING):
            call_llm("sys", large_prompt, MockResponse, config=dummy_config)

            assert "Large prompt detected" in caplog.text
            assert "100004" in caplog.text  # 3 chars 'sys' + 100001 chars
