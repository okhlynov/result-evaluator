from unittest.mock import MagicMock, patch

import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from result_evaluator.runtime.config import LLMConfig
from result_evaluator.runtime.llm import call_llm


class MockResponse(BaseModel):
    message: str


@pytest.fixture
def mock_config() -> LLMConfig:
    return LLMConfig(
        api_key="test_key",
        model="test_model",
        endpoint="http://test",
        timeout=30,
        max_tokens=None,
    )


def test_call_llm_authentication_error(mock_config: LLMConfig):
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai:
        # Simulate AuthenticationError
        mock_openai.return_value.beta.chat.completions.parse.side_effect = (
            AuthenticationError(message="Auth failed", response=MagicMock(), body={})
        )

        result = call_llm("sys", "user", MockResponse, config=mock_config)

        assert result.success is False
        assert result.value is None
        assert result.error_type == "connection"
        assert result.error == "Authentication failed: check JUDGE_LLM_API_KEY"


def test_call_llm_api_connection_error(mock_config: LLMConfig):
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai:
        mock_openai.return_value.beta.chat.completions.parse.side_effect = (
            APIConnectionError(message="Connection failed", request=MagicMock())
        )

        result = call_llm("sys", "user", MockResponse, config=mock_config)

        assert result.success is False
        assert result.value is None
        assert result.error_type == "connection"
        assert result.error == "Connection failed: check network/endpoint"


def test_call_llm_api_timeout_error(mock_config: LLMConfig):
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai:
        mock_openai.return_value.beta.chat.completions.parse.side_effect = (
            APITimeoutError(request=MagicMock())
        )

        result = call_llm("sys", "user", MockResponse, config=mock_config)

        assert result.success is False
        assert result.value is None
        assert result.error_type == "connection"
        assert result.error == f"Request timeout after {mock_config.timeout}s"


def test_call_llm_rate_limit_error(mock_config: LLMConfig):
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai:
        mock_openai.return_value.beta.chat.completions.parse.side_effect = (
            RateLimitError(message="Rate limit", response=MagicMock(), body={})
        )

        result = call_llm("sys", "user", MockResponse, config=mock_config)

        assert result.success is False
        assert result.value is None
        assert result.error_type == "connection"
        assert result.error == "Rate limit exceeded: wait before retrying"


def test_call_llm_not_found_error(mock_config: LLMConfig):
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai:
        mock_openai.return_value.beta.chat.completions.parse.side_effect = (
            NotFoundError(message="Not found", response=MagicMock(), body={})
        )

        result = call_llm("sys", "user", MockResponse, config=mock_config)

        assert result.success is False
        assert result.value is None
        assert result.error_type == "connection"
        assert result.error == f"Model '{mock_config.model}' not found"


def test_call_llm_api_error(mock_config: LLMConfig):
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai:
        mock_openai.return_value.beta.chat.completions.parse.side_effect = APIError(
            message="Generic API error", request=MagicMock(), body={}
        )

        result = call_llm("sys", "user", MockResponse, config=mock_config)

        assert result.success is False
        assert result.value is None
        assert result.error_type == "response"
        assert "API error: Generic API error" in str(result.error)


def test_call_llm_validation_error(mock_config: LLMConfig):
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai:
        # Simulate Pydantic ValidationError
        # This is harder to simulate directly from parse() unless parse calls Pydantic and it fails.
        # But here we are patching parse() so we can just raise ValidationError directly.

        # We need a construct a ValidationError properly or just mock it enough?
        # Pydantic ValidationError requires a list of errors and the model.
        # It's easier to just raise it.
        mock_openai.return_value.beta.chat.completions.parse.side_effect = (
            ValidationError.from_exception_data("title", line_errors=[])
        )

        result = call_llm("sys", "user", MockResponse, config=mock_config)

        assert result.success is False
        assert result.value is None
        assert result.error_type == "validation"
        assert "Response doesn't match schema" in str(result.error)


def test_call_llm_generic_exception(mock_config: LLMConfig):
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai:
        mock_openai.return_value.beta.chat.completions.parse.side_effect = Exception(
            "Boom!"
        )

        result = call_llm("sys", "user", MockResponse, config=mock_config)

        assert result.success is False
        assert result.value is None
        assert result.error_type == "response"
        assert "Unexpected error: Boom!" in str(result.error)
