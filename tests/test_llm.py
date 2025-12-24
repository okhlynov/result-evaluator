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
from result_evaluator.runtime.llm import Result, call_llm


def test_call_llm_validation_system_prompt_empty():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        call_llm("", "user prompt", str)


def test_call_llm_validation_system_prompt_none():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        call_llm(None, "user prompt", str)


def test_call_llm_validation_system_prompt_whitespace():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        call_llm("   ", "user prompt", str)


def test_call_llm_validation_user_prompt_empty():
    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        call_llm("system prompt", "", str)


def test_call_llm_validation_user_prompt_none():
    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        call_llm("system prompt", None, str)


def test_call_llm_validation_user_prompt_whitespace():
    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        call_llm("system prompt", "   ", str)


class MockResponse(BaseModel):
    message: str


DUMMY_CONFIG = LLMConfig(
    api_key="test", model="test", endpoint="http://test", timeout=30, max_tokens=1000
)


def test_call_llm_openai_integration():
    """Test that OpenAI client is called correctly."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_parsed = MockResponse(message="success")

    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    with patch(
        "result_evaluator.runtime.llm.OpenAI", return_value=mock_client
    ) as mock_openai_cls:
        config = LLMConfig(
            api_key="test_key",
            model="test_model",
            endpoint="http://test",
            timeout=30,
            max_tokens=None,
        )
        result = call_llm("sys", "user", MockResponse, config=config)

        assert result.success
        assert result.value == mock_parsed

        mock_openai_cls.assert_called_once_with(
            api_key="test_key", base_url="http://test", timeout=30.0
        )

        mock_client.beta.chat.completions.parse.assert_called_once_with(
            model="test_model",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
            ],
            response_format=MockResponse,
            max_tokens=None,
        )


def test_call_llm_loads_config_if_none(monkeypatch: pytest.MonkeyPatch):
    """Test that config is loaded if None is passed."""
    # We mock load_llm_config
    mock_config = LLMConfig(
        api_key="env_key",
        model="env_model",
        endpoint=None,
        timeout=60,
        max_tokens=None,
    )
    monkeypatch.setattr(
        "result_evaluator.runtime.llm.load_llm_config", lambda: mock_config
    )

    # We also need to mock OpenAI to avoid actual call
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        # Setup mock return
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.return_value.choices[
            0
        ].message.parsed = "parsed_value"

        call_llm("system", "user", str, config=None)

        mock_openai_cls.assert_called_once()
        # check if called with config from env
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["api_key"] == "env_key"


def test_call_llm_authentication_error():
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = AuthenticationError(
            message="Invalid API key", response=MagicMock(), body={}
        )

        result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

        assert not result.success
        assert result.value is None
        assert result.error_type == "connection"
        assert "Authentication failed" in str(result.error)


def test_call_llm_api_connection_error():
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = APIConnectionError(
            message="Connection error", request=MagicMock()
        )

        result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

        assert not result.success
        assert result.error_type == "connection"
        assert "Connection failed" in str(result.error)


def test_call_llm_api_timeout_error():
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = APITimeoutError(
            request=MagicMock()
        )

        # We need a config to check timeout in error message
        config = LLMConfig(api_key="k", model="m", timeout=123)
        result = call_llm("sys", "user", MockResponse, config=config)

        assert not result.success
        assert result.error_type == "connection"
        assert "Request timeout after 123" in str(result.error)


def test_call_llm_rate_limit_error():
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(), body={}
        )

        result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

        assert not result.success
        assert result.error_type == "connection"
        assert "Rate limit exceeded" in str(result.error)


def test_call_llm_not_found_error():
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = NotFoundError(
            message="Not found", response=MagicMock(), body={}
        )

        config = LLMConfig(api_key="k", model="missing-model")
        result = call_llm("sys", "user", MockResponse, config=config)

        assert not result.success
        assert result.error_type == "connection"
        assert "Model 'missing-model' not found" in str(result.error)


def test_call_llm_api_error():
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = APIError(
            message="Some API error", request=MagicMock(), body={}
        )

        result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

        assert not result.success
        assert result.error_type == "response"
        assert "API error: Some API error" in str(result.error)


def test_call_llm_validation_error():
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        # Simulating Pydantic validation error raised by parse()
        # This usually happens when the model output doesn't match the schema
        mock_instance.beta.chat.completions.parse.side_effect = (
            ValidationError.from_exception_data("Validation error", line_errors=[])
        )

        result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

        assert not result.success
        assert result.error_type == "validation"
        assert "Response doesn't match schema" in str(result.error)


def test_call_llm_generic_exception():
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = Exception("Boom")

        result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

        assert not result.success
        assert result.error_type == "response"
        assert "Unexpected error: Boom" in str(result.error)


def test_call_llm_refusal():
    """Test handling of refusal (parsed is None)."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = None  # Refusal
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    with patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client):
        result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

        assert not result.success
        assert result.error_type == "response"
        assert "Model refused" in result.error


def test_result_ok():
    res = Result.ok("data")
    assert res.success is True
    assert res.value == "data"
    assert res.error is None
    assert res.error_type is None


def test_result_fail():
    res = Result.fail("connection", "timeout")
    assert res.success is False
    assert res.value is None
    assert res.error == "timeout"
    assert res.error_type == "connection"
