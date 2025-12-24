from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from result_evaluator.runtime.config import LLMConfig
from result_evaluator.runtime.llm import call_llm


class MockLLMResponse(BaseModel):
    """Mock Pydantic model for LLM response."""

    answer: str
    confidence: float


def test_call_llm_success_returns_result_with_success_true() -> None:
    """Test successful call returns Result with success=True."""
    mock_response = MockLLMResponse(answer="The answer is 42", confidence=0.95)

    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = mock_response
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        config = LLMConfig(
            api_key="test", model="gpt-4o", endpoint=None, timeout=60, max_tokens=None
        )
        result = call_llm(
            "System prompt", "User prompt", MockLLMResponse, config=config
        )

        assert result.success is True
        assert result.error is None
        assert result.error_type is None


def test_call_llm_success_parsed_object_in_value() -> None:
    """Test parsed Pydantic object is in result.value."""
    mock_response = MockLLMResponse(answer="The answer is 42", confidence=0.95)

    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = mock_response
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        config = LLMConfig(
            api_key="test", model="gpt-4o", endpoint=None, timeout=60, max_tokens=None
        )
        result = call_llm(
            "System prompt", "User prompt", MockLLMResponse, config=config
        )

        assert isinstance(result.value, MockLLMResponse)
        assert result.value.answer == "The answer is 42"
        assert result.value.confidence == 0.95


def test_call_llm_with_explicit_config() -> None:
    """Test call with explicit config parameter."""
    mock_response = MockLLMResponse(answer="Yes", confidence=1.0)

    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = mock_response
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        config = LLMConfig(
            api_key="custom-key",
            model="custom-model",
            endpoint="https://custom.api",
            timeout=45,
            max_tokens=500,
        )

        call_llm("sys", "user", MockLLMResponse, config=config)

        mock_openai_cls.assert_called_once_with(
            api_key="custom-key", base_url="https://custom.api", timeout=45.0
        )

        mock_client.beta.chat.completions.parse.assert_called_once_with(
            model="custom-model",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
            ],
            response_format=MockLLMResponse,
            max_tokens=500,
        )


def test_call_llm_with_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test call with config loaded from environment (config=None)."""
    mock_response = MockLLMResponse(answer="Env config works", confidence=0.8)

    # Mock environment variables for config loading
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "env-key")
    monkeypatch.setenv("JUDGE_LLM_MODEL", "env-model")
    monkeypatch.setenv("JUDGE_LLM_TIMEOUT", "30")

    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = mock_response
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        # Call with config=None (default)
        result = call_llm("sys", "user", MockLLMResponse, config=None)

        assert result.success is True
        assert result.value == mock_response

        # Verify it used the env config
        mock_openai_cls.assert_called_once()
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["api_key"] == "env-key"
        assert kwargs["timeout"] == 30.0
