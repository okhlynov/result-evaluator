from unittest.mock import MagicMock, patch
import logging

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

DUMMY_CONFIG = LLMConfig(api_key="test", model="test", endpoint="http://test")

def test_call_llm_openai_integration():
    """Test that OpenAI client is called correctly."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_parsed = MockResponse(message="success")
    
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed
    mock_client.beta.chat.completions.parse.return_value = mock_completion
    
    with patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client) as mock_openai_cls:
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
            api_key="test_key",
            base_url="http://test",
            timeout=30.0
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
    monkeypatch.setattr("result_evaluator.runtime.llm.load_llm_config", lambda: mock_config)
    
    # We also need to mock OpenAI to avoid actual call
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        # Setup mock return
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.return_value.choices[0].message.parsed = "parsed_value"
        
        call_llm("system", "user", str, config=None)
        
        mock_openai_cls.assert_called_once()
        # check if called with config from env
        args, kwargs = mock_openai_cls.call_args
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
        mock_instance.beta.chat.completions.parse.side_effect = ValidationError.from_exception_data(
            "Validation error", line_errors=[]
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

def test_call_llm_logging_success(caplog):
    """Test successful call logs correct info."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_parsed = MockResponse(message="success")
    
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed
    mock_client.beta.chat.completions.parse.return_value = mock_completion
    
    with patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client):
        with caplog.at_level(logging.DEBUG):
            call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)
            
            assert "LLM call completed" in caplog.text
            
            # Find the record with our message
            record = next(r for r in caplog.records if "LLM call completed" in r.message)
            assert hasattr(record, "llm_call")
            assert record.llm_call["model"] == "test"
            assert record.llm_call["latency_ms"] >= 0
            
            assert DUMMY_CONFIG.api_key not in caplog.text

def test_call_llm_logging_error(caplog):
    """Test error call logs exception."""
    with patch("result_evaluator.runtime.llm.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_instance.beta.chat.completions.parse.side_effect = Exception("Boom")
        
        with caplog.at_level(logging.ERROR):
            call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)
            
            assert "Unexpected error" in caplog.text
            # Verify exc_info was logged (at least one record has exc_info)
            assert any(record.exc_info for record in caplog.records)

def test_call_llm_logging_large_prompt(caplog):
    """Test warning for large prompts."""
    large_prompt = "x" * 100_001
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = MockResponse(message="ok")
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    with patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client):
        with caplog.at_level(logging.WARNING):
            call_llm("sys", large_prompt, MockResponse, config=DUMMY_CONFIG)
            
            assert "Large prompt detected" in caplog.text
            assert "100004" in caplog.text # 3 chars 'sys' + 100001 chars

def test_call_llm_refusal():
    """Test handling of refusal (parsed is None)."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = None # Refusal
    mock_client.beta.chat.completions.parse.return_value = mock_completion
    
    with patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client):
        result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)
        
        assert not result.success
        assert result.error_type == "response"
        assert "Model refused" in result.error