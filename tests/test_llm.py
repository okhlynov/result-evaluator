from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from result_evaluator.runtime.config import LLMConfig
from result_evaluator.runtime.llm import call_llm


def test_call_llm_validation_system_prompt_empty():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        call_llm("", "user prompt", str)

def test_call_llm_validation_system_prompt_whitespace():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        call_llm("   ", "user prompt", str)

def test_call_llm_validation_user_prompt_empty():
    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        call_llm("system prompt", "", str)

def test_call_llm_validation_user_prompt_whitespace():
    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        call_llm("system prompt", "   ", str)

class MockResponse(BaseModel):
    message: str

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
