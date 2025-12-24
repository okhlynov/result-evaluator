import pytest

from result_evaluator.runtime.config import LLMConfig, load_llm_config


def test_load_llm_config_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("JUDGE_LLM_MODEL", "gpt-4")
    config = load_llm_config()
    assert config.api_key == "test-key"
    assert config.model == "gpt-4"
    assert config.timeout == 60
    assert config.max_tokens is None
    assert config.endpoint is None


def test_load_llm_config_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUDGE_LLM_API_KEY", raising=False)
    monkeypatch.setenv("JUDGE_LLM_MODEL", "gpt-4")
    with pytest.raises(
        ValueError, match="Missing required environment variable: JUDGE_LLM_API_KEY"
    ):
        load_llm_config()


def test_load_llm_config_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "test-key")
    monkeypatch.delenv("JUDGE_LLM_MODEL", raising=False)
    with pytest.raises(
        ValueError, match="Missing required environment variable: JUDGE_LLM_MODEL"
    ):
        load_llm_config()


def test_load_llm_config_custom_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-key")
    monkeypatch.setenv("CUSTOM_MODEL", "custom-model")
    config = load_llm_config(prefix="CUSTOM")
    assert config.api_key == "custom-key"
    assert config.model == "custom-model"


def test_load_llm_config_optional_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("JUDGE_LLM_MODEL", "gpt-4")
    monkeypatch.setenv("JUDGE_LLM_TIMEOUT", "30")
    monkeypatch.setenv("JUDGE_LLM_MAX_TOKENS", "100")
    monkeypatch.setenv("JUDGE_LLM_ENDPOINT", "https://api.openai.com/v1")

    config = load_llm_config()
    assert config.timeout == 30
    assert config.max_tokens == 100
    assert config.endpoint == "https://api.openai.com/v1"


def test_load_llm_config_invalid_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("JUDGE_LLM_MODEL", "gpt-4")
    monkeypatch.setenv("JUDGE_LLM_TIMEOUT", "not-an-int")
    with pytest.raises(
        ValueError, match="Environment variable JUDGE_LLM_TIMEOUT must be an integer"
    ):
        load_llm_config()


def test_load_llm_config_invalid_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("JUDGE_LLM_MODEL", "gpt-4")
    monkeypatch.setenv("JUDGE_LLM_MAX_TOKENS", "not-an-int")
    with pytest.raises(
        ValueError, match="Environment variable JUDGE_LLM_MAX_TOKENS must be an integer"
    ):
        load_llm_config()


def test_llm_config_validation_empty_key() -> None:
    with pytest.raises(ValueError, match="Field cannot be empty or whitespace only"):
        LLMConfig(
            api_key="  ",
            model="gpt-4",
            endpoint="https://api.openai.com/v1",
            timeout=60,
            max_tokens=1000,
        )


def test_llm_config_validation_empty_model() -> None:
    with pytest.raises(ValueError, match="Field cannot be empty or whitespace only"):
        LLMConfig(
            api_key="test-key",
            model="  ",
            endpoint="https://api.openai.com/v1",
            timeout=60,
            max_tokens=1000,
        )


def test_llm_config_validation_zero_timeout() -> None:
    with pytest.raises(ValueError, match="Timeout must be a positive integer"):
        LLMConfig(
            api_key="test-key",
            model="gpt-4",
            endpoint="https://api.openai.com/v1",
            timeout=0,
            max_tokens=1000,
        )


def test_llm_config_validation_negative_timeout() -> None:
    with pytest.raises(ValueError, match="Timeout must be a positive integer"):
        LLMConfig(
            api_key="test-key",
            model="gpt-4",
            endpoint="https://api.openai.com/v1",
            timeout=-1,
            max_tokens=1000,
        )
