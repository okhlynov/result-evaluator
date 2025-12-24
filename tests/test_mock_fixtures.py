from unittest.mock import Mock

from pydantic import BaseModel

from result_evaluator.runtime.config import LLMConfig
from result_evaluator.runtime.llm import call_llm


class MockResponse(BaseModel):
    message: str


DUMMY_CONFIG = LLMConfig(api_key="test", model="test") # pyright: ignore[reportCallIssue]


def test_mock_llm_success_fixture(mock_llm_success: Mock) -> None:
    """Verify that mock_llm_success fixture works."""
    result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

    assert result.success
    assert result.value.message  == "success" # pyright: ignore[reportOptionalMemberAccess]
    mock_llm_success.beta.chat.completions.parse.assert_called_once()


def test_mock_llm_connection_error_fixture(mock_llm_connection_error: Mock) -> None:
    """Verify that mock_llm_connection_error fixture works."""
    result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

    assert not result.success
    assert result.error_type == "connection"
    assert "Authentication failed" in (result.error or "")


def test_mock_llm_validation_error_fixture(mock_llm_validation_error: Mock) -> None:
    """Verify that mock_llm_validation_error fixture works."""
    result = call_llm("sys", "user", MockResponse, config=DUMMY_CONFIG)

    assert not result.success
    assert result.error_type == "validation"
    assert "Response doesn't match schema" in (result.error or "")
