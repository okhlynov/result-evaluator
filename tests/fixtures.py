"""Test fixtures and helper functions for test cases."""

from unittest.mock import Mock

import pytest
from openai import AuthenticationError
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture


def dummy_inference(input_data: dict) -> dict:
    """Simple inference function that returns a constant dict for testing.

    Args:
        input_data: Input data dict (passed but not used in this dummy implementation)

    Returns:
        A constant dict with predefined structure
    """
    return {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }


def echo_inference(input_data: dict) -> dict:
    """Inference function that echoes back the input data.

    Args:
        input_data: Input data dict to echo back

    Returns:
        Dict containing the original input wrapped in a result key
    """
    return {
        "status": "success",
        "input_received": input_data,
    }


def rich_inference(input_data: dict) -> dict:
    """Inference function returning rich structured data for testing multiple operators.

    Args:
        input_data: Input data dict (passed but not used)

    Returns:
        Dict with various data types for comprehensive operator testing
    """
    return {
        "status": "completed",
        "message": "Processing complete with code-123",
        "tags": ["python", "testing", "yaml"],
        "metadata": {
            "version": "1.0.0",
            "author": "test-user",
        },
        "items": ["item1", "item2", "item3", "item4", "item5"],
    }


@pytest.fixture
def mock_llm_success(mocker: MockerFixture) -> Mock:
    """Fixture that returns a mock OpenAI client configured for success.

    Returns:
        Mock: Properly configured mock OpenAI client with successful response
    """
    class DefaultResponse(BaseModel):
        message: str

    mock_parsed = DefaultResponse(message="success")
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(parsed=mock_parsed))]
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    mocker.patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_llm_connection_error(mocker: MockerFixture) -> Mock:
    """Fixture that returns a mock OpenAI client that raises AuthenticationError.

    Returns:
        Mock: Properly configured mock OpenAI client raising AuthenticationError
    """
    mock_client = Mock()
    mock_client.beta.chat.completions.parse.side_effect = AuthenticationError(
        message="Authentication failed", response=Mock(), body={}
    )

    mocker.patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_llm_validation_error(mocker: MockerFixture) -> Mock:
    """Fixture that returns a mock OpenAI client that raises ValidationError.

    Returns:
        Mock: Properly configured mock OpenAI client raising ValidationError
    """
    mock_client = Mock()
    # Pydantic's ValidationError requires some data to be instantiated properly
    error = ValidationError.from_exception_data("Validation error", line_errors=[])
    mock_client.beta.chat.completions.parse.side_effect = error

    mocker.patch("result_evaluator.runtime.llm.OpenAI", return_value=mock_client)
    return mock_client