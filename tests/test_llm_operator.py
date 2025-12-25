"""Test infrastructure for LLM judge operator with mocked LLM."""

from unittest.mock import Mock

import pytest
from _pytest.logging import LogCaptureFixture
from pytest_mock import MockerFixture

from result_evaluator.runtime.llm import Result
from result_evaluator.runtime.operators import LLMJudgeResponse, op_llm_judge


def create_llm_response(verdict: bool, reasoning: str) -> Result[LLMJudgeResponse]:
    """Helper function to build successful LLM responses.

    Args:
        verdict: The verdict from the LLM judge
        reasoning: The reasoning/explanation from the LLM judge

    Returns:
        Result[LLMJudgeResponse]: A successful result with the response
    """
    response = LLMJudgeResponse(verdict=verdict, reasoning=reasoning)
    return Result.ok(response)


def create_llm_error_response(
    error_type: str, message: str
) -> Result[LLMJudgeResponse]:
    """Helper function to build failed LLM responses.

    Args:
        error_type: Type of error (e.g., "connection", "validation")
        message: Error message description

    Returns:
        Result[LLMJudgeResponse]: A failed result with error details
    """
    return Result.fail(error_type, message)


@pytest.fixture
def mock_call_llm_success(mocker: MockerFixture) -> Mock:
    """Fixture that returns a mock call_llm configured for success.

    Returns a mock that always returns a successful result with:
    - verdict=True
    - reasoning="Semantically equivalent"

    Returns:
        Mock: Configured mock for call_llm
    """
    mock_response = create_llm_response(True, "Semantically equivalent")
    mocker.patch("result_evaluator.runtime.operators.load_llm_config")
    return mocker.patch(
        "result_evaluator.runtime.operators.call_llm",
        return_value=mock_response,
    )


@pytest.fixture
def mock_call_llm_failure(mocker: MockerFixture) -> Mock:
    """Fixture that returns a mock call_llm configured for failure.

    Returns a mock that always returns a failed result with:
    - error_type="connection"
    - error="Connection timeout"

    Returns:
        Mock: Configured mock for call_llm
    """
    mock_response = create_llm_error_response("connection", "Connection timeout")
    mocker.patch("result_evaluator.runtime.operators.load_llm_config")
    return mocker.patch(
        "result_evaluator.runtime.operators.call_llm",
        return_value=mock_response,
    )


@pytest.fixture
def mock_call_llm_configurable(mocker: MockerFixture):
    """Fixture that returns a mock call_llm with configurable verdict.

    This fixture provides a mock that can be configured to return different
    verdicts and reasoning via its return_value.

    Returns:
        Mock: Configured mock for call_llm with return_value property for customization
    """
    mocker.patch("result_evaluator.runtime.operators.load_llm_config")
    mock = mocker.patch(
        "result_evaluator.runtime.operators.call_llm",
    )
    mock.return_value = create_llm_response(True, "Default response")
    return mock


# ============================================================================
# Basic Infrastructure Tests
# ============================================================================


def test_create_llm_response_success() -> None:
    """Test that create_llm_response creates successful results."""
    result = create_llm_response(True, "Test reasoning")
    assert result.success is True
    assert result.value is not None
    assert result.value.verdict is True
    assert result.value.reasoning == "Test reasoning"
    assert result.error is None


def test_create_llm_response_false_verdict() -> None:
    """Test create_llm_response with False verdict."""
    result = create_llm_response(False, "Mismatch found")
    assert result.success is True
    assert result.value is not None
    assert result.value.verdict is False
    assert result.value.reasoning == "Mismatch found"


def test_create_llm_error_response() -> None:
    """Test that create_llm_error_response creates failed results."""
    result = create_llm_error_response("connection", "Network error")
    assert result.success is False
    assert result.value is None
    assert result.error == "Network error"
    assert result.error_type == "connection"


def test_create_llm_error_response_validation() -> None:
    """Test create_llm_error_response with validation error."""
    result = create_llm_error_response("validation", "Invalid response format")
    assert result.success is False
    assert result.error_type == "validation"
    assert result.error == "Invalid response format"


# ============================================================================
# Mock Fixture Tests
# ============================================================================


def test_mock_call_llm_success_fixture(mock_call_llm_success: Mock) -> None:
    """Test that mock_call_llm_success fixture returns successful responses."""
    result = op_llm_judge("test", {"ground_truth": "test"})
    assert result.ok is True
    assert mock_call_llm_success.called


def test_mock_call_llm_failure_fixture(mock_call_llm_failure: Mock) -> None:
    """Test that mock_call_llm_failure fixture returns error responses."""
    result = op_llm_judge("test", {"ground_truth": "test"})
    assert result.ok is False
    assert mock_call_llm_failure.called


def test_mock_call_llm_configurable_verdict_true(
    mock_call_llm_configurable: Mock,
) -> None:
    """Test configurable mock with verdict=True."""
    mock_call_llm_configurable.return_value = create_llm_response(
        True, "Matches perfectly"
    )

    result = op_llm_judge("actual", {"ground_truth": "expected", "expected": True})
    assert result.ok is True
    assert mock_call_llm_configurable.called


def test_mock_call_llm_configurable_verdict_false(
    mock_call_llm_configurable: Mock,
) -> None:
    """Test configurable mock with verdict=False."""
    mock_call_llm_configurable.return_value = create_llm_response(
        False, "Different meanings"
    )

    result = op_llm_judge("actual", {"ground_truth": "expected", "expected": True})
    assert result.ok is False
    assert "Different meanings" in (result.message or "")
    assert mock_call_llm_configurable.called


def test_mock_call_llm_configurable_expected_false(
    mock_call_llm_configurable: Mock,
) -> None:
    """Test configurable mock with expected=False."""
    mock_call_llm_configurable.return_value = create_llm_response(
        False, "Correctly identified as different"
    )

    result = op_llm_judge("actual", {"ground_truth": "expected", "expected": False})
    assert result.ok is True
    assert mock_call_llm_configurable.called


# ============================================================================
# Mock Call Argument Verification Tests
# ============================================================================


def test_mock_captures_system_prompt(mock_call_llm_configurable: Mock) -> None:
    """Test that mock can verify system prompt was passed correctly."""
    mock_call_llm_configurable.return_value = create_llm_response(True, "OK")

    op_llm_judge("actual", {"ground_truth": "expected"})

    assert mock_call_llm_configurable.called
    call_args = mock_call_llm_configurable.call_args
    system_prompt = call_args[0][0]
    assert isinstance(system_prompt, str)
    assert len(system_prompt) > 0


def test_mock_captures_user_prompt(mock_call_llm_configurable: Mock) -> None:
    """Test that mock can verify user prompt was passed correctly."""
    mock_call_llm_configurable.return_value = create_llm_response(True, "OK")

    op_llm_judge("actual", {"ground_truth": "expected"})

    assert mock_call_llm_configurable.called
    call_args = mock_call_llm_configurable.call_args
    user_prompt = call_args[0][1]
    assert isinstance(user_prompt, str)
    assert "actual" in user_prompt
    assert "expected" in user_prompt


def test_mock_captures_response_type(mock_call_llm_configurable: Mock) -> None:
    """Test that mock can verify response type was passed correctly."""
    mock_call_llm_configurable.return_value = create_llm_response(True, "OK")

    op_llm_judge("actual", {"ground_truth": "expected"})

    assert mock_call_llm_configurable.called
    call_args = mock_call_llm_configurable.call_args
    response_type = call_args[0][2]
    assert response_type is LLMJudgeResponse


# ============================================================================
# Integration Tests with Infrastructure
# ============================================================================


def test_llm_judge_with_dict_selection(mock_call_llm_success: Mock) -> None:
    """Test op_llm_judge with dict selection using mock infrastructure."""
    selection = {"key": "value", "nested": {"field": 42}}
    result = op_llm_judge(selection, {"ground_truth": "some_data"})

    assert result.ok is True
    assert mock_call_llm_success.called
    # Verify prompts contain serialized data
    call_args = mock_call_llm_success.call_args
    user_prompt = call_args[0][1]
    assert "key" in user_prompt
    assert "value" in user_prompt


def test_llm_judge_with_list_selection(mock_call_llm_success: Mock) -> None:
    """Test op_llm_judge with list selection using mock infrastructure."""
    selection = [1, 2, 3, 4, 5]
    result = op_llm_judge(selection, {"ground_truth": [1, 2, 3]})

    assert result.ok is True
    assert mock_call_llm_success.called
    call_args = mock_call_llm_success.call_args
    user_prompt = call_args[0][1]
    assert "1" in user_prompt and "2" in user_prompt


def test_llm_judge_with_string_selection(mock_call_llm_success: Mock) -> None:
    """Test op_llm_judge with string selection using mock infrastructure."""
    selection = "output text"
    result = op_llm_judge(selection, {"ground_truth": "expected text"})

    assert result.ok is True
    assert mock_call_llm_success.called
    call_args = mock_call_llm_success.call_args
    user_prompt = call_args[0][1]
    assert "output text" in user_prompt


def test_llm_judge_with_custom_prompts(mock_call_llm_success: Mock) -> None:
    """Test op_llm_judge with custom prompts using mock infrastructure."""
    params = {
        "ground_truth": "expected",
        "system_prompt": "Custom: {actual} vs {ground_truth}",
        "prompt": "Check: {actual} == {ground_truth}",
    }
    result = op_llm_judge("actual", params)

    assert result.ok is True
    assert mock_call_llm_success.called
    call_args = mock_call_llm_success.call_args
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]
    assert "Custom:" in system_prompt
    assert "Check:" in user_prompt


def test_multiple_mock_calls_track_all_invocations(
    mock_call_llm_configurable: Mock,
) -> None:
    """Test that mock correctly tracks multiple calls."""
    mock_call_llm_configurable.return_value = create_llm_response(True, "OK")

    op_llm_judge("actual1", {"ground_truth": "expected1"})
    op_llm_judge("actual2", {"ground_truth": "expected2"})
    op_llm_judge("actual3", {"ground_truth": "expected3"})

    assert mock_call_llm_configurable.call_count == 3
    calls = mock_call_llm_configurable.call_args_list
    assert len(calls) == 3


# ============================================================================
# Verdict Comparison Tests
# ============================================================================


def test_verdict_match_true_true(mock_call_llm_configurable: Mock) -> None:
    """Test verdict=True with expected=True (default) → ok=True.

    When LLM returns verdict=True and expected defaults to True,
    the operator should return ok=True.
    """
    mock_call_llm_configurable.return_value = create_llm_response(
        True, "Semantically equivalent"
    )

    result = op_llm_judge("actual", {"ground_truth": "expected"})

    assert result.ok is True
    assert mock_call_llm_configurable.called


def test_verdict_mismatch_false_true(mock_call_llm_configurable: Mock) -> None:
    """Test verdict=False with expected=True → ok=False with reasoning in message.

    When LLM returns verdict=False but expected is True,
    the operator should return ok=False and include reasoning in message.
    """
    reasoning = "Outputs convey different meanings"
    mock_call_llm_configurable.return_value = create_llm_response(False, reasoning)

    result = op_llm_judge("actual", {"ground_truth": "expected", "expected": True})

    assert result.ok is False
    assert result.message is not None
    assert reasoning in result.message
    assert mock_call_llm_configurable.called


def test_verdict_match_false_false(mock_call_llm_configurable: Mock) -> None:
    """Test verdict=False with expected=False → ok=True.

    When LLM returns verdict=False and expected is False,
    the operator should return ok=True (testing for non-equivalence works).
    """
    mock_call_llm_configurable.return_value = create_llm_response(
        False, "Correctly identified as different"
    )

    result = op_llm_judge("actual", {"ground_truth": "expected", "expected": False})

    assert result.ok is True
    assert mock_call_llm_configurable.called


def test_verdict_mismatch_true_false(mock_call_llm_configurable: Mock) -> None:
    """Test verdict=True with expected=False → ok=False.

    When LLM returns verdict=True but expected is False,
    the operator should return ok=False because the verdict
    doesn't match the expected inverse condition.
    """
    mock_call_llm_configurable.return_value = create_llm_response(
        True, "Semantically equivalent"
    )

    result = op_llm_judge("actual", {"ground_truth": "expected", "expected": False})

    assert result.ok is False
    assert mock_call_llm_configurable.called


def test_default_expected_true(mock_call_llm_configurable: Mock) -> None:
    """Test that expected parameter defaults to True when omitted.

    When expected is not provided in parameters,
    it should default to True, treating True verdict as success.
    """
    mock_call_llm_configurable.return_value = create_llm_response(True, "Default test")

    # Call without providing expected parameter
    result = op_llm_judge("actual", {"ground_truth": "expected"})

    assert result.ok is True
    assert mock_call_llm_configurable.called


def test_reasoning_in_message_when_verdict_mismatch(
    mock_call_llm_configurable: Mock,
) -> None:
    """Test that reasoning from LLMJudgeResponse appears in message on failure.

    When verdict doesn't match expected, the operator message
    should contain the reasoning provided by the LLM response.
    """
    detailed_reasoning = (
        "The outputs have fundamentally different approaches and results"
    )
    mock_call_llm_configurable.return_value = create_llm_response(
        False, detailed_reasoning
    )

    result = op_llm_judge("actual", {"ground_truth": "expected", "expected": True})

    assert result.ok is False
    assert result.message is not None
    assert detailed_reasoning in result.message


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_empty_selection(mock_call_llm_success: Mock) -> None:
    """Test: Empty string selection "" → serialized and judged.

    When selection is an empty string, it should be serialized
    as {"value": ""} and sent to LLM for judgment.
    """
    selection = ""
    result = op_llm_judge(selection, {"ground_truth": "expected"})

    assert result.ok is True
    assert mock_call_llm_success.called
    call_args = mock_call_llm_success.call_args
    user_prompt = call_args[0][1]
    # Empty string should be serialized as {"value": ""}
    assert '""' in user_prompt  # JSON representation of empty string


def test_none_selection(mock_call_llm_success: Mock) -> None:
    """Test: None selection → serialized as {"value": null} and judged.

    When selection is None, it should be serialized as {"value": null}
    (JSON representation of None) and sent to LLM.
    """
    selection = None
    result = op_llm_judge(selection, {"ground_truth": "expected"})

    assert result.ok is True
    assert mock_call_llm_success.called
    call_args = mock_call_llm_success.call_args
    user_prompt = call_args[0][1]
    # None should be serialized as {"value": null}
    assert "null" in user_prompt


def test_large_selection_warning(
    mock_call_llm_success: Mock, caplog: LogCaptureFixture
) -> None:
    """Test: Large selection (>50KB) → warning logged but processing continues.

    When selection size exceeds 50KB when serialized, a warning should be
    logged but the operator should continue processing and return success.
    """
    # Create a large string (>50KB)
    large_selection = "x" * (50_001)

    with caplog.at_level("WARNING"):
        result = op_llm_judge(large_selection, {"ground_truth": "expected"})

    # Processing should continue successfully despite size warning
    assert result.ok is True
    assert mock_call_llm_success.called
    # Check that warning was logged
    assert any(
        "50" in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )


def test_custom_user_prompt(mock_call_llm_configurable: Mock) -> None:
    """Test: Custom user prompt → correctly formatted and used.

    When a custom prompt template is provided, it should be used
    instead of the default user prompt.
    """
    mock_call_llm_configurable.return_value = create_llm_response(True, "OK")

    custom_prompt = "Compare {actual} with {ground_truth}"
    result = op_llm_judge(
        "actual", {"ground_truth": "expected", "prompt": custom_prompt}
    )

    assert result.ok is True
    assert mock_call_llm_configurable.called
    call_args = mock_call_llm_configurable.call_args
    user_prompt = call_args[0][1]
    # Verify the custom prompt was used (placeholders filled)
    assert "Compare" in user_prompt
    assert "actual" in user_prompt
    assert "expected" in user_prompt


def test_custom_system_prompt(mock_call_llm_configurable: Mock) -> None:
    """Test: Custom system prompt → correctly formatted and used.

    When a custom system_prompt template is provided, it should be used
    instead of the default system prompt.
    """
    mock_call_llm_configurable.return_value = create_llm_response(True, "OK")

    custom_system = "System: Compare {actual} vs {ground_truth}"
    result = op_llm_judge(
        "actual", {"ground_truth": "expected", "system_prompt": custom_system}
    )

    assert result.ok is True
    assert mock_call_llm_configurable.called
    call_args = mock_call_llm_configurable.call_args
    system_prompt = call_args[0][0]
    # Verify the custom system prompt was used (placeholders filled)
    assert "System:" in system_prompt
    assert "Compare" in system_prompt


def test_both_custom_prompts(mock_call_llm_configurable: Mock) -> None:
    """Test: Both custom prompts provided → both used correctly.

    When both system_prompt and prompt are provided, both custom
    templates should be used instead of defaults.
    """
    mock_call_llm_configurable.return_value = create_llm_response(True, "OK")

    custom_system = "Evaluate: {actual} vs {ground_truth}"
    custom_user = "Determine if {actual} matches {ground_truth}"

    result = op_llm_judge(
        "actual",
        {
            "ground_truth": "expected",
            "system_prompt": custom_system,
            "prompt": custom_user,
        },
    )

    assert result.ok is True
    assert mock_call_llm_configurable.called
    call_args = mock_call_llm_configurable.call_args
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]

    # Verify both custom prompts were used
    assert "Evaluate:" in system_prompt
    assert "Determine if" in user_prompt


def test_ground_truth_dict(mock_call_llm_success: Mock) -> None:
    """Test: Ground truth as dict → serialized properly.

    When ground_truth is a dict, it should be serialized as-is
    (not wrapped in {"value": ...}).
    """
    ground_truth = {"key": "value", "nested": {"field": 42}}
    result = op_llm_judge("actual", {"ground_truth": ground_truth})

    assert result.ok is True
    assert mock_call_llm_success.called
    call_args = mock_call_llm_success.call_args
    user_prompt = call_args[0][1]
    # Dict should be serialized with its keys visible
    assert "key" in user_prompt
    assert "value" in user_prompt
    assert "nested" in user_prompt


def test_ground_truth_string(mock_call_llm_success: Mock) -> None:
    """Test: Ground truth as string → wrapped and serialized properly.

    When ground_truth is a string, it should be wrapped in {"value": ...}
    to preserve the type information for the LLM.
    """
    ground_truth = "expected output text"
    result = op_llm_judge("actual", {"ground_truth": ground_truth})

    assert result.ok is True
    assert mock_call_llm_success.called
    call_args = mock_call_llm_success.call_args
    user_prompt = call_args[0][1]
    # String should be wrapped in {"value": ...}
    assert "expected output text" in user_prompt
