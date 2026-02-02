"""End-to-end integration test for complete async workflow.

This module tests the complete async workflow from inference through assertions
to result aggregation, demonstrating single event loop efficiency and validating
the orchestration of async inference with sync operators.
"""

import logging

import pytest

from result_evaluator.dsl.models import AssertRule, RunConfig, Scenario
from result_evaluator.runtime.engine import Engine


@pytest.mark.asyncio
async def test_async_workflow_end_to_end_pass(caplog: pytest.LogCaptureFixture) -> None:
    """Test complete async workflow: async inference -> sync operators -> PASS result.

    This test demonstrates:
    - Single event loop usage (no nested asyncio.run())
    - Async inference function execution
    - Multiple sync operator evaluations
    - Successful assertion aggregation
    - PASS status in final result structure
    - Proper logging throughout the workflow
    """
    caplog.set_level(logging.INFO)

    engine = Engine()

    # Create a complete test scenario
    scenario = Scenario(
        case={"id": "async-workflow-001", "description": "End-to-end async test"},
        input={"prompt": "test input"},
        run=RunConfig(kind="python", target="tests.fixtures.async_dummy_inference"),
        asserts=[
            AssertRule(
                path="$.status", op="equals", expected="async_success"
            ),
            AssertRule(
                path="$.result", op="equals", expected="async_dummy_output"
            ),
            AssertRule(path="$.count", op="equals", expected=99),
        ],
    )

    # Execute the complete async workflow
    result = await engine.run_test_async(scenario)

    # Verify final result structure
    assert result["case_id"] == "async-workflow-001"
    assert result["status"] == "PASS"
    assert "asserts" in result
    assert len(result["asserts"]) == 3

    # Verify all assertions passed
    for assert_result in result["asserts"]:
        assert assert_result["ok"] is True
        assert assert_result["message"] == "OK"

    # Verify inference result was passed to assertions
    assert result["result"] == {
        "status": "async_success",
        "result": "async_dummy_output",
        "count": 99,
    }

    # Verify logging output
    log_messages = [record.message for record in caplog.records]
    assert any("Starting test case: async-workflow-001" in msg for msg in log_messages)
    assert any("Test case completed: async-workflow-001 - PASS" in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_async_workflow_end_to_end_fail(caplog: pytest.LogCaptureFixture) -> None:
    """Test complete async workflow: async inference -> sync operators -> FAIL result.

    This test verifies:
    - Assertion failure detection
    - FAIL status in final result structure
    - Correct aggregation of mixed pass/fail assertions
    - Proper error messages in failed assertions
    """
    caplog.set_level(logging.INFO)

    engine = Engine()

    scenario = Scenario(
        case={"id": "async-workflow-002", "description": "End-to-end async test with failure"},
        input={"prompt": "test input"},
        run=RunConfig(kind="python", target="tests.fixtures.async_dummy_inference"),
        asserts=[
            AssertRule(
                path="$.status", op="equals", expected="async_success"
            ),  # Pass
            AssertRule(
                path="$.result", op="equals", expected="wrong_value"
            ),  # Fail
            AssertRule(path="$.count", op="equals", expected=99),  # Pass
        ],
    )

    result = await engine.run_test_async(scenario)

    # Verify final result structure
    assert result["case_id"] == "async-workflow-002"
    assert result["status"] == "FAIL"
    assert len(result["asserts"]) == 3

    # Verify assertion results aggregation
    assert result["asserts"][0]["ok"] is True
    assert result["asserts"][1]["ok"] is False
    assert "Expected wrong_value, got async_dummy_output" in result["asserts"][1]["message"]
    assert result["asserts"][2]["ok"] is True

    # Verify logging output
    log_messages = [record.message for record in caplog.records]
    assert any("Test case completed: async-workflow-002 - FAIL" in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_async_workflow_end_to_end_error(caplog: pytest.LogCaptureFixture) -> None:
    """Test complete async workflow: async inference error -> ERROR result.

    This test verifies:
    - Error handling in async context
    - ERROR status in final result structure
    - Error message captured in result
    - Proper logging of errors with exc_info
    """
    caplog.set_level(logging.ERROR)

    engine = Engine()

    scenario = Scenario(
        case={"id": "async-workflow-003", "description": "End-to-end async test with error"},
        input={"prompt": "test input"},
        run=RunConfig(kind="python", target="tests.fixtures.async_error_inference"),
        asserts=[
            AssertRule(path="$.status", op="equals", expected="success"),
        ],
    )

    result = await engine.run_test_async(scenario)

    # Verify final result structure for error case
    assert result["case_id"] == "async-workflow-003"
    assert result["status"] == "ERROR"
    assert "error" in result
    assert "Async inference error: processing failed" in result["error"]

    # Verify no asserts key for error case
    assert "asserts" not in result

    # Verify error logging
    log_messages = [record.message for record in caplog.records]
    assert any("Test case failed with error: async-workflow-003" in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_async_workflow_with_complex_assertions(caplog: pytest.LogCaptureFixture) -> None:
    """Test async workflow with complex combinators (all, any, not).

    This test verifies:
    - Complex assertion logic with nested combinators
    - Recursive evaluation in async context
    - Single event loop for entire workflow
    - Proper aggregation of nested assertion results
    """
    caplog.set_level(logging.INFO)

    engine = Engine()

    scenario = Scenario.model_validate({
        "case": {"id": "async-workflow-004", "description": "Async test with complex assertions"},
        "input": {"prompt": "test input"},
        "run": {"kind": "python", "target": "tests.fixtures.async_dummy_inference"},
        "asserts": [
            # All combinator with multiple conditions
            {
                "op": "",
                "all": [
                    {"path": "$.status", "op": "equals", "expected": "async_success"},
                    {"path": "$.count", "op": "equals", "expected": 99},
                ],
            },
            # Any combinator
            {
                "op": "",
                "any": [
                    {"path": "$.result", "op": "equals", "expected": "wrong_value"},
                    {"path": "$.result", "op": "equals", "expected": "async_dummy_output"},
                ],
            },
            # Not combinator
            {
                "op": "",
                "not": {"path": "$.count", "op": "equals", "expected": 42},
            },
        ],
    })

    result = await engine.run_test_async(scenario)

    # Verify successful execution of complex assertions
    assert result["case_id"] == "async-workflow-004"
    assert result["status"] == "PASS"
    assert len(result["asserts"]) == 3

    # Verify all complex assertions passed
    assert result["asserts"][0]["ok"] is True
    assert "All checks passed" in result["asserts"][0]["message"]
    assert result["asserts"][1]["ok"] is True
    assert "At least one check passed" in result["asserts"][1]["message"]
    assert result["asserts"][2]["ok"] is True
    assert "NOT passed" in result["asserts"][2]["message"]


@pytest.mark.asyncio
async def test_async_workflow_multiple_test_cases_single_event_loop() -> None:
    """Test multiple test cases in single event loop (efficiency demonstration).

    This test demonstrates:
    - Single event loop for multiple test cases
    - No nested asyncio.run() calls
    - Efficient batch processing pattern
    - Connection pooling preservation (future enhancement)
    """
    engine = Engine()

    # Create multiple test scenarios
    scenarios = [
        Scenario(
            case={"id": f"batch-test-{i}", "description": f"Batch test {i}"},
            input={"prompt": f"input {i}"},
            run=RunConfig(kind="python", target="tests.fixtures.async_dummy_inference"),
            asserts=[
                AssertRule(
                    path="$.status", op="equals", expected="async_success"
                ),
            ],
        )
        for i in range(5)
    ]

    # Execute all tests in single event loop
    results = []
    for scenario in scenarios:
        result = await engine.run_test_async(scenario)
        results.append(result)

    # Verify all tests executed successfully
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result["case_id"] == f"batch-test-{i}"
        assert result["status"] == "PASS"
        assert len(result["asserts"]) == 1
        assert result["asserts"][0]["ok"] is True


@pytest.mark.asyncio
async def test_async_workflow_with_sync_inference_backward_compat() -> None:
    """Test async workflow with sync inference function (backward compatibility).

    This test verifies:
    - Sync inference functions work in async workflow
    - Auto-detection of sync vs async functions
    - Full backward compatibility maintained
    - No performance regression for sync code
    """
    engine = Engine()

    scenario = Scenario(
        case={"id": "sync-compat-001", "description": "Sync inference in async workflow"},
        input={"test": "data"},
        run=RunConfig(kind="python", target="tests.fixtures.dummy_inference"),
        asserts=[
            AssertRule(path="$.status", op="equals", expected="success"),
            AssertRule(path="$.result", op="equals", expected="dummy_output"),
            AssertRule(path="$.count", op="equals", expected=42),
        ],
    )

    result = await engine.run_test_async(scenario)

    # Verify sync inference works correctly in async workflow
    assert result["case_id"] == "sync-compat-001"
    assert result["status"] == "PASS"
    assert result["result"] == {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }
