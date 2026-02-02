"""Backward compatibility tests for sync wrapper methods.

This test file verifies that the sync wrapper methods (run_inference, eval_assert, run_test)
maintain backward compatibility after refactoring to use async internally.
All sync APIs must work without changes to existing code.
"""

import pytest

from result_evaluator.dsl.models import AssertRule, RunConfig, Scenario
from result_evaluator.runtime.engine import Engine


def test_run_inference_sync_no_event_loop_required():
    """Test run_inference can be called from non-async context without event loop.

    Verifies DoD #4: Sync methods can be called from non-async context.
    """
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.dummy_inference")
    input_data = {"test": "data"}

    # This call should work in non-async context (no await, no event loop)
    result = engine.run_inference(run_config, input_data)

    assert isinstance(result, dict)
    assert result == {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }


def test_run_inference_sync_return_type_unchanged():
    """Test run_inference returns same type and structure as before refactoring.

    Verifies DoD #5: Sync methods return same types and structures.
    """
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.echo_inference")
    input_data = {"key": "value", "number": 123}

    result = engine.run_inference(run_config, input_data)

    # Return type must be dict (not coroutine, not awaitable)
    assert isinstance(result, dict)
    assert "status" in result
    assert "input_received" in result
    assert result["input_received"] == input_data


def test_eval_assert_sync_no_event_loop_required():
    """Test eval_assert can be called from non-async context without event loop.

    Verifies DoD #4: Sync methods can be called from non-async context.
    """
    document = {"name": "Alice", "age": 30}
    rule = AssertRule.model_validate(
        {"path": "$.name", "op": "equals", "expected": "Alice"}
    )

    engine = Engine()
    # This call should work in non-async context (no await, no event loop)
    ok, message = engine.eval_assert(rule, document)

    assert isinstance(ok, bool)
    assert isinstance(message, str)
    assert ok is True


def test_eval_assert_sync_return_type_unchanged():
    """Test eval_assert returns same tuple structure as before refactoring.

    Verifies DoD #5: Sync methods return same types and structures.
    """
    document = {"name": "Alice", "age": 30}
    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {"path": "$.age", "op": "equals", "expected": 30},
            ],
        }
    )

    engine = Engine()
    result = engine.eval_assert(rule, document)

    # Return type must be tuple of (bool, str)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)


def test_eval_assert_sync_with_multiple_operators():
    """Test eval_assert with various sync operators maintains backward compatibility.

    Verifies DoD #2: eval_assert() with sync operator returns correct result.
    """
    document = {
        "status": "completed",
        "message": "Processing complete with code-123",
        "tags": ["python", "testing", "yaml"],
        "count": 5,
    }

    # Test equals operator
    rule_equals = AssertRule.model_validate(
        {"path": "$.status", "op": "equals", "expected": "completed"}
    )
    engine = Engine()
    ok, message = engine.eval_assert(rule_equals, document)
    assert ok is True

    # Test contains operator
    rule_contains = AssertRule.model_validate(
        {"path": "$.tags", "op": "contains", "expected": "python"}
    )
    ok, message = engine.eval_assert(rule_contains, document)
    assert ok is True

    # Test length_ge operator
    rule_length = AssertRule.model_validate(
        {"path": "$.tags", "op": "length_ge", "expected": 3}
    )
    ok, message = engine.eval_assert(rule_length, document)
    assert ok is True

    # Test match_regex operator
    rule_regex = AssertRule.model_validate(
        {"path": "$.message", "op": "match_regex", "expected": "^Processing.*code-\\d+$"}
    )
    ok, message = engine.eval_assert(rule_regex, document)
    assert ok is True


def test_run_test_sync_no_event_loop_required():
    """Test run_test can be called from non-async context without event loop.

    Verifies DoD #4: Sync methods can be called from non-async context.
    """
    test_case = Scenario.model_validate(
        {
            "case": {"id": "sync_compat_test", "description": "Backward compat test"},
            "input": {"test": "data"},
            "run": {"kind": "python", "target": "tests.fixtures.dummy_inference"},
            "asserts": [
                {"path": "$.status", "op": "equals", "expected": "success"},
                {"path": "$.count", "op": "equals", "expected": 42},
            ],
        }
    )

    engine = Engine()
    # This call should work in non-async context (no await, no event loop)
    result = engine.run_test(test_case)

    assert isinstance(result, dict)
    assert result["status"] == "PASS"


def test_run_test_sync_return_type_unchanged():
    """Test run_test returns same dict structure as before refactoring.

    Verifies DoD #5: Sync methods return same types and structures.
    """
    test_case = Scenario.model_validate(
        {
            "case": {"id": "type_check_test", "description": "Type verification test"},
            "input": {"test": "data"},
            "run": {"kind": "python", "target": "tests.fixtures.dummy_inference"},
            "asserts": [
                {"path": "$.status", "op": "equals", "expected": "success"},
            ],
        }
    )

    engine = Engine()
    result = engine.run_test(test_case)

    # Return type must be dict with specific structure
    assert isinstance(result, dict)
    assert "case_id" in result
    assert "status" in result
    assert "asserts" in result
    assert "result" in result
    assert isinstance(result["asserts"], list)
    assert result["case_id"] == "type_check_test"


def test_run_test_sync_complete_workflow():
    """Test complete run_test workflow with sync inference function and operators.

    Verifies DoD #3: run_test() complete workflow with sync components returns correct result.
    """
    test_case = Scenario.model_validate(
        {
            "case": {
                "id": "complete_workflow_test",
                "description": "Full sync workflow test",
            },
            "input": {"test": "data"},
            "run": {"kind": "python", "target": "tests.fixtures.rich_inference"},
            "asserts": [
                {"path": "$.status", "op": "exists"},
                {"path": "$.status", "op": "equals", "expected": "completed"},
                {"path": "$.message", "op": "contains", "expected": "code-123"},
                {
                    "path": "$.message",
                    "op": "match_regex",
                    "expected": "^Processing.*code-\\d+$",
                },
                {"path": "$.tags", "op": "contains", "expected": "python"},
                {"path": "$.tags", "op": "length_ge", "expected": 3},
                {"path": "$.items", "op": "length_ge", "expected": 5},
                {
                    "path": "$.metadata.version",
                    "op": "match_regex",
                    "expected": "^\\d+\\.\\d+\\.\\d+$",
                },
            ],
        }
    )

    engine = Engine()
    result = engine.run_test(test_case)

    # Verify complete workflow executed correctly
    assert result["case_id"] == "complete_workflow_test"
    assert result["status"] == "PASS"
    assert len(result["asserts"]) == 8
    assert all(a["ok"] for a in result["asserts"])
    assert result["result"]["status"] == "completed"


def test_run_test_sync_workflow_with_composition():
    """Test run_test with sync components and assertion composition (all/any/not).

    Verifies DoD #3: Complex sync workflows maintain backward compatibility.
    """
    test_case = Scenario.model_validate(
        {
            "case": {
                "id": "composition_test",
                "description": "Test assertion composition",
            },
            "input": {"test": "data"},
            "run": {"kind": "python", "target": "tests.fixtures.dummy_inference"},
            "asserts": [
                {
                    "op": "",
                    "all": [
                        {"path": "$.status", "op": "equals", "expected": "success"},
                        {"path": "$.count", "op": "equals", "expected": 42},
                    ],
                },
                {
                    "op": "",
                    "any": [
                        {"path": "$.count", "op": "equals", "expected": 42},
                        {"path": "$.count", "op": "equals", "expected": 100},
                    ],
                },
                {
                    "op": "",
                    "not": {"path": "$.status", "op": "equals", "expected": "failure"},
                },
            ],
        }
    )

    engine = Engine()
    result = engine.run_test(test_case)

    assert result["status"] == "PASS"
    assert len(result["asserts"]) == 3
    assert all(a["ok"] for a in result["asserts"])


def test_sync_api_unchanged_interface():
    """Test that sync API method signatures remain unchanged.

    Verifies DoD #6: All existing sync test cases continue to pass without modification.
    This test ensures the method signatures haven't changed.
    """
    engine = Engine()

    # Test run_inference signature
    run_config = RunConfig(kind="python", target="tests.fixtures.dummy_inference")
    input_data = {"test": "data"}
    result = engine.run_inference(run_config, input_data)
    assert isinstance(result, dict)

    # Test eval_assert signature
    rule = AssertRule.model_validate(
        {"path": "$.status", "op": "equals", "expected": "success"}
    )
    ok, message = engine.eval_assert(rule, result)
    assert isinstance(ok, bool)
    assert isinstance(message, str)

    # Test run_test signature
    test_case = Scenario.model_validate(
        {
            "case": {"id": "api_test"},
            "input": {"test": "data"},
            "run": {"kind": "python", "target": "tests.fixtures.dummy_inference"},
            "asserts": [{"path": "$.status", "op": "equals", "expected": "success"}],
        }
    )
    test_result = engine.run_test(test_case)
    assert isinstance(test_result, dict)
    assert "case_id" in test_result
    assert "status" in test_result


def test_sync_methods_callable_in_regular_function():
    """Test sync methods can be called from regular (non-async) functions.

    Verifies DoD #4: Demonstrates sync methods work in typical synchronous code.
    """

    def regular_function_using_engine():
        """A regular synchronous function that uses the engine."""
        engine = Engine()
        run_config = RunConfig(kind="python", target="tests.fixtures.dummy_inference")
        input_data = {"test": "data"}

        # All these calls should work in regular sync function
        inference_result = engine.run_inference(run_config, input_data)

        rule = AssertRule.model_validate(
            {"path": "$.status", "op": "equals", "expected": "success"}
        )
        ok, message = engine.eval_assert(rule, inference_result)

        test_case = Scenario.model_validate(
            {
                "case": {"id": "nested_test"},
                "input": input_data,
                "run": run_config.model_dump(),
                "asserts": [rule.model_dump()],
            }
        )
        test_result = engine.run_test(test_case)

        return inference_result, ok, message, test_result

    # Call regular function
    inference_result, ok, message, test_result = regular_function_using_engine()

    # Verify all operations completed successfully
    assert inference_result == {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }
    assert ok is True
    assert test_result["status"] == "PASS"


def test_sync_methods_no_coroutine_leakage():
    """Test that sync methods do not return coroutines or awaitables.

    Verifies DoD #5: Return types are correct, not async artifacts.
    """
    import inspect

    engine = Engine()

    # Test run_inference doesn't return coroutine
    run_config = RunConfig(kind="python", target="tests.fixtures.dummy_inference")
    result = engine.run_inference(run_config, {})
    assert not inspect.iscoroutine(result)
    assert isinstance(result, dict)

    # Test eval_assert doesn't return coroutine
    rule = AssertRule.model_validate(
        {"path": "$.status", "op": "equals", "expected": "success"}
    )
    result = engine.eval_assert(rule, {"status": "success"})
    assert not inspect.iscoroutine(result)
    assert isinstance(result, tuple)

    # Test run_test doesn't return coroutine
    test_case = Scenario.model_validate(
        {
            "case": {"id": "test"},
            "input": {},
            "run": {"kind": "python", "target": "tests.fixtures.dummy_inference"},
            "asserts": [{"path": "$.status", "op": "equals", "expected": "success"}],
        }
    )
    result = engine.run_test(test_case)
    assert not inspect.iscoroutine(result)
    assert isinstance(result, dict)
