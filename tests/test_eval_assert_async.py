"""Unit tests for eval_assert_async with sync operators.

This module tests the async eval_assert_async() method with synchronous operators,
verifying that sync operators are correctly called (not awaited) and that recursive
combinator logic (all_, any_, not_) works correctly in async context.
"""

import pytest

from result_evaluator.dsl.models import AssertRule
from result_evaluator.runtime.engine import Engine


@pytest.mark.asyncio
async def test_eval_assert_async_sync_operator_equals() -> None:
    """Test that sync operator (op_equals) is correctly called without await."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {"path": "$.name", "op": "equals", "expected": "Alice"}
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "OK"


@pytest.mark.asyncio
async def test_eval_assert_async_sync_operator_fails() -> None:
    """Test that sync operator returns expected failure result."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {"path": "$.name", "op": "equals", "expected": "Bob"}
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is False
    assert "Expected Bob, got Alice" in message


@pytest.mark.asyncio
async def test_eval_assert_async_all_combinator_with_sync_operators() -> None:
    """Test that all_ combinator executes multiple sync operators correctly in async context."""
    document = {"name": "Alice", "age": 30, "city": "New York"}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {"path": "$.age", "op": "equals", "expected": 30},
                {"path": "$.city", "op": "equals", "expected": "New York"},
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "All checks passed"


@pytest.mark.asyncio
async def test_eval_assert_async_all_combinator_one_fails() -> None:
    """Test that all_ combinator fails when one sync operator fails."""
    document = {"name": "Alice", "age": 30, "city": "Boston"}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {"path": "$.age", "op": "equals", "expected": 30},
                {"path": "$.city", "op": "equals", "expected": "New York"},
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is False
    assert "Expected New York, got Boston" in message


@pytest.mark.asyncio
async def test_eval_assert_async_any_combinator_with_sync_operators() -> None:
    """Test that any_ combinator executes multiple sync operators correctly in async context."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "any": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 30},
                {"path": "$.name", "op": "equals", "expected": "Charlie"},
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "At least one check passed"


@pytest.mark.asyncio
async def test_eval_assert_async_any_combinator_all_fail() -> None:
    """Test that any_ combinator fails when all sync operators fail."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "any": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 25},
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is False
    assert "None passed" in message
    assert "Expected Bob, got Alice" in message
    assert "Expected 25, got 30" in message


@pytest.mark.asyncio
async def test_eval_assert_async_not_combinator_with_sync_operator() -> None:
    """Test that not_ combinator works correctly with sync operator in async context."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "not": {"path": "$.name", "op": "equals", "expected": "Bob"},
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "NOT passed"


@pytest.mark.asyncio
async def test_eval_assert_async_not_combinator_fails() -> None:
    """Test that not_ combinator fails when inner sync operator passes."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "not": {"path": "$.name", "op": "equals", "expected": "Alice"},
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is False
    assert "NOT failed" in message


@pytest.mark.asyncio
async def test_eval_assert_async_nested_combinators_all_within_all() -> None:
    """Test nested combinators: all_ containing all_ with sync operators."""
    document = {
        "user": {"name": "Alice", "age": 30},
        "address": {"city": "New York", "zip": "10001"},
    }

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.user.name", "op": "equals", "expected": "Alice"},
                {
                    "op": "",
                    "all": [
                        {"path": "$.address.city", "op": "equals", "expected": "New York"},
                        {"path": "$.address.zip", "op": "equals", "expected": "10001"},
                    ],
                },
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "All checks passed"


@pytest.mark.asyncio
async def test_eval_assert_async_nested_combinators_all_within_any() -> None:
    """Test nested combinators: all_ within any_ with sync operators."""
    document = {"name": "Alice", "age": 30, "city": "Boston"}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "any": [
                {
                    "op": "",
                    "all": [
                        {"path": "$.name", "op": "equals", "expected": "Alice"},
                        {"path": "$.age", "op": "equals", "expected": 30},
                    ],
                },
                {"path": "$.city", "op": "equals", "expected": "New York"},
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "At least one check passed"


@pytest.mark.asyncio
async def test_eval_assert_async_nested_combinators_any_within_all() -> None:
    """Test nested combinators: any_ within all_ with sync operators."""
    document = {"name": "Alice", "status": "active", "role": "admin"}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {
                    "op": "",
                    "any": [
                        {"path": "$.role", "op": "equals", "expected": "admin"},
                        {"path": "$.role", "op": "equals", "expected": "superuser"},
                    ],
                },
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "All checks passed"


@pytest.mark.asyncio
async def test_eval_assert_async_complex_nested_combinators() -> None:
    """Test deeply nested combinators: not_ within any_ within all_."""
    document = {"name": "Alice", "age": 30, "status": "active"}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {
                    "op": "",
                    "any": [
                        {"path": "$.age", "op": "equals", "expected": 25},
                        {
                            "op": "",
                            "not": {"path": "$.status", "op": "equals", "expected": "inactive"},
                        },
                    ],
                },
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "All checks passed"


@pytest.mark.asyncio
async def test_eval_assert_async_operator_error_propagates() -> None:
    """Test that errors from sync operators propagate correctly in async context."""
    document = {"name": "Alice", "age": 30}

    # Test with unknown operator
    rule = AssertRule.model_validate(
        {"path": "$.name", "op": "unknown_operator", "expected": "Alice"}
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is False
    assert "Unknown operator: unknown_operator" in message


@pytest.mark.asyncio
async def test_eval_assert_async_missing_path() -> None:
    """Test that missing path is handled correctly in async context."""
    document = {"name": "Alice"}

    rule = AssertRule.model_validate({"op": "equals", "expected": "Alice"})

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is False
    assert "No path specified" in message


@pytest.mark.asyncio
async def test_eval_assert_async_multiple_operators() -> None:
    """Test eval_assert_async with various sync operators (contains, length_ge, match_regex)."""
    document = {
        "tags": ["python", "testing", "async"],
        "message": "Processing complete with code-123",
        "items": [1, 2, 3, 4, 5],
    }

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.tags", "op": "contains", "expected": "python"},
                {"path": "$.tags", "op": "length_ge", "expected": 3},
                {"path": "$.message", "op": "match_regex", "expected": "^Processing.*code-\\d+$"},
                {"path": "$.items", "op": "length_ge", "expected": 5},
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "All checks passed"


@pytest.mark.asyncio
async def test_eval_assert_async_not_contains_operator() -> None:
    """Test eval_assert_async with not_contains sync operator."""
    document = {"tags": ["python", "testing"], "message": "Processing complete"}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.tags", "op": "not_contains", "expected": "production"},
                {"path": "$.message", "op": "not_contains", "expected": "error"},
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "All checks passed"


@pytest.mark.asyncio
async def test_eval_assert_async_exists_operator() -> None:
    """Test eval_assert_async with exists sync operator."""
    document = {"name": "Alice", "age": None, "tags": []}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "exists"},
                {
                    "op": "",
                    "not": {"path": "$.age", "op": "exists"},
                },
                {
                    "op": "",
                    "not": {"path": "$.tags", "op": "exists"},
                },
            ],
        }
    )

    engine = Engine()
    ok, message = await engine.eval_assert_async(rule, document)

    assert ok is True
    assert message == "All checks passed"
