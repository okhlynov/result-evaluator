"""Tests for runtime TestEngine.eval_assert method."""

from dsl.models import AssertRule
from runtime.engine import TestEngine


def test_eval_assert_all_success() -> None:
    """Test eval_assert with all composition where all assertions pass."""
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

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "All checks passed"


def test_eval_assert_all_fail_one() -> None:
    """Test eval_assert with all composition where one assertion fails."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 30},
            ],
        }
    )

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "Expected Bob, got Alice" in message


def test_eval_assert_all_fail_both() -> None:
    """Test eval_assert with all composition where both assertions fail."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 25},
            ],
        }
    )

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "Expected Bob, got Alice" in message
    assert "Expected 25, got 30" in message


def test_eval_assert_any_success() -> None:
    """Test eval_assert with any composition where at least one assertion passes."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "any": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 30},
            ],
        }
    )

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "At least one check passed"


def test_eval_assert_any_fail() -> None:
    """Test eval_assert with any composition where all assertions fail."""
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

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "None passed" in message
    assert "Expected Bob, got Alice" in message
    assert "Expected 25, got 30" in message


def test_eval_assert_not_success() -> None:
    """Test eval_assert with not composition where inner assertion fails."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "not": {"path": "$.name", "op": "equals", "expected": "Bob"},
        }
    )

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "NOT passed"


def test_eval_assert_not_fail() -> None:
    """Test eval_assert with not composition where inner assertion passes."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "not": {"path": "$.name", "op": "equals", "expected": "Alice"},
        }
    )

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "NOT failed" in message


def test_eval_assert_all_complex_success() -> None:
    """Test eval_assert with complex all composition on nested document."""
    document = {
        "name": "Alice",
        "age": 30,
        "address": {"city": "New York", "zip": "10001"},
        "tags": ["developer", "python", "testing"],
    }

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {"path": "$.age", "op": "equals", "expected": 30},
                {"path": "$.address.city", "op": "equals", "expected": "New York"},
                {"path": "$.address.zip", "op": "equals", "expected": "10001"},
                {"path": "$.tags", "op": "contains", "expected": "python"},
                {"path": "$.tags", "op": "length_ge", "expected": 3},
            ],
        }
    )

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "All checks passed"


def test_eval_assert_all_complex_fail() -> None:
    """Test eval_assert with complex all composition where one nested check fails."""
    document = {
        "name": "Alice",
        "age": 30,
        "address": {"city": "New York", "zip": "10001"},
        "tags": ["developer", "python", "testing"],
    }

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {"path": "$.age", "op": "equals", "expected": 30},
                {"path": "$.address.city", "op": "equals", "expected": "Boston"},
                {"path": "$.tags", "op": "contains", "expected": "python"},
            ],
        }
    )

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "Expected Boston, got New York" in message


def test_eval_assert_all_nested_composition() -> None:
    """Test eval_assert with nested all compositions (all within all)."""
    document = {
        "name": "Alice",
        "age": 30,
        "address": {"city": "New York", "zip": "10001"},
    }

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {
                    "op": "",
                    "all": [
                        {
                            "path": "$.address.city",
                            "op": "equals",
                            "expected": "New York",
                        },
                        {"path": "$.address.zip", "op": "equals", "expected": "10001"},
                    ],
                },
            ],
        }
    )

    engine = TestEngine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "All checks passed"
