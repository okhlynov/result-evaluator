"""Tests for runtime operators."""

from runtime.operators import (
    op_contains,
    op_equals,
    op_exists,
    op_length_ge,
    op_match_regex,
)


def test_op_equals_str_equal() -> None:
    """Test op_equals with equal strings."""
    result = op_equals("hello", {"expected": "hello"})
    assert result.ok is True
    assert result.message is None
    assert result.got == "hello"


def test_op_equals_str_not_equal() -> None:
    """Test op_equals with different strings."""
    result = op_equals("hello", {"expected": "world"})
    assert result.ok is False
    assert result.message == "Expected world, got hello"
    assert result.got == "hello"


def test_op_equals_dict_equal() -> None:
    """Test op_equals with equal dicts."""
    result = op_equals({"a": 1, "b": 2}, {"expected": {"a": 1, "b": 2}})
    assert result.ok is True
    assert result.message is None
    assert result.got == {"a": 1, "b": 2}


def test_op_equals_dict_not_equal() -> None:
    """Test op_equals with different dicts."""
    result = op_equals({"a": 1}, {"expected": {"b": 2}})
    assert result.ok is False
    assert result.message == "Expected {'b': 2}, got {'a': 1}"
    assert result.got == {"a": 1}


def test_op_equals_int_equal() -> None:
    """Test op_equals with equal ints."""
    result = op_equals(42, {"expected": 42})
    assert result.ok is True
    assert result.message is None
    assert result.got == 42


def test_op_equals_int_not_equal() -> None:
    """Test op_equals with different ints."""
    result = op_equals(42, {"expected": 100})
    assert result.ok is False
    assert result.message == "Expected 100, got 42"
    assert result.got == 42


def test_op_equals_none_equal() -> None:
    """Test op_equals with None values."""
    result = op_equals(None, {"expected": None})
    assert result.ok is True
    assert result.message is None
    assert result.got is None


def test_op_equals_none_not_equal() -> None:
    """Test op_equals with None vs non-None."""
    result = op_equals(None, {"expected": "something"})
    assert result.ok is False
    assert result.message == "Expected something, got None"
    assert result.got is None


def test_op_equals_list_equal() -> None:
    """Test op_equals with equal lists."""
    result = op_equals([1, 2, 3], {"expected": [1, 2, 3]})
    assert result.ok is True
    assert result.message is None
    assert result.got == [1, 2, 3]


def test_op_equals_list_not_equal() -> None:
    """Test op_equals with different lists."""
    result = op_equals([1, 2], {"expected": [3, 4]})
    assert result.ok is False
    assert result.message == "Expected [3, 4], got [1, 2]"
    assert result.got == [1, 2]


def test_op_exists_str() -> None:
    """Test op_exists with non-empty string."""
    result = op_exists("hello", {})
    assert result.ok is True
    assert result.message is None
    assert result.got == "hello"


def test_op_exists_str_empty() -> None:
    """Test op_exists with empty string (should still exist)."""
    result = op_exists("", {})
    assert result.ok is True
    assert result.message is None
    assert result.got == ""


def test_op_exists_dict() -> None:
    """Test op_exists with non-empty dict."""
    result = op_exists({"a": 1}, {})
    assert result.ok is True
    assert result.message is None
    assert result.got == {"a": 1}


def test_op_exists_dict_empty() -> None:
    """Test op_exists with empty dict (should still exist)."""
    result = op_exists({}, {})
    assert result.ok is True
    assert result.message is None
    assert result.got == {}


def test_op_exists_int() -> None:
    """Test op_exists with non-zero int."""
    result = op_exists(42, {})
    assert result.ok is True
    assert result.message is None
    assert result.got == 42


def test_op_exists_int_zero() -> None:
    """Test op_exists with zero (should still exist)."""
    result = op_exists(0, {})
    assert result.ok is True
    assert result.message is None
    assert result.got == 0


def test_op_exists_list() -> None:
    """Test op_exists with non-empty list."""
    result = op_exists([1, 2, 3], {})
    assert result.ok is True
    assert result.message is None
    assert result.got == [1, 2, 3]


def test_op_exists_none() -> None:
    """Test op_exists with None (should not exist)."""
    result = op_exists(None, {})
    assert result.ok is False
    assert result.message == "Selection is empty or None"
    assert result.got is None


def test_op_exists_list_empty() -> None:
    """Test op_exists with empty list (should not exist)."""
    result = op_exists([], {})
    assert result.ok is False
    assert result.message == "Selection is empty or None"
    assert result.got == []


def test_op_contains_str_found() -> None:
    """Test op_contains with substring found in string."""
    result = op_contains("hello world", {"expected": "world"})
    assert result.ok is True
    assert result.message is None
    assert result.got == "hello world"


def test_op_contains_str_not_found() -> None:
    """Test op_contains with substring not found in string."""
    result = op_contains("hello world", {"expected": "python"})
    assert result.ok is False
    assert result.message == "'python' not found in hello world"
    assert result.got == "hello world"


def test_op_contains_list_found() -> None:
    """Test op_contains with item found in list."""
    result = op_contains([1, 2, 3], {"expected": 2})
    assert result.ok is True
    assert result.message is None
    assert result.got == [1, 2, 3]


def test_op_contains_list_not_found() -> None:
    """Test op_contains with item not found in list."""
    result = op_contains([1, 2, 3], {"expected": 5})
    assert result.ok is False
    assert result.message == "'5' not found in [1, 2, 3]"
    assert result.got == [1, 2, 3]


def test_op_contains_invalid_type() -> None:
    """Test op_contains with invalid type (dict)."""
    result = op_contains({"a": 1}, {"expected": "a"})  # type: ignore[arg-type]
    assert result.ok is False
    assert result.message == "Cannot check 'contains' for <class 'dict'>"
    assert result.got == {"a": 1}


def test_op_length_ge_str_sufficient() -> None:
    """Test op_length_ge with string length >= expected."""
    result = op_length_ge("hello", {"expected": 3})
    assert result.ok is True
    assert result.message is None
    assert result.got == 5


def test_op_length_ge_str_insufficient() -> None:
    """Test op_length_ge with string length < expected."""
    result = op_length_ge("hi", {"expected": 5})
    assert result.ok is False
    assert result.message == "Length 2 < 5"
    assert result.got == 2


def test_op_length_ge_list_sufficient() -> None:
    """Test op_length_ge with list length >= expected."""
    result = op_length_ge([1, 2, 3, 4], {"expected": 3})
    assert result.ok is True
    assert result.message is None
    assert result.got == 4


def test_op_length_ge_list_insufficient() -> None:
    """Test op_length_ge with list length < expected."""
    result = op_length_ge([1, 2], {"expected": 5})
    assert result.ok is False
    assert result.message == "Length 2 < 5"
    assert result.got == 2


def test_op_length_ge_dict_sufficient() -> None:
    """Test op_length_ge with dict length >= expected."""
    result = op_length_ge({"a": 1, "b": 2, "c": 3}, {"expected": 2})
    assert result.ok is True
    assert result.message is None
    assert result.got == 3


def test_op_length_ge_equal_boundary() -> None:
    """Test op_length_ge with exact length match (boundary case)."""
    result = op_length_ge([1, 2, 3], {"expected": 3})
    assert result.ok is True
    assert result.message is None
    assert result.got == 3


def test_op_length_ge_invalid_type() -> None:
    """Test op_length_ge with invalid type (int has no __len__)."""
    result = op_length_ge(42, {"expected": 5})  # type: ignore[arg-type]
    assert result.ok is False
    assert result.message == "Selection has no length: <class 'int'>"
    assert result.got == 42


def test_op_match_regex_digits_match() -> None:
    """Test op_match_regex with string matching digits pattern."""
    result = op_match_regex("12345", {"expected": r"^\d+$"})
    assert result.ok is True
    assert result.message is None
    assert result.got == "12345"


def test_op_match_regex_digits_no_match() -> None:
    """Test op_match_regex with string not matching digits pattern."""
    result = op_match_regex("abc123", {"expected": r"^\d+$"})
    assert result.ok is False
    assert result.message == "'abc123' doesn't match pattern '^\\d+$'"
    assert result.got == "abc123"


def test_op_match_regex_none_input() -> None:
    """Test op_match_regex with None input (should raise TypeError)."""
    try:
        op_match_regex(None, {"expected": r"^\d+$"})  # type: ignore[arg-type]
        raise AssertionError("Expected TypeError to be raised")
    except TypeError:
        pass  # Expected behavior


def test_op_match_regex_dict_input() -> None:
    """Test op_match_regex with dict input (should raise TypeError)."""
    try:
        op_match_regex({"key": "value"}, {"expected": r"^\d+$"})  # type: ignore[arg-type]
        raise AssertionError("Expected TypeError to be raised")
    except TypeError:
        pass  # Expected behavior
