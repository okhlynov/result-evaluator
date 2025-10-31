"""Tests for runtime operators."""

from result_evaluator.runtime.operators import (
    op_contains,
    op_not_contains,
    op_equals,
    op_exists,
    op_length_ge,
    op_match_regex,
    op_sequence_in_order,
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


def test_op_not_contains_str_found() -> None:
    """Test op_contains with substring found in string."""
    result = op_not_contains("hello world", {"expected": "world"})
    assert result.ok is False
    assert result.message == "'world' present in hello world"
    assert result.got == "hello world"


def test_op_not_contains_str_not_found() -> None:
    """Test op_contains with substring not found in string."""
    result = op_not_contains("hello world", {"expected": "python"})
    assert result.ok is True
    assert result.message is None
    assert result.got == "hello world"


def test_op_not_contains_list_found() -> None:
    """Test op_contains with item found in list."""
    result = op_not_contains([1, 2, 3], {"expected": 2})
    assert result.ok is True
    assert result.message == "'2' present in [1, 2, 3]"
    assert result.got == [1, 2, 3]


def test_op_not_contains_list_not_found() -> None:
    """Test op_contains with item not found in list."""
    result = op_not_contains([1, 2, 3], {"expected": 5})
    assert result.ok is True
    assert result.message is None
    assert result.got == [1, 2, 3]


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


def test_op_sequence_in_order_basic_match() -> None:
    """Test op_sequence_in_order with basic sequence match with interleaved items."""
    result = op_sequence_in_order(
        ["A", "X", "B", "Y", "Z", "C", "D"], {"expected": {"data": ["A", "B", "C"], "limit": 7}}
    )
    assert result.ok is True
    assert result.message is None
    assert result.got == ["A", "X", "B", "Y", "Z", "C", "D"]


def test_op_sequence_in_order_out_of_order() -> None:
    """Test op_sequence_in_order with items out of order."""
    result = op_sequence_in_order(
        ["A", "C", "B"], {"expected": {"data": ["A", "B", "C"], "limit": 3}}
    )
    assert result.ok is False
    # When "A" is found, we look for "B" next, but "C" comes before "B"
    # So we find "B" second (skipping "C"), then we can't find "C" after that
    assert "Expected item 'C' not found in order within first 3 items" in result.message
    assert result.got == ["A", "C", "B"]


def test_op_sequence_in_order_missing_item() -> None:
    """Test op_sequence_in_order with missing item within limit."""
    result = op_sequence_in_order(
        ["A", "B", "X", "Y", "Z"], {"expected": {"data": ["A", "B", "C"], "limit": 4}}
    )
    assert result.ok is False
    assert "Expected item 'C' not found in order within first 4 items" in result.message
    assert result.got == ["A", "B", "X", "Y", "Z"]


def test_op_sequence_in_order_limit_smaller_than_selection() -> None:
    """Test op_sequence_in_order with limit smaller than selection."""
    result = op_sequence_in_order(
        ["A", "B", "C", "D", "E"], {"expected": {"data": ["A", "B"], "limit": 3}}
    )
    assert result.ok is True
    assert result.message is None
    assert result.got == ["A", "B", "C", "D", "E"]


def test_op_sequence_in_order_limit_larger_than_selection() -> None:
    """Test op_sequence_in_order with limit larger than selection."""
    result = op_sequence_in_order(
        ["A", "B", "C"], {"expected": {"data": ["A", "B"], "limit": 100}}
    )
    assert result.ok is True
    assert result.message is None
    assert result.got == ["A", "B", "C"]


def test_op_sequence_in_order_empty_expected() -> None:
    """Test op_sequence_in_order with empty expected list (should pass)."""
    result = op_sequence_in_order(["A", "B", "C"], {"expected": {"data": [], "limit": 3}})
    assert result.ok is True
    assert result.message is None
    assert result.got == ["A", "B", "C"]


def test_op_sequence_in_order_empty_selection() -> None:
    """Test op_sequence_in_order with empty selection and non-empty expected."""
    result = op_sequence_in_order([], {"expected": {"data": ["A", "B"], "limit": 10}})
    assert result.ok is False
    assert "Expected item 'A' not found within first 10 items" in result.message
    assert result.got == []


def test_op_sequence_in_order_single_string_selection() -> None:
    """Test op_sequence_in_order with single string (auto-converts to list)."""
    result = op_sequence_in_order("A", {"expected": {"data": ["A"], "limit": 5}})
    assert result.ok is True
    assert result.message is None
    assert result.got == ["A"]


def test_op_sequence_in_order_single_int_selection() -> None:
    """Test op_sequence_in_order with single int (auto-converts to list)."""
    result = op_sequence_in_order(123, {"expected": {"data": ["123"], "limit": 5}})
    assert result.ok is True    


def test_op_sequence_in_order_expected_not_list() -> None:
    """Test op_sequence_in_order with expected not a dict."""
    result = op_sequence_in_order(["A", "B"], {"expected": "not a dict"})
    assert result.ok is False
    assert result.message == "Parameter 'expected' must be a dict with 'data' and 'limit' fields, got str"
    assert result.got == ["A", "B"]


def test_op_sequence_in_order_non_string_in_selection() -> None:
    """Test op_sequence_in_order with non-string items in selection."""
    result = op_sequence_in_order([1, 2, 3], {"expected": {"data": ["A"], "limit": 3}})
    assert result.ok is False
    assert "Selection must be a list of strings, found int" in result.message
    assert result.got == [1, 2, 3]


def test_op_sequence_in_order_non_string_in_expected() -> None:
    """Test op_sequence_in_order with non-string items in expected."""
    result = op_sequence_in_order(["A", "B"], {"expected": {"data": [1, 2], "limit": 3}})
    assert result.ok is False
    assert "All items in 'expected.data' must be strings, found int" in result.message
    assert result.got == ["A", "B"]


def test_op_sequence_in_order_negative_limit() -> None:
    """Test op_sequence_in_order with negative limit."""
    result = op_sequence_in_order(["A", "B"], {"expected": {"data": ["A"], "limit": -1}})
    assert result.ok is False
    assert "Parameter 'expected.limit' must be a positive integer, got -1" in result.message
    assert result.got == ["A", "B"]


def test_op_sequence_in_order_zero_limit() -> None:
    """Test op_sequence_in_order with zero limit."""
    result = op_sequence_in_order(["A", "B"], {"expected": {"data": ["A"], "limit": 0}})
    assert result.ok is False
    assert "Parameter 'expected.limit' must be a positive integer, got 0" in result.message
    assert result.got == ["A", "B"]


def test_op_sequence_in_order_single_item() -> None:
    """Test op_sequence_in_order with single expected item found."""
    result = op_sequence_in_order(["X", "A", "Y"], {"expected": {"data": ["A"], "limit": 3}})
    assert result.ok is True
    assert result.message is None
    assert result.got == ["X", "A", "Y"]


def test_op_sequence_in_order_duplicate_items() -> None:
    """Test op_sequence_in_order with duplicate items in expected list."""
    result = op_sequence_in_order(
        ["A", "X", "A", "Y", "A"], {"expected": {"data": ["A", "A", "A"], "limit": 5}}
    )
    assert result.ok is True
    assert result.message is None
    assert result.got == ["A", "X", "A", "Y", "A"]


def test_op_sequence_in_order_missing_expected_param() -> None:
    """Test op_sequence_in_order with missing 'expected' parameter."""
    result = op_sequence_in_order(["A", "B"], {})
    assert result.ok is False
    assert result.message == "Missing required parameter 'expected'"
    assert result.got == ["A", "B"]


def test_op_sequence_in_order_missing_limit_param() -> None:
    """Test op_sequence_in_order with missing 'limit' field in expected."""
    result = op_sequence_in_order(["A", "B"], {"expected": {"data": ["A"]}})
    assert result.ok is False
    assert result.message == "Parameter 'expected' must contain 'limit' field"
    assert result.got == ["A", "B"]
