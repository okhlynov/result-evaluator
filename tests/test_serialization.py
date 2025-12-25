"""Tests for serialization of selection and ground truth data.

These tests verify that selection and ground truth data is correctly serialized
to JSON with proper wrapping behavior:
- Dicts: serialized as-is (no wrapper)
- Non-dicts (lists, scalars, None): wrapped in {"value": ...}
- Ground truth lists: special case, treated as complex type (no wrapper)
"""

import json
from typing import Any

from result_evaluator.runtime.operators import (
    _serialize_ground_truth,
    _serialize_selection,
)


class TestSerializeSelection:
    """Tests for _serialize_selection function."""

    def test_serialize_dict_no_wrapping(self) -> None:
        """Test dict selection is serialized without "value" wrapper."""
        selection = {"foo": "bar"}
        result = _serialize_selection(selection)

        # Verify valid JSON
        parsed = json.loads(result)

        # Verify dict is not wrapped
        assert parsed == {"foo": "bar"}
        assert "value" not in parsed

    def test_serialize_dict_with_multiple_keys(self) -> None:
        """Test dict with multiple keys is serialized correctly without wrapping."""
        selection = {"key1": "value1", "key2": "value2", "key3": 42}
        result = _serialize_selection(selection)

        parsed = json.loads(result)
        assert parsed == selection
        assert "value" not in parsed

    def test_serialize_list_wrapped(self) -> None:
        """Test list selection is wrapped in {"value": [...]}.

        Lists should be wrapped to distinguish them from dict-based structures
        and to preserve type information for the LLM.
        """
        selection = [1, 2, 3]
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        # Verify list is wrapped in "value" key
        assert "value" in parsed
        assert parsed["value"] == [1, 2, 3]
        assert len(parsed) == 1  # Only "value" key should exist

    def test_serialize_list_with_strings(self) -> None:
        """Test list with string elements is wrapped correctly."""
        selection = ["a", "b", "c"]
        result = _serialize_selection(selection)

        parsed = json.loads(result)
        assert parsed == {"value": ["a", "b", "c"]}

    def test_serialize_string_wrapped(self) -> None:
        """Test string selection is wrapped in {"value": "..."}.

        Scalar strings are wrapped to preserve type information and maintain
        consistency with other scalar types.
        """
        selection = "hello"
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        # Verify string is wrapped in "value" key
        assert "value" in parsed
        assert parsed["value"] == "hello"
        assert len(parsed) == 1

    def test_serialize_int_wrapped(self) -> None:
        """Test integer selection is wrapped in {"value": ...}."""
        selection = 42
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": 42}

    def test_serialize_float_wrapped(self) -> None:
        """Test float selection is wrapped in {"value": ...}."""
        selection = 3.14
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": 3.14}

    def test_serialize_bool_wrapped(self) -> None:
        """Test boolean selection is wrapped in {"value": ...}."""
        selection = True
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": True}

        selection = False
        result = _serialize_selection(selection)
        parsed = json.loads(result)
        assert parsed == {"value": False}

    def test_serialize_none_wrapped(self) -> None:
        """Test None selection is wrapped in {"value": null}.

        None is a special case that needs to be wrapped to distinguish it
        from the absence of data.
        """
        selection = None
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        # Verify None is wrapped in "value" key as null
        assert "value" in parsed
        assert parsed["value"] is None
        assert len(parsed) == 1

    def test_serialize_nested_structure_list_of_dicts(self) -> None:
        """Test nested list of dicts is wrapped in {"value": [...]}.

        Complex nested structures (non-dicts at top level) should be wrapped
        to maintain consistency.
        """
        selection = [{"a": 1}, {"b": 2}]
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        # List should be wrapped, but dicts inside should remain as-is
        assert "value" in parsed
        assert parsed["value"] == [{"a": 1}, {"b": 2}]

    def test_serialize_dict_with_nested_list(self) -> None:
        """Test dict containing nested list is not wrapped."""
        selection = {"data": [1, 2, 3]}
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        # Dict should not be wrapped
        assert parsed == {"data": [1, 2, 3]}
        assert "value" not in parsed

    def test_serialize_dict_with_nested_dict(self) -> None:
        """Test dict containing nested dict is not wrapped."""
        selection = {"outer": {"inner": "value"}}
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"outer": {"inner": "value"}}
        assert "value" not in parsed

    def test_serialize_empty_dict(self) -> None:
        """Test empty dict is serialized without wrapping."""
        selection = {}
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {}
        assert "value" not in parsed

    def test_serialize_empty_list(self) -> None:
        """Test empty list is wrapped in {"value": []}."""
        selection = []
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": []}


class TestSerializeGroundTruth:
    """Tests for _serialize_ground_truth function.

    Note: Ground truth has different behavior than selection for lists.
    Lists are treated as complex types (not wrapped).
    """

    def test_serialize_dict_no_wrapping(self) -> None:
        """Test dict ground truth is serialized without "value" wrapper."""
        ground_truth = {"foo": "bar"}
        result = _serialize_ground_truth(ground_truth)

        parsed = json.loads(result)

        assert parsed == {"foo": "bar"}
        assert "value" not in parsed

    def test_serialize_list_no_wrapping(self) -> None:
        """Test list ground truth is NOT wrapped (differs from selection).

        Lists are treated as complex types in ground truth, so they are
        serialized as-is without wrapping.
        """
        ground_truth = [1, 2, 3]
        result = _serialize_ground_truth(ground_truth)

        parsed = json.loads(result)

        # Lists should NOT be wrapped in ground truth
        assert parsed == [1, 2, 3]
        assert not isinstance(parsed, dict) or "value" not in parsed

    def test_serialize_string_wrapped(self) -> None:
        """Test string ground truth is wrapped in {"value": "..."}."""
        ground_truth = "expected"
        result = _serialize_ground_truth(ground_truth)

        parsed = json.loads(result)

        assert parsed == {"value": "expected"}

    def test_serialize_int_wrapped(self) -> None:
        """Test integer ground truth is wrapped in {"value": ...}."""
        ground_truth = 100
        result = _serialize_ground_truth(ground_truth)

        parsed = json.loads(result)

        assert parsed == {"value": 100}

    def test_serialize_none_wrapped(self) -> None:
        """Test None ground truth is wrapped in {"value": null}."""
        ground_truth = None
        result = _serialize_ground_truth(ground_truth)

        parsed = json.loads(result)

        assert parsed == {"value": None}

    def test_serialize_list_of_dicts_no_wrapping(self) -> None:
        """Test list of dicts ground truth is not wrapped."""
        ground_truth = [{"a": 1}, {"b": 2}]
        result = _serialize_ground_truth(ground_truth)

        parsed = json.loads(result)

        # Lists should not be wrapped in ground truth
        assert parsed == [{"a": 1}, {"b": 2}]
        assert isinstance(parsed, list)

    def test_serialize_dict_with_nested_list(self) -> None:
        """Test dict containing list ground truth is not wrapped."""
        ground_truth = {"data": [1, 2, 3]}
        result = _serialize_ground_truth(ground_truth)

        parsed = json.loads(result)

        assert parsed == {"data": [1, 2, 3]}
        assert "value" not in parsed


class TestSerializationEdgeCases:
    """Tests for edge cases in serialization."""

    def test_serialize_unicode_string(self) -> None:
        """Test serialization with Unicode characters."""
        selection = "Hello 世界 🌍"
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": "Hello 世界 🌍"}

    def test_serialize_special_characters_in_dict(self) -> None:
        """Test serialization of dict with special characters."""
        selection = {"key": 'value with "quotes" and \\backslash'}
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == selection

    def test_serialize_large_number(self) -> None:
        """Test serialization of large numbers."""
        selection = 999999999999999
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": 999999999999999}

    def test_serialize_negative_number(self) -> None:
        """Test serialization of negative numbers."""
        selection = -42
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": -42}

    def test_serialize_scientific_notation(self) -> None:
        """Test serialization of numbers in scientific notation."""
        selection = 1.23e-4
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": 1.23e-4}

    def test_serialize_zero(self) -> None:
        """Test serialization of zero."""
        selection = 0
        result = _serialize_selection(selection)

        parsed = json.loads(result)

        assert parsed == {"value": 0}


class TestSerializationConsistency:
    """Tests to ensure serialization is consistent and JSON is valid."""

    def test_serialized_output_is_valid_json(self) -> None:
        """Verify all serialized outputs are valid JSON."""
        test_cases: list[Any] = [
            {"key": "value"},
            [1, 2, 3],
            "string",
            42,
            3.14,
            True,
            None,
            [{"a": 1}],
        ]

        for case in test_cases:
            result = _serialize_selection(case)
            # Should not raise - valid JSON
            parsed = json.loads(result)
            assert parsed is not None

    def test_serialization_roundtrip(self) -> None:
        """Test that serialized data can be deserialized back."""
        test_cases: list[tuple[Any, Any]] = [
            ({"key": "value"}, {"key": "value"}),  # Dict unchanged
            ([1, 2, 3], {"value": [1, 2, 3]}),  # List wrapped
            ("hello", {"value": "hello"}),  # String wrapped
            (42, {"value": 42}),  # Int wrapped
            (None, {"value": None}),  # None wrapped
        ]

        for input_data, expected_parsed in test_cases:
            result = _serialize_selection(input_data)
            parsed = json.loads(result)
            assert parsed == expected_parsed
