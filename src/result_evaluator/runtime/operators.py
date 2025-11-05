import re
from collections.abc import Callable, Sized
from dataclasses import dataclass
from typing import Any


@dataclass
class OpResult:
    """Результат выполнения оператора"""

    ok: bool
    message: str | None = None
    got: Any = None


Operator = Callable[[Any, dict[str, Any]], OpResult]


def op_exists(selection: Any, _: dict[str, Any]) -> OpResult:
    """Проверяет что значение существует и не пустое"""
    is_empty = selection is None or (
        isinstance(selection, list) and len(selection) == 0
    )
    ok = not is_empty
    return OpResult(
        ok=ok, message="Selection is empty or None" if not ok else None, got=selection
    )

def op_equals(selection: Any, params: dict[str, Any]) -> OpResult:
    """Строгое равенство"""
    expected = params["expected"]
    ok = selection == expected
    return OpResult(
        ok=ok,
        message=f"Expected {expected}, got {selection}" if not ok else None,
        got=selection,
    )


def op_contains(selection: str | list[Any], params: dict[str, Any]) -> OpResult:
    """Проверяет вхождение элемента"""
    expected = params["expected"]

    if isinstance(selection, str):
        ok = expected in selection
    elif isinstance(selection, list):
        ok = expected in selection
    else:
        return OpResult(
            False, f"Cannot check 'contains' for {type(selection)}", selection
        )

    return OpResult(
        ok=ok,
        message=f"'{expected}' not found in {selection}" if not ok else None,
        got=selection,
    )
    
def op_not_contains(selection: str | list[Any], params: dict[str, Any]) -> OpResult:
    """Проверяет отсутствие элемента"""
    expected = params["expected"]

    if isinstance(selection, str):
        ok = expected not in selection
    elif isinstance(selection, list):
        ok = expected not in selection
    else:
        return OpResult(
            False, f"Cannot check 'contains' for {type(selection)}", selection
        )

    return OpResult(
        ok=ok,
        message=f"'{expected}' present in {selection}" if not ok else None,
        got=selection,
    )


def op_length_ge(selection: Sized, params: dict[str, Any]) -> OpResult:
    """Проверяет что длина >= ожидаемой"""
    expected = params["expected"]

    if not hasattr(selection, "__len__"):
        return OpResult(False, f"Selection has no length: {type(selection)}", selection)

    length = len(selection)
    ok = length >= expected
    return OpResult(
        ok=ok, message=f"Length {length} < {expected}" if not ok else None, got=length
    )


def op_match_regex(selection: str, params: dict[str, Any]) -> OpResult:
    """Проверяет соответствие регулярному выражению"""
    pattern = params["expected"]

    ok = re.match(pattern, selection) is not None
    return OpResult(
        ok=ok,
        message=f"'{selection}' doesn't match pattern '{pattern}'" if not ok else None,
        got=selection,
    )


def op_sequence_in_order(selection: Any, params: dict[str, Any]) -> OpResult:
    """Проверяет что список строк содержит ожидаемые элементы в заданном порядке"""
    # Validate required parameters
    if "expected" not in params:
        return OpResult(False, "Missing required parameter 'expected'", selection)

    expected_obj = params["expected"]

    # Auto-convert single items to list
    if not isinstance(selection, list):
        selection = [str(selection)]

    # Validate expected is a dict with 'data' and 'limit' fields
    if not isinstance(expected_obj, dict):
        return OpResult(
            False,
            f"Parameter 'expected' must be a dict with 'data' and 'limit' fields, got {type(expected_obj).__name__}",
            selection,
        )

    # Validate 'data' field exists
    if "data" not in expected_obj:
        return OpResult(
            False,
            "Parameter 'expected' must contain 'data' field",
            selection,
        )

    # Validate 'limit' field exists
    if "limit" not in expected_obj:
        return OpResult(
            False,
            "Parameter 'expected' must contain 'limit' field",
            selection,
        )

    expected = expected_obj["data"]
    limit = expected_obj["limit"]

    # Validate expected data is a list
    if not isinstance(expected, list):
        return OpResult(
            False,
            f"Parameter 'expected.data' must be a list, got {type(expected).__name__}",
            selection,
        )

    # Validate limit is a positive integer
    if not isinstance(limit, int) or limit <= 0:
        return OpResult(
            False,
            f"Parameter 'expected.limit' must be a positive integer, got {limit}",
            selection,
        )

    # Empty expected list should pass
    if len(expected) == 0:
        return OpResult(True, None, selection)

    # Validate all items in expected are strings
    for item in expected:
        if not isinstance(item, str):
            return OpResult(
                False,
                f"All items in 'expected.data' must be strings, found {type(item).__name__}",
                selection,
            )

    # Validate all items in selection are strings
    for item in selection:
        if not isinstance(item, str):
            return OpResult(
                False,
                f"Selection must be a list of strings, found {type(item).__name__}",
                selection,
            )

    # Get top N items from selection
    items = selection[:limit]

    # Check if expected items appear in order
    exp_idx = 0 # номер ожидаемого элемента, которого ищем
    for item in items:
        if exp_idx < len(expected) and item == expected[exp_idx]: 
            exp_idx += 1 # элемент встретился, берем следующий

    # Success if all expected items were found in order
    ok = exp_idx == len(expected)

    if not ok:
        if exp_idx == 0:
            message = (
                f"Expected item '{expected[0]}' not found within first {limit} items"
            )
        else:
            message = f"Expected item '{expected[exp_idx]}' not found in order within first {limit} items"
    else:
        message = None

    return OpResult(ok=ok, message=message, got=selection)


# Реестр всех операторов
OPERATORS: dict[str, Operator] = {
    "exists": op_exists,
    "equals": op_equals,
    "contains": op_contains,
    "length_ge": op_length_ge,
    "match_regex": op_match_regex,
    "sequence_in_order": op_sequence_in_order,
}
