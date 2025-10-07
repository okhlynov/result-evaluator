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


# Реестр всех операторов
OPERATORS: dict[str, Operator] = {
    "exists": op_exists,
    "equals": op_equals,
    "contains": op_contains,
    "length_ge": op_length_ge,
    "match_regex": op_match_regex,
}
