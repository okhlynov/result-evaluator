import json
import logging
import re
from collections.abc import Callable, Sized
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .config import load_llm_config
from .llm import call_llm

logger = logging.getLogger(__name__)


class LLMJudgeResponse(BaseModel):
    """Pydantic model for structured LLM response with verdict and reasoning."""

    verdict: bool
    reasoning: str


# Default prompt templates for LLM judge operator
DEFAULT_SYSTEM_PROMPT = """You are a semantic equivalence judge. Analyze the actual output and ground truth to determine if they are semantically equivalent, even if they differ in structure or wording."""

DEFAULT_USER_PROMPT = """Your task is to determine if the actual output semantically matches the ground truth.

Actual output: {actual}
Ground truth: {ground_truth}

Consider: structural equivalence, semantic meaning, and logical correctness.
Respond with your verdict and reasoning."""


@dataclass
class OpResult:
    """Результат выполнения оператора"""

    ok: bool
    message: str | None = None
    got: Any = None


Operator = Callable[[Any, dict[str, Any]], OpResult]


def _serialize_selection(selection: Any) -> str:
    """Serialize selection data to JSON string for LLM consumption.

    Args:
        selection: Any Python object to serialize

    Returns:
        JSON string representation

    Notes:
        - Dict selections are serialized as-is
        - Non-dict types (list, int, str, None, etc.) are wrapped in {"value": ...}
          before serialization to preserve type information for the LLM
    """
    if isinstance(selection, dict):
        return json.dumps(selection)
    else:
        return json.dumps({"value": selection})


def _serialize_ground_truth(ground_truth: Any) -> str:
    """Serialize ground truth data to JSON string for LLM consumption.

    Args:
        ground_truth: Any Python object to serialize

    Returns:
        JSON string representation

    Notes:
        - Complex types (dict, list) are serialized as-is
        - Simple types are wrapped in {"value": ...} to maintain consistency
    """
    if isinstance(ground_truth, (dict, list)):
        return json.dumps(ground_truth)
    else:
        return json.dumps({"value": ground_truth})


def _build_prompts(
    actual: str, ground_truth: str, params: dict[str, Any]
) -> tuple[str, str]:
    """Build system and user prompts for LLM judge.

    Args:
        actual: Serialized actual output string
        ground_truth: Serialized ground truth string
        params: Operator parameters containing optional custom prompts

    Returns:
        Tuple of (system_prompt, user_prompt)

    Notes:
        - Uses default templates if custom prompts not provided in params
        - Custom prompts override defaults
        - Handles KeyError for missing placeholders in custom prompts
    """
    # Extract custom prompts from params, if provided
    custom_system_prompt = params.get("system_prompt")
    custom_user_prompt = params.get("prompt")

    # Use custom or default system prompt
    if custom_system_prompt is not None:
        try:
            system_prompt = custom_system_prompt.format(
                actual=actual, ground_truth=ground_truth
            )
        except KeyError as e:
            raise ValueError(
                f"Missing placeholder {e} in custom system_prompt template"
            ) from e
    else:
        try:
            system_prompt = DEFAULT_SYSTEM_PROMPT.format(
                actual=actual, ground_truth=ground_truth
            )
        except KeyError:
            system_prompt = DEFAULT_SYSTEM_PROMPT

    # Use custom or default user prompt
    if custom_user_prompt is not None:
        try:
            user_prompt = custom_user_prompt.format(
                actual=actual, ground_truth=ground_truth
            )
        except KeyError as e:
            raise ValueError(
                f"Missing placeholder {e} in custom user_prompt template"
            ) from e
    else:
        user_prompt = DEFAULT_USER_PROMPT.format(
            actual=actual, ground_truth=ground_truth
        )

    return system_prompt, user_prompt


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
            False, f"Cannot check 'op_not_contains' for {type(selection)}", selection
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
    exp_idx = 0  # номер ожидаемого элемента, которого ищем
    for item in items:
        if exp_idx < len(expected) and item == expected[exp_idx]:
            exp_idx += 1  # элемент встретился, берем следующий

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


def op_llm_judge(selection: Any, params: dict[str, Any]) -> OpResult:
    """LLM-based semantic equivalence judge using structured output."""
    # Validate required ground_truth parameter
    if "ground_truth" not in params:
        return OpResult(False, "Missing required parameter 'ground_truth'", selection)

    ground_truth = params["ground_truth"]
    expected = params.get("expected", True) 
    if expected is None:
        expected = True

    # Serialize selection
    try:
        serialized_selection = _serialize_selection(selection)
    except (TypeError, ValueError) as e:
        return OpResult(False, f"Error serializing selection: {str(e)}", selection)

    # Serialize ground truth
    try:
        serialized_ground_truth = _serialize_ground_truth(ground_truth)
    except (TypeError, ValueError) as e:
        return OpResult(False, f"Error serializing ground truth: {str(e)}", selection)

    # Size guard: warn if serialized selection > 50KB
    if len(serialized_selection) > 50000:
        logger.warning(
            "Serialized selection size: %d bytes (>50KB)",
            len(serialized_selection),
        )

    # Build prompts
    try:
        system_prompt, user_prompt = _build_prompts(
            serialized_selection, serialized_ground_truth, params
        )
    except ValueError as e:
        return OpResult(False, str(e), selection)

    # Call LLM
    config = load_llm_config()
    llm_result = call_llm(system_prompt, user_prompt, LLMJudgeResponse, config=config)

    # Handle LLM call failure or empty response
    if not llm_result.success or llm_result.value is None:
        error_msg = llm_result.error or "Unknown LLM error"
        return OpResult(False, f"LLM judge error: {error_msg}", selection)

    # Compare verdict with expected
    response = llm_result.value
    logger.debug(
                "LLM judge debug",
                extra={
                    "llm_call": {
                        "model": config.model,
                        "response": response.model_dump(),
                        "expected": expected,
                        "params" : params
                    }
                },
            )
    if response.verdict == expected:
        return OpResult(ok=True, message=None, got=selection)
    else:
        return OpResult(
            ok=False,
            message=f"Semantic mismatch: {response.reasoning}",
            got=selection,
        )


# Реестр всех операторов
OPERATORS: dict[str, Operator] = {
    "exists": op_exists,
    "equals": op_equals,
    "contains": op_contains,
    "not_contains": op_not_contains,
    "length_ge": op_length_ge,
    "match_regex": op_match_regex,
    "sequence_in_order": op_sequence_in_order,
    "llm_judge": op_llm_judge,
}
