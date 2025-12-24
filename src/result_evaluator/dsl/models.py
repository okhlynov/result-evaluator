from typing import Any, Literal

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Как запустить инференс"""

    kind: Literal["python", "http", "file"]
    target: str  # python path: module.function
    timeout_ms: int = 5000


class AssertRule(BaseModel):
    """Single assertion rule for test case validation.

    Fields:
        path: JSONPath expression to extract value from inference result
        op: Operator name (equals, contains, llm_judge, etc.)
        expected: Expected value for comparison
        config: Optional operator-specific configuration dictionary.
            Different operators use different config keys:
            - llm_judge: {prompt, system_prompt, response_path}
            - Other operators typically don't require config
        where: Optional list of sub-rules for filtering (not yet implemented)
        all_: Logical AND composition of multiple rules
        any_: Logical OR composition of multiple rules
        not_: Logical NOT negation of a rule

    Example:
        Basic assertion without config:
        >>> rule = AssertRule(path="$.status", op="equals", expected="success")

        LLM judge with config:
        >>> rule = AssertRule(
        ...     path="$.answer",
        ...     op="llm_judge",
        ...     expected="true",
        ...     config={
        ...         "prompt": "Is {input} semantically equivalent to {expected}?",
        ...         "response_path": "$.verdict",
        ...     },
        ... )
    """

    path: str | None = None
    op: str
    expected: Any = None
    config: dict[str, Any] | None = None
    where: list["AssertRule"] | None = None

    # Композиции
    all_: list["AssertRule"] | None = Field(None, alias="all")
    any_: list["AssertRule"] | None = Field(None, alias="any")
    not_: "AssertRule | None" = Field(None, alias="not")


class Scenario(BaseModel):
    """Полный тест-кейс"""

    case: dict[str, Any]  # id, description, tags
    input: dict[str, Any]
    run: RunConfig
    asserts: list[AssertRule]
