from typing import Any, Literal

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Как запустить инференс"""

    kind: Literal["python", "http", "file"]
    target: str  # python path: module.function
    timeout_ms: int = 5000


class AssertRule(BaseModel):
    """Одно утверждение"""

    path: str | None = None
    op: str
    expected: Any = None
    config: dict[str, str] | None = None
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
