"""
LLM operations runtime types.
"""
from dataclasses import dataclass


@dataclass
class Result[T]:
    """
    Generic result type for LLM operations.
    
    Attributes:
        success: Whether the operation was successful
        value: The successful result value (if success=True)
        error: Error message (if success=False)
        error_type: Classification of the error (if success=False)
            Common types: "connection", "response", "validation"
    """

    success: bool
    value: T | None = None
    error: str | None = None
    error_type: str | None = None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        """Create a successful result."""
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error_type: str, message: str) -> "Result[T]":
        """
        Create an error result.
        
        Args:
            error_type: One of "connection", "response", "validation" (or custom)
            message: Human-readable error description
        """
        return cls(success=False, error=message, error_type=error_type)