"""Configuration module for LLM environment variables."""

import os

from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """Configuration for OpenAI API configuration."""

    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(..., description="Model name (e.g., gpt-4o)")
    endpoint: str | None = Field(None, description="Custom API endpoint")
    timeout: int = Field(60, description="Request timeout in seconds")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")

    @field_validator("api_key", "model")
    @classmethod
    def check_not_empty(cls, v: str) -> str:
        """Ensure the string field is not empty or whitespace only."""
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v

    @field_validator("timeout")
    @classmethod
    def check_positive_timeout(cls, v: int) -> int:
        """Ensure the timeout is a positive integer."""
        if v <= 0:
            raise ValueError("Timeout must be a positive integer")
        return v


def load_llm_config(prefix: str = "JUDGE_LLM") -> LLMConfig:
    """
    Load LLM configuration from environment variables.

    Reads variables with the specified prefix (default: JUDGE_LLM_).
    Required variables:
        - {PREFIX}_API_KEY: OpenAI API key
        - {PREFIX}_MODEL: Model name (e.g., gpt-4o)

    Optional variables:
        - {PREFIX}_ENDPOINT: Custom API endpoint
        - {PREFIX}_TIMEOUT: Request timeout in seconds (default: 60)
        - {PREFIX}_MAX_TOKENS: Maximum tokens to generate

    Args:
        prefix: Prefix for environment variables.

    Returns:
        LLMConfig: Configuration object.

    Raises:
        ValueError: If required environment variables are missing or invalid.

    Example:
        >>> import os
        >>> os.environ["JUDGE_LLM_API_KEY"] = "sk-..."
        >>> os.environ["JUDGE_LLM_MODEL"] = "gpt-4o"
        >>> config = load_llm_config()
    """
    api_key = os.getenv(f"{prefix}_API_KEY")
    if not api_key:
        raise ValueError(f"Missing required environment variable: {prefix}_API_KEY")

    model = os.getenv(f"{prefix}_MODEL")
    if not model:
        raise ValueError(f"Missing required environment variable: {prefix}_MODEL")

    endpoint = os.getenv(f"{prefix}_ENDPOINT")

    timeout_str = os.getenv(f"{prefix}_TIMEOUT")
    timeout = 60
    if timeout_str:
        try:
            timeout = int(timeout_str)
        except ValueError:
            raise ValueError(
                f"Environment variable {prefix}_TIMEOUT must be an integer, got '{timeout_str}'"
            ) from None

    max_tokens_str = os.getenv(f"{prefix}_MAX_TOKENS")
    max_tokens = None
    if max_tokens_str:
        try:
            max_tokens = int(max_tokens_str)
        except ValueError:
            raise ValueError(
                f"Environment variable {prefix}_MAX_TOKENS must be an integer, got '{max_tokens_str}'"
            ) from None

    return LLMConfig(
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        timeout=timeout,
        max_tokens=max_tokens,
    )
