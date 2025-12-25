"""
LLM operations runtime types.
"""

import functools
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from .config import LLMConfig, load_llm_config

logger = logging.getLogger(__name__)


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


def with_llm_error_handling[**P, R](
    func: Callable[P, Result[R]],
) -> Callable[P, Result[R]]:
    """Decorator to handle LLM-related exceptions and convert them to Result."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[R]:
        # Bind arguments to access inputs for error messages
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        config: LLMConfig | None = bound.arguments.get("config")

        # Fallback values for error messages
        timeout_val = config.timeout if config else "unknown"
        model_val = config.model if config else "unknown"

        try:
            return func(*args, **kwargs)
        except AuthenticationError:
            logger.error("Authentication failed", exc_info=True)
            return Result.fail(
                "connection", "Authentication failed: check JUDGE_LLM_API_KEY"
            )
        except APITimeoutError:
            logger.error("Request timeout", exc_info=True)
            return Result.fail("connection", f"Request timeout after {timeout_val}s")
        except APIConnectionError:
            logger.error("Connection failed", exc_info=True)
            return Result.fail(
                "connection", "Connection failed: check network/endpoint"
            )
        except RateLimitError:
            logger.error("Rate limit exceeded", exc_info=True)
            return Result.fail(
                "connection", "Rate limit exceeded: wait before retrying"
            )
        except NotFoundError:
            logger.error(f"Model '{model_val}' not found", exc_info=True)
            return Result.fail("connection", f"Model '{model_val}' not found")
        except APIError as e:
            logger.error("API error", exc_info=True)
            return Result.fail("response", f"API error: {str(e)}")
        except ValidationError as e:
            logger.error("Validation error", exc_info=True)
            return Result.fail("validation", f"Response doesn't match schema: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error", exc_info=True)
            return Result.fail("response", f"Unexpected error: {str(e)}")

    return wrapper


def with_llm_logging[**P, R](
    func: Callable[P, Result[R]],
) -> Callable[P, Result[R]]:
    """Decorator to log LLM call metrics and warnings."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[R]:
        # Bind arguments to access inputs for logging
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        config: LLMConfig | None = bound.arguments.get("config")
        system_prompt = bound.arguments.get("system_prompt", "")
        user_prompt = bound.arguments.get("user_prompt", "")
        response_type = bound.arguments.get("response_type")

        # Warning for large prompt
        if len(system_prompt) + len(user_prompt) > 100_000:
            logger.warning(
                "Large prompt detected: %d characters",
                len(system_prompt) + len(user_prompt),
            )

        # Log the actual prompts at DEBUG level
        logger.debug("LLM Call - System Prompt:\n%s", system_prompt)
        logger.debug("LLM Call - User Prompt:\n%s", user_prompt)

        start_time = time.time()
        result = func(*args, **kwargs)
        latency_ms = int((time.time() - start_time) * 1000)

        if result.success and config:
            logger.debug(
                "LLM call completed",
                extra={
                    "llm_call": {
                        "model": config.model,
                        "latency_ms": latency_ms,
                        "system_prompt_length": len(system_prompt),
                        "user_prompt_length": len(user_prompt),
                        "response_type": (
                            getattr(response_type, "__name__", str(response_type))
                            if response_type is not None
                            else "unknown"
                        ),
                        "response": (
                            result.value.model_dump()  # type: ignore[union-attr]
                            if result.value is not None
                            and hasattr(result.value, "model_dump")
                            else result.value
                        ),
                    }
                },
            )

        return result

    return wrapper


@with_llm_logging
@with_llm_error_handling
def _execute_llm_request[T](
    system_prompt: str,
    user_prompt: str,
    response_type: type[T],
    config: LLMConfig,
) -> Result[T]:
    """Execute the OpenAI API request."""
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.endpoint,
        timeout=float(config.timeout),
    )

    completion = client.beta.chat.completions.parse(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_type,
        max_tokens=config.max_tokens,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return Result.fail("response", "Model refused to generate structured output")

    return Result.ok(parsed)


def call_llm[T](
    system_prompt: str,
    user_prompt: str,
    response_type: type[T],
    config: LLMConfig | None = None,
) -> Result[T]:
    """
    Call LLM with structured output.

    Args:
        system_prompt: System instruction for the LLM.
        user_prompt: User input/query.
        response_type: The expected type of the response (e.g., Pydantic model).
        config: LLM configuration. If None, loads from environment.

    Returns:
        Result[T]: The parsed response or error details.

    Raises:
        ValueError: If prompts are empty or whitespace-only.
    """
    if not system_prompt or not system_prompt.strip():
        raise ValueError("system_prompt cannot be empty or whitespace-only")

    if not user_prompt or not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty or whitespace-only")

    if config is None:
        config = load_llm_config()

    return _execute_llm_request(
        system_prompt, user_prompt, response_type, config=config
    )
