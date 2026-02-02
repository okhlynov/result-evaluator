"""Unit tests for async Engine methods.

This module tests the async methods of the Engine class, specifically
run_inference_async() which handles both async and sync inference functions.
"""

import pytest

from result_evaluator.dsl.models import RunConfig
from result_evaluator.runtime.engine import Engine


@pytest.mark.asyncio
async def test_run_inference_async_with_async_function() -> None:
    """Test run_inference_async correctly awaits async inference function."""
    engine = Engine()
    run_config = RunConfig(
        kind="python", target="tests.fixtures.async_dummy_inference"
    )
    input_data = {"test": "data"}

    result = await engine.run_inference_async(run_config, input_data)

    assert result == {
        "status": "async_success",
        "result": "async_dummy_output",
        "count": 99,
    }


@pytest.mark.asyncio
async def test_run_inference_async_with_async_function_passes_input() -> None:
    """Test run_inference_async passes input_data correctly to async function."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.async_echo_inference")
    input_data = {"key": "value", "number": 456}

    result = await engine.run_inference_async(run_config, input_data)

    assert result["status"] == "async_success"
    assert result["input_received"] == input_data


@pytest.mark.asyncio
async def test_run_inference_async_with_sync_function() -> None:
    """Test run_inference_async correctly calls sync inference function."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.dummy_inference")
    input_data = {"test": "data"}

    result = await engine.run_inference_async(run_config, input_data)

    assert result == {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }


@pytest.mark.asyncio
async def test_run_inference_async_with_sync_function_passes_input() -> None:
    """Test run_inference_async passes input_data correctly to sync function."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.echo_inference")
    input_data = {"sync_key": "sync_value", "number": 789}

    result = await engine.run_inference_async(run_config, input_data)

    assert result["status"] == "success"
    assert result["input_received"] == input_data


@pytest.mark.asyncio
async def test_run_inference_async_error_in_async_function() -> None:
    """Test run_inference_async propagates errors from async inference function."""
    engine = Engine()
    run_config = RunConfig(
        kind="python", target="tests.fixtures.async_error_inference"
    )
    input_data = {"test": "data"}

    with pytest.raises(RuntimeError) as exc_info:
        await engine.run_inference_async(run_config, input_data)

    assert "Async inference error: processing failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_inference_async_error_in_sync_function() -> None:
    """Test run_inference_async propagates errors from sync inference function."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.sync_error_inference")
    input_data = {"test": "data"}

    with pytest.raises(ValueError) as exc_info:
        await engine.run_inference_async(run_config, input_data)

    assert "Sync inference error: invalid input" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_inference_async_invalid_module() -> None:
    """Test run_inference_async raises ModuleNotFoundError for invalid module."""
    engine = Engine()
    run_config = RunConfig(
        kind="python", target="nonexistent.module.async_function"
    )
    input_data = {}

    with pytest.raises(ModuleNotFoundError):
        await engine.run_inference_async(run_config, input_data)


@pytest.mark.asyncio
async def test_run_inference_async_invalid_function() -> None:
    """Test run_inference_async raises AttributeError for invalid function."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.nonexistent_func")
    input_data = {}

    with pytest.raises(AttributeError):
        await engine.run_inference_async(run_config, input_data)


@pytest.mark.asyncio
async def test_run_inference_async_unsupported_kind() -> None:
    """Test run_inference_async raises NotImplementedError for unsupported kinds."""
    engine = Engine()

    # Test with 'http' kind
    run_config_http = RunConfig(kind="http", target="http://example.com/api")
    with pytest.raises(NotImplementedError, match="Run kind 'http' not implemented"):
        await engine.run_inference_async(run_config_http, {})

    # Test with 'file' kind
    run_config_file = RunConfig(kind="file", target="/path/to/file.json")
    with pytest.raises(NotImplementedError, match="Run kind 'file' not implemented"):
        await engine.run_inference_async(run_config_file, {})
