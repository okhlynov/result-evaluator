"""Test fixtures and helper functions for test cases."""


def dummy_inference(input_data: dict) -> dict:
    """Simple inference function that returns a constant dict for testing.

    Args:
        input_data: Input data dict (passed but not used in this dummy implementation)

    Returns:
        A constant dict with predefined structure
    """
    return {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }


def echo_inference(input_data: dict) -> dict:
    """Inference function that echoes back the input data.

    Args:
        input_data: Input data dict to echo back

    Returns:
        Dict containing the original input wrapped in a result key
    """
    return {
        "status": "success",
        "input_received": input_data,
    }
