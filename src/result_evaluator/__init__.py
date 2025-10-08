"""Result evaluator - test framework for evaluating complex inference results.

This package provides a DSL for defining test cases with JSONPath queries
and composable assertions.

Example:
    >>> from result_evaluator import Engine, load_test_case
    >>> test = load_test_case("test.yaml")
    >>> engine = Engine()
    >>> result = engine.run_test(test)
    >>> print(result["status"])
"""

# DSL exports
from .dsl import AssertRule, RunConfig, Scenario, load_test_case

# Runtime exports
from .runtime import OPERATORS, Engine, OpResult, eval_path

__all__ = [
    # DSL
    "AssertRule",
    "RunConfig",
    "Scenario",
    "load_test_case",
    # Runtime
    "Engine",
    "OPERATORS",
    "OpResult",
    "eval_path",
]


def main() -> None:
    """Main entry point for the result-evaluator CLI."""
    print("Hello from result-evaluator!")
