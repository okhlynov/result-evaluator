"Tests for runtime Engine methods."

import pytest
import yaml

from result_evaluator.dsl.models import AssertRule, RunConfig, Scenario
from result_evaluator.runtime.engine import Engine


def test_eval_assert_all_success() -> None:
    """Test eval_assert with all composition where all assertions pass."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {"path": "$.age", "op": "equals", "expected": 30},
            ],
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "All checks passed"


def test_eval_assert_all_fail_one() -> None:
    """Test eval_assert with all composition where one assertion fails."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 30},
            ],
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "Expected Bob, got Alice" in message


def test_eval_assert_all_fail_both() -> None:
    """Test eval_assert with all composition where both assertions fail."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 25},
            ],
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "Expected Bob, got Alice" in message
    assert "Expected 25, got 30" in message


def test_eval_assert_any_success() -> None:
    """Test eval_assert with any composition where at least one assertion passes."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "any": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 30},
            ],
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "At least one check passed"


def test_eval_assert_any_fail() -> None:
    """Test eval_assert with any composition where all assertions fail."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "any": [
                {"path": "$.name", "op": "equals", "expected": "Bob"},
                {"path": "$.age", "op": "equals", "expected": 25},
            ],
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "None passed" in message
    assert "Expected Bob, got Alice" in message
    assert "Expected 25, got 30" in message


def test_eval_assert_not_success() -> None:
    """Test eval_assert with not composition where inner assertion fails."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "not": {"path": "$.name", "op": "equals", "expected": "Bob"},
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "NOT passed"


def test_eval_assert_not_fail() -> None:
    """Test eval_assert with not composition where inner assertion passes."""
    document = {"name": "Alice", "age": 30}

    rule = AssertRule.model_validate(
        {
            "op": "",
            "not": {"path": "$.name", "op": "equals", "expected": "Alice"},
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "NOT failed" in message


def test_eval_assert_all_complex_success() -> None:
    """Test eval_assert with complex all composition on nested document."""
    document = {
        "name": "Alice",
        "age": 30,
        "address": {"city": "New York", "zip": "10001"},
        "tags": ["developer", "python", "testing"],
    }

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {"path": "$.age", "op": "equals", "expected": 30},
                {"path": "$.address.city", "op": "equals", "expected": "New York"},
                {"path": "$.address.zip", "op": "equals", "expected": "10001"},
                {"path": "$.tags", "op": "contains", "expected": "python"},
                {"path": "$.tags", "op": "length_ge", "expected": 3},
            ],
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "All checks passed"


def test_eval_assert_all_complex_fail() -> None:
    """Test eval_assert with complex all composition where one nested check fails."""
    document = {
        "name": "Alice",
        "age": 30,
        "address": {"city": "New York", "zip": "10001"},
        "tags": ["developer", "python", "testing"],
    }

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {"path": "$.age", "op": "equals", "expected": 30},
                {"path": "$.address.city", "op": "equals", "expected": "Boston"},
                {"path": "$.tags", "op": "contains", "expected": "python"},
            ],
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is False
    assert "Expected Boston, got New York" in message


def test_eval_assert_all_nested_composition() -> None:
    """Test eval_assert with nested all compositions (all within all)."""
    document = {
        "name": "Alice",
        "age": 30,
        "address": {"city": "New York", "zip": "10001"},
    }

    rule = AssertRule.model_validate(
        {
            "op": "",
            "all": [
                {"path": "$.name", "op": "equals", "expected": "Alice"},
                {
                    "op": "",
                    "all": [
                        {
                            "path": "$.address.city",
                            "op": "equals",
                            "expected": "New York",
                        },
                        {"path": "$.address.zip", "op": "equals", "expected": "10001"},
                    ],
                },
            ],
        }
    )

    engine = Engine()
    ok, message = engine.eval_assert(rule, document)

    assert ok is True
    assert message == "All checks passed"


def test_run_inference_python_success() -> None:
    """Test run_inference with python kind returns constant dict."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.dummy_inference")
    input_data = {"test": "data"}

    result = engine.run_inference(run_config, input_data)

    assert result == {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }


def test_run_inference_python_with_input() -> None:
    """Test run_inference properly passes input_data to target function."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.echo_inference")
    input_data = {"key": "value", "number": 123}

    result = engine.run_inference(run_config, input_data)

    assert result["status"] == "success"
    assert result["input_received"] == input_data


def test_run_inference_invalid_module() -> None:
    """Test run_inference raises error when module path doesn't exist."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="nonexistent.module.some_function")
    input_data = {}

    with pytest.raises(ModuleNotFoundError):
        engine.run_inference(run_config, input_data)


def test_run_inference_invalid_function() -> None:
    """Test run_inference raises error when function doesn't exist in module."""
    engine = Engine()
    run_config = RunConfig(kind="python", target="tests.fixtures.nonexistent_func")
    input_data = {}

    with pytest.raises(AttributeError):
        engine.run_inference(run_config, input_data)


def test_run_inference_unsupported_kind() -> None:
    """Test run_inference raises NotImplementedError for unsupported run kinds."""
    engine = Engine()

    # Test with 'http' kind
    run_config_http = RunConfig(kind="http", target="http://example.com/api")
    with pytest.raises(NotImplementedError, match="Run kind 'http' not implemented"):
        engine.run_inference(run_config_http, {})

    # Test with 'file' kind
    run_config_file = RunConfig(kind="file", target="/path/to/file.json")
    with pytest.raises(NotImplementedError, match="Run kind 'file' not implemented"):
        engine.run_inference(run_config_file, {})


def test_run_test_success() -> None:
    """Test run_test with all assertions passing."""
    test_case = Scenario.model_validate(
        {
            "case": {"id": "test_success_case", "description": "Test success scenario"},
            "input": {"test": "data"},
            "run": {"kind": "python", "target": "tests.fixtures.dummy_inference"},
            "asserts": [
                {"path": "$.status", "op": "equals", "expected": "success"},
                {"path": "$.count", "op": "equals", "expected": 42},
            ],
        }
    )

    engine = Engine()
    result = engine.run_test(test_case)

    assert result["case_id"] == "test_success_case"
    assert result["status"] == "PASS"
    assert len(result["asserts"]) == 2
    assert all(a["ok"] for a in result["asserts"])
    assert result["result"] == {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }


def test_run_test_fail() -> None:
    """Test run_test with at least one assertion failing."""
    test_case = Scenario.model_validate(
        {
            "case": {"id": "test_fail_case", "description": "Test fail scenario"},
            "input": {"test": "data"},
            "run": {"kind": "python", "target": "tests.fixtures.dummy_inference"},
            "asserts": [
                {"path": "$.status", "op": "equals", "expected": "success"},
                {"path": "$.count", "op": "equals", "expected": 100},
            ],
        }
    )

    engine = Engine()
    result = engine.run_test(test_case)

    assert result["case_id"] == "test_fail_case"
    assert result["status"] == "FAIL"
    assert len(result["asserts"]) == 2
    assert result["asserts"][0]["ok"] is True
    assert result["asserts"][1]["ok"] is False
    assert "Expected 100, got 42" in result["asserts"][1]["message"]
    assert result["result"] == {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }


def test_run_test_from_yaml_success() -> None:
    """Test run_test with test case loaded from YAML string."""
    yaml_content = """
case:
  id: yaml_success_case
  description: Test success scenario from YAML
input:
  test: data
run:
  kind: python
  target: tests.fixtures.dummy_inference
asserts:
  - path: $.status
    op: equals
    expected: success
  - path: $.count
    op: equals
    expected: 42
"""

    yaml_data = yaml.safe_load(yaml_content)
    test_case = Scenario.model_validate(yaml_data)

    engine = Engine()
    result = engine.run_test(test_case)

    assert result["case_id"] == "yaml_success_case"
    assert result["status"] == "PASS"
    assert len(result["asserts"]) == 2
    assert all(a["ok"] for a in result["asserts"])
    assert result["result"] == {
        "status": "success",
        "result": "dummy_output",
        "count": 42,
    }


def test_run_test_from_yaml_fail() -> None:
    """Test run_test with test case from YAML where assertions fail."""
    yaml_content = """
case:
  id: yaml_fail_case
  description: Test fail scenario from YAML
input:
  test: data
run:
  kind: python
  target: tests.fixtures.dummy_inference
asserts:
  - path: $.status
    op: equals
    expected: success
  - path: $.count
    op: equals
    expected: 100
"""

    yaml_data = yaml.safe_load(yaml_content)
    test_case = Scenario.model_validate(yaml_data)

    engine = Engine()
    result = engine.run_test(test_case)

    assert result["case_id"] == "yaml_fail_case"
    assert result["status"] == "FAIL"
    assert len(result["asserts"]) == 2
    assert result["asserts"][0]["ok"] is True
    assert result["asserts"][1]["ok"] is False
    assert "Expected 100, got 42" in result["asserts"][1]["message"]


def test_run_test_from_yaml_complex_composition() -> None:
    """Test run_test with complex assertion compositions from YAML."""
    yaml_content = """
case:
  id: yaml_complex_case
  description: Test complex assertions with all/any/not from YAML
input:
  test: data
run:
  kind: python
  target: tests.fixtures.dummy_inference
asserts:
  - op: ""
    all:
      - path: $.status
        op: equals
        expected: success
      - op: ""
        any:
          - path: $.count
            op: equals
            expected: 42
          - path: $.count
            op: equals
            expected: 100
  - op: ""
    not:
      path: $.status
      op: equals
      expected: failure
"""

    yaml_data = yaml.safe_load(yaml_content)
    test_case = Scenario.model_validate(yaml_data)

    engine = Engine()
    result = engine.run_test(test_case)

    assert result["case_id"] == "yaml_complex_case"
    assert result["status"] == "PASS"
    assert len(result["asserts"]) == 2
    assert all(a["ok"] for a in result["asserts"])


def test_run_test_from_yaml_multiple_operators() -> None:
    """Test run_test with various operators from YAML."""
    yaml_content = """
case:
  id: yaml_operators_case
  description: Test multiple operators from YAML
input:
  test: data
run:
  kind: python
  target: tests.fixtures.rich_inference
asserts:
  - path: $.status
    op: exists
  - path: $.status
    op: equals
    expected: completed
  - path: $.message
    op: contains
    expected: code-123
  - path: $.message
    op: match_regex
    expected: '^Processing.*code-\d+$'
  - path: $.tags
    op: contains
    expected: python
  - path: $.tags
    op: length_ge
    expected: 3
  - path: $.items
    op: length_ge
    expected: 5
  - path: $.metadata.version
    op: match_regex
    expected: '^\d+\.\d+\.\d+$'
"""

    yaml_data = yaml.safe_load(yaml_content)
    test_case = Scenario.model_validate(yaml_data)

    engine = Engine()
    result = engine.run_test(test_case)

    assert result["case_id"] == "yaml_operators_case"
    assert result["status"] == "PASS"
    assert len(result["asserts"]) == 8
    assert all(a["ok"] for a in result["asserts"])


def test_run_test_from_yaml_not_contains() -> None:
    """Test run_test with not_contains operator from YAML."""
    yaml_content = """
case:
  id: yaml_not_contains_case
  description: Test not_contains operator for strings and lists
input:
  test: data
run:
  kind: python
  target: tests.fixtures.rich_inference
asserts:
  - path: $.message
    op: not_contains
    expected: error
  - path: $.message
    op: not_contains
    expected: complete
  - path: $.tags
    op: not_contains
    expected: production
  - path: $.tags
    op: not_contains
    expected: testing
"""

    yaml_data = yaml.safe_load(yaml_content)
    test_case = Scenario.model_validate(yaml_data)

    engine = Engine()
    result = engine.run_test(test_case)

    assert result["case_id"] == "yaml_not_contains_case"
    assert result["status"] == "FAIL"
    assert len(result["asserts"]) == 4

    # First assertion: message does NOT contain "error" -> should PASS
    assert result["asserts"][0]["ok"] is True
    assert result["asserts"][0]["message"] == "OK"

    # Second assertion: message does NOT contain "complete" -> should FAIL
    assert result["asserts"][1]["ok"] is False
    assert "'complete' present in" in result["asserts"][1]["message"]

    # Third assertion: tags do NOT contain "production" -> should PASS
    assert result["asserts"][2]["ok"] is True
    assert result["asserts"][2]["message"] == "OK"

    # Fourth assertion: tags do NOT contain "testing" -> should FAIL
    assert result["asserts"][3]["ok"] is False
    assert "'testing' present in" in result["asserts"][3]["message"]


def test_run_test_llm_judge_with_config_stub() -> None:
    """Test that llm_judge operator with config parses correctly (stub implementation).

    This test verifies that the config field is properly parsed and accessible,
    even though the llm_judge operator execution is not yet implemented.
    """
    yaml_content = """
case:
  id: llm_judge_config_test
  description: Test llm_judge operator with config
input:
  answer: "Paris is the capital of France"
run:
  kind: python
  target: tests.fixtures.dummy_inference
asserts:
  - op: llm_judge
    path: $.result
    config:
      prompt: "Is {input} semantically equivalent to {expected}?"
      system_prompt: "You are a fair judge. Respond only with true or false."
      response_path: "$.verdict"
    expected: "dummy_output"
"""

    yaml_data = yaml.safe_load(yaml_content)
    test_case = Scenario.model_validate(yaml_data)

    # Verify the config was parsed correctly
    assert len(test_case.asserts) == 1
    assert test_case.asserts[0].op == "llm_judge"
    assert test_case.asserts[0].config is not None
    assert test_case.asserts[0].config["prompt"] == "Is {input} semantically equivalent to {expected}?"
    assert test_case.asserts[0].config["system_prompt"] == "You are a fair judge. Respond only with true or false."
    assert test_case.asserts[0].config["response_path"] == "$.verdict"

    engine = Engine()
    result = engine.run_test(test_case)

    # Since llm_judge is not in OPERATORS, it should fail with "Unknown operator"
    assert result["status"] == "FAIL"
    assert result["asserts"][0]["ok"] is False
    assert "Unknown operator: llm_judge" in result["asserts"][0]["message"]