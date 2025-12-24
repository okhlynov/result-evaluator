"""Tests for DSL model validation and YAML parsing.

This module tests the Pydantic models directly, focusing on:
- Field validation
- YAML parsing with various configurations
- Backward compatibility
- Config field behavior
"""

import pytest
import yaml
from pydantic import ValidationError

from result_evaluator.dsl.models import AssertRule


def test_assert_rule_with_simple_config() -> None:
    """Test AssertRule parses YAML with simple config values."""
    yaml_content = """
    op: llm_judge
    path: $.answer
    config:
      prompt: "Is {input} correct?"
      response_path: "$.verdict"
    expected: "true"
    """

    yaml_data = yaml.safe_load(yaml_content)
    rule = AssertRule.model_validate(yaml_data)

    assert rule.op == "llm_judge"
    assert rule.path == "$.answer"
    assert rule.config is not None
    assert rule.config["prompt"] == "Is {input} correct?"
    assert rule.config["response_path"] == "$.verdict"
    assert rule.expected == "true"


def test_assert_rule_with_nested_config() -> None:
    """Test AssertRule parses YAML with nested dictionary config values."""
    yaml_content = """
    op: llm_judge
    config:
      params:
        temperature: 0.7
        top_p: 1.0
      provider:
        name: openai
        model: gpt-4
    """

    yaml_data = yaml.safe_load(yaml_content)
    rule = AssertRule.model_validate(yaml_data)

    assert rule.config is not None
    assert isinstance(rule.config["params"], dict)
    assert rule.config["params"]["temperature"] == 0.7
    assert rule.config["provider"]["name"] == "openai"


def test_assert_rule_with_mixed_config() -> None:
    """Test AssertRule parses YAML with mixed type config values (bool, int, list)."""
    yaml_content = """
    op: llm_judge
    config:
      strict: true
      retries: 3
      tags: ["llm", "eval"]
    """

    yaml_data = yaml.safe_load(yaml_content)
    rule = AssertRule.model_validate(yaml_data)

    assert rule.config is not None
    assert rule.config["strict"] is True
    assert rule.config["retries"] == 3
    assert rule.config["tags"] == ["llm", "eval"]


def test_assert_rule_without_config() -> None:
    """Test backward compatibility: AssertRule parses valid YAML without config field."""
    yaml_content = """
    op: contains
    path: $.text
    expected: "hello"
    """

    yaml_data = yaml.safe_load(yaml_content)
    rule = AssertRule.model_validate(yaml_data)

    assert rule.op == "contains"
    assert rule.config is None
    assert rule.expected == "hello"


def test_assert_rule_config_in_composition() -> None:
    """Test config field parsing within composition operators (all/any)."""
    yaml_content = """
    op: and
    all:
      - op: llm_judge
        config:
          model: gpt-4
      - op: regex
        config:
          pattern: "^[A-Z]"
    """

    yaml_data = yaml.safe_load(yaml_content)
    rule = AssertRule.model_validate(yaml_data)

    assert rule.op == "and"
    assert rule.all_ is not None
    assert len(rule.all_) == 2
    
    assert rule.all_[0].op == "llm_judge"
    assert rule.all_[0].config is not None
    assert rule.all_[0].config["model"] == "gpt-4"
    
    assert rule.all_[1].op == "regex"
    assert rule.all_[1].config is not None
    assert rule.all_[1].config["pattern"] == "^[A-Z]"