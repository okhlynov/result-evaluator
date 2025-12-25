from result_evaluator.dsl.models import AssertRule


def test_assert_rule_has_config_field():
    """Verify that AssertRule accepts a 'config' field."""
    rule = AssertRule(  # type: ignore
        op="llm_judge",
        expected="something",
        config={"model": "gpt-4", "temperature": "0.7"},
    )
    assert rule.config == {"model": "gpt-4", "temperature": "0.7"}
    assert rule.op == "llm_judge"


def test_assert_rule_config_is_optional():
    """Verify that 'config' field is optional."""
    rule = AssertRule(op="equals", expected=1)  # type: ignore
    assert rule.config is None
