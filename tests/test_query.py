"""Tests for runtime query functions."""

from runtime.query import eval_path


def test_eval_path_empty_path() -> None:
    """Test eval_path with empty path returns entire document."""
    doc = {"name": "John", "age": 30}
    result = eval_path(doc, "")
    assert result == {"name": "John", "age": 30}


def test_eval_path_simple_field() -> None:
    """Test eval_path with simple field access."""
    doc = {"name": "John", "age": 30}
    result = eval_path(doc, "$.name")
    assert result == "John"


def test_eval_path_simple_field_no_dollar() -> None:
    """Test eval_path with simple field access without $ prefix."""
    doc = {"name": "John", "age": 30}
    result = eval_path(doc, "name")
    assert result == "John"


def test_eval_path_nested_field() -> None:
    """Test eval_path with nested field access."""
    doc = {"user": {"name": "Alice", "email": "alice@example.com"}}
    result = eval_path(doc, "$.user.name")
    assert result == "Alice"


def test_eval_path_array_element() -> None:
    """Test eval_path with array element access."""
    doc = {"items": ["apple", "banana", "cherry"]}
    result = eval_path(doc, "$.items[0]")
    assert result == "apple"


def test_eval_path_array_wildcard() -> None:
    """Test eval_path with array wildcard returns list."""
    doc = {"items": ["apple", "banana", "cherry"]}
    result = eval_path(doc, "$.items[*]")
    assert result == ["apple", "banana", "cherry"]


def test_eval_path_non_existent() -> None:
    """Test eval_path with non-existent path returns empty list."""
    doc = {"name": "John", "age": 30}
    result = eval_path(doc, "$.nonexistent")
    assert result == []


def test_eval_path_deeply_nested() -> None:
    """Test eval_path with deeply nested structure."""
    doc = {"company": {"department": {"team": {"lead": "Bob"}}}}
    result = eval_path(doc, "$.company.department.team.lead")
    assert result == "Bob"


def test_eval_path_array_of_dicts() -> None:
    """Test eval_path with array of dictionaries."""
    doc = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    result = eval_path(doc, "$.users[0].name")
    assert result == "Alice"


def test_eval_path_multiple_matches() -> None:
    """Test eval_path with multiple matches returns list."""
    doc = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    result = eval_path(doc, "$.users[*].name")
    assert result == ["Alice", "Bob"]
