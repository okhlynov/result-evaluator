from pathlib import Path

import yaml

from dsl.models import TestCase


def load_test_case(file_path: str | Path) -> TestCase:
    """Загружает тест-кейс из YAML файла"""
    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TestCase(**data)
