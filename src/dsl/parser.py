from pathlib import Path

import yaml

from dsl.models import Scenario


def load_test_case(file_path: str | Path) -> Scenario:
    """Загружает тест-кейс из YAML файла"""
    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scenario(**data)
