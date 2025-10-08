from typing import Any

from jsonpath_ng.ext import parse


def eval_path(doc: dict[str, Any], path: str) -> Any:
    """
    Выполняет JSONPath запрос к документу.
    Возвращает список найденных значений или [] если ничего не найдено.
    """
    if not path:
        return doc

    expr = parse(path)
    matches = [m.value for m in expr.find(doc)]

    if len(matches) == 1:
        return matches[0]
    return matches
