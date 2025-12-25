"""JSONL formatter for structured logging."""

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JSONLFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSONL format."""

    def format(self, record: logging.LogRecord) -> str:
        # Build base log object
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Collect extra fields (fields not in standard LogRecord attributes)
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
            "getMessage",
        }

        extra_data: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                extra_data[key] = self._serialize_value(value)

        # Add data field only if there are extra fields
        if extra_data:
            log_obj["data"] = extra_data

        return json.dumps(log_obj, ensure_ascii=False)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value according to JSONL format rules."""
        # Rule 1: Dict or list - output as-is (JSON-compatible)
        if isinstance(value, (dict | list)):
            return value

        # Rule 2: Pydantic model - use .model_dump()
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return value.model_dump()
            except Exception:
                # Fallback to string on error
                return str(value)

        # Rule 3: Everything else - convert to string
        return str(value)
