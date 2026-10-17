"""Structured JSON logs. Call logs carry technical fields only: no phone numbers and, unless
VA_LOG_TRANSCRIPTS is enabled, no transcripts."""

from __future__ import annotations

import json
import logging
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
                "level": record.levelname.lower(), "logger": record.name, "msg": record.getMessage()}
        call = getattr(record, "call", None)
        if call:
            data["call"] = call
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            data.update(fields)
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def setup(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)  # access logs would carry call ids
