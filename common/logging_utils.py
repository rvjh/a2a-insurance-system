"""
Structured JSON logging with correlation IDs.

Why JSON logs: every observability platform (Datadog, ELK, CloudWatch,
Grafana Loki) ingests JSON natively. Plain text breaks parsing.

Why correlation IDs: in a multi-agent system, ONE user request hits
gateway -> agent A -> tool -> agent B -> tool. Without a shared trace_id
threaded through every log line, debugging is impossible.
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any
import json
from datetime import datetime, timezone

# ContextVar = async-safe thread-local. Survives across `await` boundaries.
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="-")
agent_name_ctx: ContextVar[str] = ContextVar("agent_name", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_ctx.get(),
            "agent": agent_name_ctx.get(),
        }
        # Allow extra={"foo": "bar"} on log calls
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # Silence noisy libraries
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_with_extra(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Emit a log record with arbitrary extra JSON fields."""
    record = logger.makeRecord(logger.name, level, "", 0, msg, None, None)
    record.extra_fields = fields  # type: ignore[attr-defined]
    logger.handle(record)
