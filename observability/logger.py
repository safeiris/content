"""JSON logger helpers with request trace support."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

_TRACE_ID = contextvars.ContextVar("trace_id", default=None)
_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Serialize log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", None) or _TRACE_ID.get()
        if trace_id:
            payload["trace_id"] = trace_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in {"args", "levelname", "levelno", "msg", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "message", "asctime"}:
                continue
            payload.setdefault(key, value)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def bind_trace_id(trace_id: str) -> None:
    _TRACE_ID.set(trace_id)


def clear_trace_id() -> None:
    _TRACE_ID.set(None)


def log_step(logger: logging.Logger, *, job_id: str, step: str, status: str, **details: Any) -> None:
    logger.info(
        "job_step",
        extra={"job_id": job_id, "step": step, "step_status": status, "details": details or None},
    )
