import json
import logging
import sys

from adaptive_rag.pii import redact_pii

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


def _redact_value(value):
    """SS10: PII redaction on logs/traces - recurses through dicts/lists so
    a raw query string nested inside `extra={"plan": {...}, "query": ...}`
    (planning.log_plan_decision's own shape) gets redacted too, not just a
    flat top-level "message" string. `pii.py` has no dependency on this
    module or anything heavier, so importing it here doesn't pull spaCy/
    FastEmbed/httpx into the most foundational module in the codebase."""
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": redact_pii(record.getMessage()),
        }
        # fields passed via logger.info(..., extra={...}) - e.g. request_id -
        # are merged in so callers get structured, queryable logs for free.
        payload.update({k: _redact_value(v) for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
