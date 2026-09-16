import json
import logging

from adaptive_rag.logging import JsonFormatter


def _format(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def _make_record(msg: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=0, msg=msg, args=(), exc_info=None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_basic_shape():
    record = _make_record("hello world")
    payload = _format(record)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "hello world"


def test_json_formatter_merges_extra_fields():
    record = _make_record("plan_decision", request_id="req-1", query="q")
    payload = _format(record)
    assert payload["request_id"] == "req-1"
    assert payload["query"] == "q"


# ---------------------------------------------------------------------------
# PII redaction (SS10)
# ---------------------------------------------------------------------------


def test_json_formatter_redacts_pii_in_message():
    record = _make_record("failed to process query from john.doe@example.com")
    payload = _format(record)
    assert "john.doe@example.com" not in payload["message"]
    assert "[REDACTED_EMAIL]" in payload["message"]


def test_json_formatter_redacts_pii_in_top_level_extra_field():
    record = _make_record("plan_decision", query="what is my SSN 123-45-6789")
    payload = _format(record)
    assert "123-45-6789" not in payload["query"]
    assert "[REDACTED_SSN]" in payload["query"]


def test_json_formatter_redacts_pii_nested_inside_dict_extra_field():
    record = _make_record("plan_decision", plan={"note": "contact 415-555-2671 for details", "top_k": 20})
    payload = _format(record)
    assert "415-555-2671" not in payload["plan"]["note"]
    assert "[REDACTED_PHONE]" in payload["plan"]["note"]
    assert payload["plan"]["top_k"] == 20  # non-string values pass through untouched


def test_json_formatter_leaves_clean_text_untouched():
    record = _make_record("plan_decision", request_id="req-abc-123")
    payload = _format(record)
    assert payload["request_id"] == "req-abc-123"  # not PII, not mangled
