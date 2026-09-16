"""Shared PII redaction (SS9/SS10): baseline regex redaction, not a full
PII detector - matches this project's documented "baseline PII hygiene
only" stance (compliance scope undetermined). A dependency-free leaf
module (no other adaptive_rag import) so it can be used from `logging.py`
(the most foundational module in this codebase, configured before anything
else runs) without pulling in ingestion/planning/retrieval's heavier
dependencies (spaCy, FastEmbed, httpx clients).

Originally lived in `recovery.py` (Phase 5, FR15 - redacting outbound web
search queries) but SS10 explicitly calls for the *same* redaction policy
to also cover logs/traces and entity-extraction outputs - a single shared
primitive, not three call sites duplicating regex patterns that could
silently drift apart.
"""
from __future__ import annotations

import re

# Order matters: SSN before the looser phone pattern, so a 9-digit SSN
# doesn't get half-swallowed by a phone match first.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)")


def redact_pii(text: str) -> str:
    """Applied at every system boundary that can leak PII: outbound web
    search queries (FR15), logs/traces (SS10), and logged entity-extraction
    outputs (SS10 - ingestion may extract names/contact info as entities)."""
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _SSN_RE.sub("[REDACTED_SSN]", text)
    text = _CREDIT_CARD_RE.sub("[REDACTED_CARD]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text
