"""Generation (Phase 6, FR16-FR18): Context Builder + the one expensive
generation-model call (`gpt-oss-20b` via Groq), streamed by default
(Mode 1, FR17) or fully buffered (Mode 2, NFR7).

FR18/NFR4: `settings.groq_model` is referenced ONLY in this module -
verified by `test_generation.py`'s code-scan test, which walks every other
file under `src/adaptive_rag/` for an actual `.groq_model` attribute
access (via the `ast` module, not string grepping - so a docstring merely
*mentioning* `settings.groq_model`, like ingestion.py's and recovery.py's
"never this one" notes, doesn't false-positive). Every other Groq call in
this codebase uses `settings.groq_extraction_model` instead - a distinct,
smaller model.

Mode 1/Mode 2 share one underlying streaming call (`GroqGenerationClient.stream`)
- Mode 1 (`generate_streaming`) yields each token as it arrives, Mode 2
(`generate_buffered`) fully consumes the same generator before returning
anything. This isn't two code paths that could drift apart; buffering
happens purely by choosing how the caller consumes one generator, so
NFR7 ("once tokens begin streaming, never retract; high-risk routes to
buffered before generation begins") holds by construction - there's no
partial-emission branch in the buffered path to accidentally trigger.
Response format verified 2026-09-16 against the real Groq API: reasoning
models like gpt-oss-20b stream a separate `delta.reasoning` field (internal
chain-of-thought) alongside `delta.content` - only `content` is surfaced to
the caller here, `reasoning` is intentionally never yielded.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from functools import lru_cache
from typing import Protocol

from adaptive_rag.config import get_settings
from adaptive_rag.grading import Grade

DEFAULT_TIMEOUT_SECONDS = 60.0

_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the information in the provided context. "
    "If the context does not contain enough information to answer confidently, say so honestly "
    "rather than making something up. Do not present information from the context's "
    "'supplementary' sources with the same confidence as 'authoritative' sources."
)


def build_context(query: str, graded_evidence: list[dict]) -> list[dict]:
    """FR16: assembles the prompt from graded evidence only - excludes any
    evidence graded Incorrect (FR16 acceptance criterion, matched exactly:
    Ambiguous evidence that reached here - which shouldn't normally happen
    once Phase 5's Recovery Planner has run - is still included, since the
    acceptance criterion only mandates excluding Incorrect). Graph-derived
    evidence needs no separate "resolve to source text" step here: every
    Phase 4 retriever (including GraphRetriever) already returns real
    resolved chunk text in `text`, never a raw graph edge label, per SS3 -
    that invariant is enforced at retrieval time, not here."""
    trusted = [e for e in graded_evidence if e.get("grade") != Grade.INCORRECT]
    context_block = "\n\n".join(f"[Source {i + 1}] {e['text']}" for i, e in enumerate(trusted))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
    ]


SOURCE_TEXT_SNIPPET_CHARS = 300


def extract_sources(graded_evidence: list[dict]) -> list[dict]:
    """API contract (SS9.1): the `sources` list attached to a response,
    tagging trust_tier explicitly so a client never has to guess whether a
    source is authoritative or (web-search) supplementary.

    `text` (Phase 10/frontend): a snippet of the evidence's own already-
    resolved chunk text (SS3's invariant - never a raw graph edge label),
    truncated so a source card renders a preview rather than the full
    passage. No document "title" field exists anywhere in the ingestion
    pipeline (SS3 tracks doc_id only) - inventing one would be fake UI
    data, so `doc_id` is the only document-identifying field."""
    trusted = [e for e in graded_evidence if e.get("grade") != Grade.INCORRECT]
    return [
        {
            "id": e.get("chunk_id", e.get("id")),
            "doc_id": e.get("doc_id"),
            "trust_tier": e.get("trust_tier", "authoritative"),
            "score": e.get("grader_score"),
            "text": (e.get("text") or "")[:SOURCE_TEXT_SNIPPET_CHARS],
        }
        for e in trusted
    ]


class GenerationLike(Protocol):
    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class GroqGenerationClient:
    """The ONE expensive generation call in this codebase (FR18/NFR4) -
    uses `settings.groq_model`, never `settings.groq_extraction_model`."""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        import httpx

        settings = get_settings()
        with httpx.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={"model": settings.groq_model, "messages": messages, "stream": True},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload.strip() == "[DONE]":
                    return
                delta = json.loads(payload)["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content


@lru_cache
def get_generation_client() -> GenerationLike:
    return GroqGenerationClient()


def generate_streaming(messages: list[dict], client: GenerationLike | None = None) -> Iterator[str]:
    """Mode 1 (FR17, default): yields tokens as they arrive. NFR7: once a
    token is yielded here it is irrevocably sent - callers must route
    high-risk queries to `generate_buffered` *before* calling this, not
    mid-stream (mode selection itself is Phase 7's FR21, not built here)."""
    model = client or get_generation_client()
    yield from model.stream(messages)


def generate_buffered(messages: list[dict], client: GenerationLike | None = None) -> str:
    """Mode 2 (NFR7, high-risk): fully consumes the stream internally and
    returns the complete text as one unit - nothing reaches the caller
    until generation finishes. There is no partial-emission code path here
    to accidentally trigger; buffering is a property of how this function
    consumes `model.stream`, not a second implementation that could drift
    from `generate_streaming`."""
    model = client or get_generation_client()
    return "".join(model.stream(messages))


def format_sse_event(event: str, data: dict) -> str:
    """SS9.1's wire format: `event: <name>\\ndata: <json>\\n\\n`."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_sse_response(messages: list[dict], sources: list[dict], client: GenerationLike | None = None) -> Iterator[str]:
    """Mode 1's actual SSE wire output (SS9.1): a `sources` event (emitted
    up front - Context Builder already selected them before generation
    starts, so the client can show citations alongside streaming tokens
    rather than waiting), then one `token` event per chunk. Deliberately
    does NOT emit `grounding` or `done` events - those need a request_id,
    latency, route, and recovery_used that only the actual `/v1/query`
    handler knows, and `grounding` is Phase 7's async check, which doesn't
    exist yet. That handler (Phase 7's job, since it also needs FR21's
    mode-routing decision before it can pick Mode 1 vs Mode 2 correctly) is
    responsible for appending those once built - this function only covers
    what Phase 6 owns."""
    yield format_sse_event("sources", {"sources": sources})
    for token in generate_streaming(messages, client=client):
        yield format_sse_event("token", {"text": token})
