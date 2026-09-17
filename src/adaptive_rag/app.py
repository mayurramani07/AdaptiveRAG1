import asyncio
import logging
import time
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from adaptive_rag.config import get_settings
from adaptive_rag.gateway import (
    cache_get,
    cache_set,
    enforce_rate_limit,
    require_api_key,
)
from adaptive_rag.generation import (
    format_sse_event,
    generate_buffered,
    generate_streaming,
)
from adaptive_rag.grounding import (
    DEFAULT_MODE2_PASS_THRESHOLD,
    check_grounding,
    check_grounding_async,
    is_passed,
    select_mode,
)
from adaptive_rag.ingestion import (
    chunk_document,
    extract_upload_text,
    get_llm_client,
    get_neo4j_driver,
    sanitize_upload_doc_id,
)
from adaptive_rag.logging import configure_logging
from adaptive_rag.observability import check_dependency_health, record_eval_event
from adaptive_rag.pipeline import run_pipeline
from adaptive_rag.recovery import INSUFFICIENT_EVIDENCE_MESSAGE
from adaptive_rag.retrieval import ingest_and_index

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Adaptive RAG")

# PRD SS10/19.2: an explicit origin allowlist for the frontend (Phase 10) -
# never "*", fails closed if CORS_ALLOWED_ORIGINS is empty/unset rather than
# permitting every origin.
_cors_origins = [origin.strip() for origin in get_settings().cors_allowed_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.get("/v1/health")
async def health() -> dict:
    deps = await check_dependency_health()
    return {"status": "ok" if all(v == "ok" for v in deps.values()) else "degraded", "dependencies": deps}


# FR-ING5 (Phase 11, reverses NG8) upload limits - two DIFFERENT concerns,
# enforced at two DIFFERENT stages, deliberately not one number:
#
# MAX_DOCUMENT_CHARS is a document-level sanity guard against genuinely
# unsafe/unbounded input (a corrupted extraction, a deliberately oversized
# paste) - NOT a proxy for how long ingestion will take. A real 100+ page
# report is well within this.
MAX_DOCUMENT_CHARS = 500_000

# MAX_CHUNKS_PER_UPLOAD is the actual cost/time driver: ingest_document
# (ingestion.py) makes 2 sequential LLM calls per chunk (extract_entities +
# extract_relationships), all inside this one blocking HTTP request (no
# background job - a deliberate Phase 11 simplification, NG8). This is the
# correct boundary for a "can this be processed synchronously" check -
# checked AFTER chunking, not by gating on raw pre-chunk document length.
# 150 chunks * 200 words/chunk (DEFAULT_CHUNK_WORDS) = ~30k words, which
# comfortably covers a real ~100k-character report (~90 chunks) with
# headroom, while keeping worst-case synchronous processing time bounded on
# free-tier infra. Raise this only alongside moving ingestion to a
# background job (out of scope here), not in isolation.
MAX_CHUNKS_PER_UPLOAD = 150


@app.post("/v1/documents")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str | None = Form(default=None),
    department: str | None = Form(default=None),
    date: str | None = Form(default=None),
    api_key: str = Depends(require_api_key),
) -> dict:
    content = await file.read()
    try:
        text = extract_upload_text(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not text:
        raise HTTPException(status_code=400, detail="no extractable text found in the uploaded file")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise HTTPException(status_code=413, detail=f"document too large ({len(text)} chars, max {MAX_DOCUMENT_CHARS})")

    doc_id = sanitize_upload_doc_id(file.filename or "document")
    chunks = chunk_document(doc_id, text)
    if len(chunks) > MAX_CHUNKS_PER_UPLOAD:
        raise HTTPException(
            status_code=413,
            detail=f"document produces too many chunks to process synchronously ({len(chunks)} chunks, max {MAX_CHUNKS_PER_UPLOAD})",
        )
    metadata = {k: v for k, v in {"doc_type": doc_type, "department": department, "date": date}.items() if v}

    def _ingest() -> int:
        driver = get_neo4j_driver()
        llm = get_llm_client()
        ingest_and_index(driver, doc_id, text, llm=llm, metadata=metadata or None)
        return len(chunks)

    chunks_indexed = await run_in_threadpool(_ingest)
    return {"doc_id": doc_id, "filename": file.filename, "chunks_indexed": chunks_indexed}


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str | None = None
    stream: bool = True


def _log_grounding_result(request_id: str, route: str, recovery_used: bool, loop: asyncio.AbstractEventLoop, future) -> None:
    """FR19: Mode 1's grounding check is a detector, not a gate - its
    result reaches the eval loop via this log line plus a Redis-backed eval
    event (FR7: "plan + final grounding score queryable by request ID
    within 5 minutes"), never by altering a response that already went out.
    Runs on the grounding thread pool's worker thread (this is an
    `add_done_callback`), not the event loop - `record_eval_event` is
    async, so it's scheduled onto the loop captured before the background
    check was kicked off, via `run_coroutine_threadsafe` (the standard way
    to call async code from a non-event-loop thread)."""
    try:
        check = future.result()
        logger.info("grounding_result", extra={"request_id": request_id, "grounded": check.grounded, "confidence": check.confidence})
        event = {"route": route, "recovery_used": recovery_used, "grounded": check.grounded, "confidence": check.confidence}
        asyncio.run_coroutine_threadsafe(record_eval_event(request_id, event), loop)
    except Exception:
        logger.warning("grounding_check_failed", extra={"request_id": request_id}, exc_info=True)


@app.post("/v1/query")
async def query(req: QueryRequest, api_key: str = Depends(require_api_key)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    await enforce_rate_limit(api_key)

    start = time.monotonic()
    cached = await cache_get(req.query)
    if cached is not None:
        cached["cache_hit"] = True
        cached["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return cached

    request_id = str(uuid.uuid4())
    # FR3/G1: everything from here on is downstream of the cache MISS -
    # blocking (OpenSearch/Neo4j/Groq are all sync clients), so it runs off
    # the event loop via run_in_threadpool, never called inline in this
    # async handler.
    result = await run_in_threadpool(run_pipeline, req.query, request_id)

    if result.insufficient_evidence:
        # SS2.5: both recovery attempts failed - return the fallback
        # directly, never generate from evidence already known to be
        # inadequate (no generation call happens on this path at all).
        payload = {
            "request_id": request_id,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "route": result.route,
            "recovery_used": result.recovery_used,
            "cache_hit": False,
        }
        await cache_set(req.query, {k: v for k, v in payload.items() if k not in ("request_id", "cache_hit")})
        payload["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return payload

    # FR21/NFR7: mode is decided from the query text alone, before
    # generation begins - never mid-stream.
    mode = select_mode(req.query)

    if mode == "mode2":
        # FR20: buffered, blocking grounding check gates release. The
        # ungrounded buffered answer is never returned on FAIL.
        answer = await run_in_threadpool(generate_buffered, result.messages)
        check = await run_in_threadpool(check_grounding, answer, result.evidence)
        passed = is_passed(check, DEFAULT_MODE2_PASS_THRESHOLD)
        final_answer = answer if passed else INSUFFICIENT_EVIDENCE_MESSAGE
        payload = {
            "request_id": request_id,
            "answer": final_answer,
            "sources": result.sources if passed else [],
            "grounding": {"mode": "sync", "status": "passed" if passed else "failed", "confidence": check.confidence},
            "route": result.route,
            "recovery_used": result.recovery_used,
            "cache_hit": False,
        }
        if passed:
            await cache_set(req.query, {k: v for k, v in payload.items() if k not in ("request_id", "cache_hit", "grounding")})
        await record_eval_event(request_id, {"route": result.route, "recovery_used": result.recovery_used, "grounded": check.grounded, "confidence": check.confidence})
        payload["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return payload

    # Mode 1 (default)
    loop = asyncio.get_running_loop()
    if not req.stream:
        answer = await run_in_threadpool(generate_buffered, result.messages)
        handle = check_grounding_async(answer, result.evidence)
        handle.future.add_done_callback(lambda f: _log_grounding_result(request_id, result.route, result.recovery_used, loop, f))
        payload = {
            "request_id": request_id,
            "answer": answer,
            "sources": result.sources,
            "route": result.route,
            "recovery_used": result.recovery_used,
            "cache_hit": False,
        }
        await cache_set(req.query, {k: v for k, v in payload.items() if k not in ("request_id", "cache_hit")})
        payload["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return payload

    # Mode 1, streaming (SS9.1) - a plain sync generator handed to
    # StreamingResponse: Starlette runs its next() calls via
    # iterate_in_threadpool (verified 2026-09-16), so this streams
    # progressively without blocking the event loop, no manual bridging
    # needed.
    collected: list[str] = []

    def event_stream():
        yield format_sse_event("sources", {"sources": result.sources})
        for token in generate_streaming(result.messages):
            collected.append(token)
            yield format_sse_event("token", {"text": token})
        handle = check_grounding_async("".join(collected), result.evidence)
        handle.future.add_done_callback(lambda f: _log_grounding_result(request_id, result.route, result.recovery_used, loop, f))
        yield format_sse_event("grounding", {"mode": "async", "status": "pending"})
        yield format_sse_event(
            "done",
            {
                "request_id": request_id,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
                "cache_hit": False,
                "route": result.route,
                "recovery_used": result.recovery_used,
            },
        )

    async def cache_streamed_answer() -> None:
        # Runs after the full response has been sent (Starlette's
        # BackgroundTask contract) - streaming a response and populating
        # the cache from it don't have to be the same synchronous step.
        if collected:
            await cache_set(
                req.query,
                {"answer": "".join(collected), "sources": result.sources, "route": result.route, "recovery_used": result.recovery_used},
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", background=BackgroundTask(cache_streamed_answer))
