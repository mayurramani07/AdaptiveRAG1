"""Full /v1/query wiring: mode routing, SSE format, insufficient-evidence
fallback, Mode 2 grounding gate. Everything downstream of the cache MISS is
faked (no real OpenSearch/Neo4j/Groq) - this tests app.py's orchestration
logic, not Phases 3-7's own behavior (each already has its own test suite).
"""
from concurrent.futures import Future

import pytest
from fastapi.testclient import TestClient

from adaptive_rag import app as app_module
from adaptive_rag.grounding import AsyncGroundingHandle, GroundingCheck
from adaptive_rag.pipeline import PipelineResult
from adaptive_rag.planning import RetrievalPlan


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def expire(self, key, seconds):
        pass


def _plan(dense=True, bm25=True, graph=False):
    return RetrievalPlan(dense=dense, bm25=bm25, graph=graph, freshness=False, apply_filters=False, top_k=20)


def _result(**overrides):
    defaults = {
        "plan": _plan(),
        "route": "hybrid-rag",
        "evidence": [{"id": "a", "doc_id": "d", "chunk_id": "a", "text": "source text"}],
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        "sources": [{"id": "a", "doc_id": "d", "trust_tier": "authoritative", "score": 5.0}],
        "recovery_used": False,
        "insufficient_evidence": False,
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)


def _resolved_handle(grounded=True, confidence=0.9):
    future = Future()
    future.set_result(GroundingCheck(grounded=grounded, confidence=confidence))
    return AsyncGroundingHandle(future=future)


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    from adaptive_rag import gateway

    redis_instance = FakeRedis()
    monkeypatch.setattr(gateway, "get_redis", lambda: redis_instance)
    return redis_instance


@pytest.fixture
def client():
    return TestClient(app_module.app)


HEADERS = {"x-api-key": "test-key"}


# ---------------------------------------------------------------------------
# CORS (Phase 10, frontend integration, PRD SS10) - allowlist, never "*"
# ---------------------------------------------------------------------------


def test_cors_allows_configured_frontend_origin(client):
    resp = client.options(
        "/v1/query",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-api-key"},
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unlisted_origin(client):
    resp = client.options(
        "/v1/query",
        headers={"Origin": "http://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert resp.headers.get("access-control-allow-origin") is None


# ---------------------------------------------------------------------------
# GET /v1/health (Phase 9 hardening, SS9.2)
# ---------------------------------------------------------------------------


def test_health_reports_ok_when_all_dependencies_reachable(client, monkeypatch):
    async def fake_check(**kwargs):
        return {"redis": "ok", "neo4j": "ok", "opensearch": "ok", "groq": "ok"}

    monkeypatch.setattr(app_module, "check_dependency_health", fake_check)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "dependencies": {"redis": "ok", "neo4j": "ok", "opensearch": "ok", "groq": "ok"}}


def test_health_reports_degraded_when_one_dependency_down(client, monkeypatch):
    async def fake_check(**kwargs):
        return {"redis": "ok", "neo4j": "error: connection refused", "opensearch": "ok", "groq": "ok"}

    monkeypatch.setattr(app_module, "check_dependency_health", fake_check)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# POST /v1/documents (Phase 11, FR-ING5, reverses NG8)
# ---------------------------------------------------------------------------


def _patch_ingestion(monkeypatch, ingest_calls):
    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: "fake-driver")
    monkeypatch.setattr(app_module, "get_llm_client", lambda: "fake-llm")

    def fake_ingest_and_index(driver, doc_id, text, llm=None, metadata=None):
        ingest_calls.append((doc_id, text, metadata))

    monkeypatch.setattr(app_module, "ingest_and_index", fake_ingest_and_index)


def test_upload_document_indexes_a_txt_file(client, monkeypatch):
    calls = []
    _patch_ingestion(monkeypatch, calls)

    resp = client.post(
        "/v1/documents",
        files={"file": ("notes.txt", b"Acme Corporation's refund policy allows returns within 30 days.", "text/plain")},
        headers=HEADERS,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "notes.txt"
    assert body["doc_id"].startswith("notes-")
    assert body["chunks_indexed"] == 1
    assert len(calls) == 1
    assert calls[0][1] == "Acme Corporation's refund policy allows returns within 30 days."


def test_upload_document_passes_through_metadata(client, monkeypatch):
    calls = []
    _patch_ingestion(monkeypatch, calls)

    resp = client.post(
        "/v1/documents",
        files={"file": ("policy.txt", b"some policy text", "text/plain")},
        data={"doc_type": "policy", "department": "finance"},
        headers=HEADERS,
    )

    assert resp.status_code == 200
    assert calls[0][2] == {"doc_type": "policy", "department": "finance"}


def test_upload_document_rejects_unsupported_file_type(client, monkeypatch):
    calls = []
    _patch_ingestion(monkeypatch, calls)

    resp = client.post("/v1/documents", files={"file": ("image.png", b"\x89PNG", "image/png")}, headers=HEADERS)

    assert resp.status_code == 400
    assert calls == []


def test_upload_document_rejects_empty_file(client, monkeypatch):
    calls = []
    _patch_ingestion(monkeypatch, calls)

    resp = client.post("/v1/documents", files={"file": ("empty.txt", b"   \n\n  ", "text/plain")}, headers=HEADERS)

    assert resp.status_code == 400
    assert "no extractable text" in resp.json()["detail"]
    assert calls == []


def test_upload_document_accepts_a_document_slightly_above_the_old_8000_char_limit(client, monkeypatch):
    # Regression test for the bug this fix addresses: an 8,000-character
    # cap was previously applied to the whole raw document before chunking
    # ever ran. A ~15,000-character document is a perfectly reasonable
    # document and must now be accepted.
    calls = []
    _patch_ingestion(monkeypatch, calls)

    text = ("word " * 3000).encode()  # ~15,000 chars - was rejected before this fix
    resp = client.post("/v1/documents", files={"file": ("medium.txt", text, "text/plain")}, headers=HEADERS)

    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] > 1  # actually went through chunking, not a single blob
    assert len(calls) == 1


def test_upload_document_accepts_a_100k_char_document_and_splits_it_into_many_chunks(client, monkeypatch):
    # The literal scenario reported: a ~106k-character real-world PDF (NIST
    # AI RMF) must be accepted and chunked, not rejected outright.
    calls = []
    _patch_ingestion(monkeypatch, calls)

    sentence = "The quick brown fox jumps over the lazy dog. "
    text = (sentence * (106_000 // len(sentence))).encode()
    assert 100_000 < len(text) < 110_000

    resp = client.post("/v1/documents", files={"file": ("nist-ai-rmf.txt", text, "text/plain")}, headers=HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_indexed"] > 50  # genuinely split into many chunks, not one
    assert len(calls) == 1
    assert calls[0][1] == text.decode().strip()  # full text reached ingestion, nothing silently truncated


def test_upload_document_rejects_genuinely_unbounded_document(client, monkeypatch):
    # Document-level validation still exists - it's just no longer scoped
    # to "any real document", only to genuinely unsafe/unbounded input.
    calls = []
    _patch_ingestion(monkeypatch, calls)

    text = ("x " * 300_000).encode()  # ~600,000 chars - over MAX_DOCUMENT_CHARS (500,000)
    resp = client.post("/v1/documents", files={"file": ("huge.txt", text, "text/plain")}, headers=HEADERS)

    assert resp.status_code == 413
    assert "document too large" in resp.json()["detail"]
    assert calls == []


def test_upload_document_rejects_when_chunk_count_exceeds_the_synchronous_processing_cap(client, monkeypatch):
    # A document under MAX_DOCUMENT_CHARS can still produce more chunks
    # than can reasonably be processed inside one blocking HTTP request
    # (2 LLM calls/chunk) - that must be caught by the chunk-count cap, not
    # the document-level char cap (this text is well under 500,000 chars).
    calls = []
    _patch_ingestion(monkeypatch, calls)

    text = ("word " * 30_200).encode()  # ~151,000 chars, ~151 chunks at 200 words/chunk
    assert len(text) < app_module.MAX_DOCUMENT_CHARS

    resp = client.post("/v1/documents", files={"file": ("toolong.txt", text, "text/plain")}, headers=HEADERS)

    assert resp.status_code == 413
    assert "too many chunks" in resp.json()["detail"]
    assert calls == []


def test_upload_document_requires_auth(client, monkeypatch):
    calls = []
    _patch_ingestion(monkeypatch, calls)

    resp = client.post("/v1/documents", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert resp.status_code == 401
    assert calls == []


def test_upload_document_two_uploads_of_same_filename_get_different_doc_ids(client, monkeypatch):
    calls = []
    _patch_ingestion(monkeypatch, calls)

    resp1 = client.post("/v1/documents", files={"file": ("notes.txt", b"version one", "text/plain")}, headers=HEADERS)
    resp2 = client.post("/v1/documents", files={"file": ("notes.txt", b"version two", "text/plain")}, headers=HEADERS)

    assert resp1.json()["doc_id"] != resp2.json()["doc_id"]  # NG8: no update-in-place


# ---------------------------------------------------------------------------
# Mode 1, buffered (stream=False)
# ---------------------------------------------------------------------------


def test_mode1_buffered_returns_generated_answer(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode1")
    monkeypatch.setattr(app_module, "generate_buffered", lambda messages: "the buffered answer")
    monkeypatch.setattr(app_module, "check_grounding_async", lambda answer, evidence: _resolved_handle())

    resp = client.post("/v1/query", json={"query": "hello", "stream": False}, headers=HEADERS)
    body = resp.json()

    assert resp.status_code == 200
    assert body["answer"] == "the buffered answer"
    assert body["route"] == "hybrid-rag"
    assert body["recovery_used"] is False
    assert "request_id" in body
    assert "latency_ms" in body


# ---------------------------------------------------------------------------
# Mode 1, streaming (SSE, the default)
# ---------------------------------------------------------------------------


def test_mode1_streaming_emits_sse_events_in_order(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode1")
    monkeypatch.setattr(app_module, "generate_streaming", lambda messages: iter(["Hello", " world"]))
    monkeypatch.setattr(app_module, "check_grounding_async", lambda answer, evidence: _resolved_handle())

    resp = client.post("/v1/query", json={"query": "hello"}, headers=HEADERS)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    events = [line for line in body.split("\n") if line.startswith("event:")]
    assert events == ["event: sources", "event: token", "event: token", "event: grounding", "event: done"]
    assert '"text": "Hello"' in body
    assert '"text": " world"' in body


def test_mode1_streaming_caches_the_full_answer_after_response(client, monkeypatch):
    cache_calls = []

    async def fake_cache_set(query, value):
        cache_calls.append((query, value))

    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode1")
    monkeypatch.setattr(app_module, "generate_streaming", lambda messages: iter(["Hello", " world"]))
    monkeypatch.setattr(app_module, "check_grounding_async", lambda answer, evidence: _resolved_handle())
    monkeypatch.setattr(app_module, "cache_set", fake_cache_set)

    resp = client.post("/v1/query", json={"query": "hello"}, headers=HEADERS)
    assert resp.status_code == 200
    # TestClient fully drains the stream (including background tasks) before returning
    assert cache_calls == [("hello", {"answer": "Hello world", "sources": _result().sources, "route": "hybrid-rag", "recovery_used": False})]


# ---------------------------------------------------------------------------
# Mode 2 - buffered + grounding-gated release (FR20)
# ---------------------------------------------------------------------------


def test_mode2_releases_answer_on_grounding_pass(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode2")
    monkeypatch.setattr(app_module, "generate_buffered", lambda messages: "a grounded medical answer")
    monkeypatch.setattr(app_module, "check_grounding", lambda answer, evidence: GroundingCheck(grounded=True, confidence=0.95))

    resp = client.post("/v1/query", json={"query": "what dosage of X is safe"}, headers=HEADERS)
    body = resp.json()

    assert resp.status_code == 200
    assert body["answer"] == "a grounded medical answer"
    assert body["grounding"] == {"mode": "sync", "status": "passed", "confidence": 0.95}


def test_mode2_never_leaks_raw_answer_on_grounding_fail(client, monkeypatch):
    """FR20 acceptance, matched exactly at the HTTP layer: a test forcing
    FAIL confirms the fallback response is returned and the ungrounded
    buffered answer is never sent."""
    from adaptive_rag.recovery import INSUFFICIENT_EVIDENCE_MESSAGE

    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode2")
    monkeypatch.setattr(app_module, "generate_buffered", lambda messages: "a hallucinated ungrounded claim")
    monkeypatch.setattr(app_module, "check_grounding", lambda answer, evidence: GroundingCheck(grounded=False, confidence=0.9))

    resp = client.post("/v1/query", json={"query": "what dosage of X is safe"}, headers=HEADERS)
    body = resp.json()

    assert resp.status_code == 200
    assert body["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert "hallucinated" not in resp.text
    assert body["grounding"]["status"] == "failed"
    assert body["sources"] == []


def test_mode2_does_not_cache_a_failed_grounding_result(client, monkeypatch):
    cache_calls = []

    async def fake_cache_set(query, value):
        cache_calls.append((query, value))

    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode2")
    monkeypatch.setattr(app_module, "generate_buffered", lambda messages: "ungrounded")
    monkeypatch.setattr(app_module, "check_grounding", lambda answer, evidence: GroundingCheck(grounded=False, confidence=0.9))
    monkeypatch.setattr(app_module, "cache_set", fake_cache_set)

    client.post("/v1/query", json={"query": "what dosage of X is safe"}, headers=HEADERS)
    assert cache_calls == []


# ---------------------------------------------------------------------------
# Insufficient evidence (Recovery Planner exhausted, SS2.5)
# ---------------------------------------------------------------------------


def test_insufficient_evidence_returns_fallback_and_skips_generation(client, monkeypatch):
    from adaptive_rag.recovery import INSUFFICIENT_EVIDENCE_MESSAGE

    generation_called = []
    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result(insufficient_evidence=True, recovery_used=True, messages=[], sources=[]))
    monkeypatch.setattr(app_module, "generate_buffered", lambda messages: generation_called.append(1) or "should never be called")
    monkeypatch.setattr(app_module, "generate_streaming", lambda messages: generation_called.append(1) or iter([]))

    resp = client.post("/v1/query", json={"query": "obscure query", "stream": False}, headers=HEADERS)
    body = resp.json()

    assert resp.status_code == 200
    assert body["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert body["recovery_used"] is True
    assert generation_called == []  # no generation call was ever made


# ---------------------------------------------------------------------------
# Online eval collection (FR7) - real record_eval_event, real (fake) Redis
# ---------------------------------------------------------------------------


def test_mode2_records_eval_event_in_redis(client, fake_redis, monkeypatch):
    """FR7 acceptance, matched literally: the grounding result is queryable
    by request_id via Redis (already provisioned, Phase 1) immediately
    after the response - not mocked away, `record_eval_event` runs for
    real against the fake Redis client."""
    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode2")
    monkeypatch.setattr(app_module, "generate_buffered", lambda messages: "a grounded answer")
    monkeypatch.setattr(app_module, "check_grounding", lambda answer, evidence: GroundingCheck(grounded=True, confidence=0.88))

    resp = client.post("/v1/query", json={"query": "what dosage of X is safe"}, headers=HEADERS)
    request_id = resp.json()["request_id"]

    assert fake_redis.store[f"eval:{request_id}"] is not None
    import json

    stored = json.loads(fake_redis.store[f"eval:{request_id}"])
    assert stored == {"route": "hybrid-rag", "recovery_used": False, "grounded": True, "confidence": 0.88}


def test_mode1_streaming_records_eval_event_after_response(client, fake_redis, monkeypatch):
    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: _result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode1")
    monkeypatch.setattr(app_module, "generate_streaming", lambda messages: iter(["Hello"]))
    monkeypatch.setattr(app_module, "check_grounding_async", lambda answer, evidence: _resolved_handle(grounded=True, confidence=0.95))

    resp = client.post("/v1/query", json={"query": "hello"}, headers=HEADERS)

    import json

    done_line = next(line for line in resp.text.split("\n") if line.startswith("data:") and "request_id" in line)
    request_id_value = json.loads(done_line[len("data: ") :])["request_id"]
    assert f"eval:{request_id_value}" in fake_redis.store
