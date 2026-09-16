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
