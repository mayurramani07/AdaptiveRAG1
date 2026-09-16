from concurrent.futures import Future

import pytest
from fastapi.testclient import TestClient

from adaptive_rag import app as app_module
from adaptive_rag.grounding import AsyncGroundingHandle, GroundingCheck
from adaptive_rag.pipeline import PipelineResult
from adaptive_rag.planning import RetrievalPlan


class FakeRedis:
    """Minimal in-memory stand-in for the redis.asyncio client used in tests."""

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


def _fake_pipeline_result(**overrides):
    plan = RetrievalPlan(dense=True, bm25=True, graph=False, freshness=False, apply_filters=False, top_k=20)
    defaults = {
        "plan": plan,
        "route": "hybrid-rag",
        "evidence": [{"id": "a", "doc_id": "d", "chunk_id": "a", "text": "source text"}],
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        "sources": [{"id": "a", "doc_id": "d", "trust_tier": "authoritative", "score": 5.0}],
        "recovery_used": False,
        "insufficient_evidence": False,
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)


def _resolved_grounding_handle(grounded=True, confidence=0.9):
    future = Future()
    future.set_result(GroundingCheck(grounded=grounded, confidence=confidence))
    return AsyncGroundingHandle(future=future)


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")


@pytest.fixture
def fake_redis(monkeypatch):
    redis_instance = FakeRedis()
    from adaptive_rag import gateway

    monkeypatch.setattr(gateway, "get_redis", lambda: redis_instance)
    return redis_instance


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Patches everything downstream of the cache-MISS branch so ingress
    tests (auth/rate-limit/cache) stay fast and deterministic - no real
    OpenSearch/Neo4j/Groq calls. Mode/generation/grounding-specific
    behavior gets its own dedicated coverage in test_app.py."""
    calls = {"run_pipeline": []}
    monkeypatch.setattr(app_module, "run_pipeline", lambda q, rid: calls["run_pipeline"].append(q) or _fake_pipeline_result())
    monkeypatch.setattr(app_module, "select_mode", lambda q: "mode1")
    monkeypatch.setattr(app_module, "generate_buffered", lambda messages: "a generated answer")
    monkeypatch.setattr(app_module, "generate_streaming", lambda messages: iter(["a ", "generated ", "answer"]))
    monkeypatch.setattr(app_module, "check_grounding_async", lambda answer, evidence: _resolved_grounding_handle())
    return calls


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_missing_api_key_is_rejected(client, fake_redis, fake_pipeline):
    resp = client.post("/v1/query", json={"query": "hello"})
    assert resp.status_code == 401


def test_empty_query_is_rejected(client, fake_redis, fake_pipeline):
    resp = client.post(
        "/v1/query", json={"query": "   "}, headers={"x-api-key": "test-key"}
    )
    assert resp.status_code == 400


def test_rate_limit_enforced(client, fake_redis, fake_pipeline):
    headers = {"x-api-key": "test-key"}
    for _ in range(2):
        resp = client.post("/v1/query", json={"query": "q", "stream": False}, headers=headers)
        assert resp.status_code == 200
    resp = client.post("/v1/query", json={"query": "q", "stream": False}, headers=headers)
    assert resp.status_code == 429


def test_cache_hit_skips_pipeline(client, fake_redis, fake_pipeline):
    headers = {"x-api-key": "test-key"}

    first = client.post("/v1/query", json={"query": "same query", "stream": False}, headers=headers)
    assert first.status_code == 200
    assert first.json()["cache_hit"] is False

    second = client.post("/v1/query", json={"query": "same query", "stream": False}, headers=headers)
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True

    assert fake_pipeline["run_pipeline"] == ["same query"]  # pipeline only ran once - not on the cache HIT
