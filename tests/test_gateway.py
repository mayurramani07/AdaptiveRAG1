import pytest
from fastapi.testclient import TestClient

from adaptive_rag import app as app_module


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
def client():
    return TestClient(app_module.app)


def test_missing_api_key_is_rejected(client, fake_redis):
    resp = client.post("/v1/query", json={"query": "hello"})
    assert resp.status_code == 401


def test_empty_query_is_rejected(client, fake_redis):
    resp = client.post(
        "/v1/query", json={"query": "   "}, headers={"x-api-key": "test-key"}
    )
    assert resp.status_code == 400


def test_rate_limit_enforced(client, fake_redis):
    headers = {"x-api-key": "test-key"}
    for _ in range(2):
        resp = client.post("/v1/query", json={"query": "q"}, headers=headers)
        assert resp.status_code == 200
    resp = client.post("/v1/query", json={"query": "q"}, headers=headers)
    assert resp.status_code == 429


def test_cache_hit_skips_pipeline(client, fake_redis, monkeypatch):
    calls = []
    monkeypatch.setattr(
        app_module,
        "_run_pipeline_stub",
        lambda q: calls.append(q) or {"answer": "x", "sources": [], "route": "r", "recovery_used": False},
    )
    headers = {"x-api-key": "test-key"}

    first = client.post("/v1/query", json={"query": "same query"}, headers=headers)
    assert first.status_code == 200
    assert first.json()["cache_hit"] is False

    second = client.post("/v1/query", json={"query": "same query"}, headers=headers)
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True

    assert calls == ["same query"]
