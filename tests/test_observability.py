import logging
from datetime import UTC, datetime, timedelta

import pytest

from adaptive_rag.observability import (
    METRICS,
    MetricsRegistry,
    check_index_staleness,
    get_eval_event,
    record_eval_event,
    traced,
)


class FakeOpenSearchAgg:
    """Fake matching the real OpenSearch max-aggregation response shape,
    verified 2026-09-16 against the live Aiven instance - a `value_as_string`
    key with a Z-suffixed ISO timestamp alongside `value` (epoch ms)."""

    def __init__(self, most_recent: datetime | None):
        self._most_recent = most_recent

    def search(self, *, index, body):
        if self._most_recent is None:
            return {"aggregations": {"most_recent": {"value": None, "value_as_string": None}}}
        iso = self._most_recent.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return {"aggregations": {"most_recent": {"value": self._most_recent.timestamp() * 1000, "value_as_string": iso}}}


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.fixture(autouse=True)
def clean_metrics():
    METRICS.reset()
    yield
    METRICS.reset()


# ---------------------------------------------------------------------------
# traced (FR22) - structured span logs
# ---------------------------------------------------------------------------


def test_traced_logs_start_and_end_with_request_id(caplog):
    with caplog.at_level(logging.INFO, logger="adaptive_rag.observability"), traced("my_stage", request_id="req-1"):
        pass
    messages = [r.message for r in caplog.records]
    assert "my_stage_start" in messages
    assert "my_stage_end" in messages
    end_record = next(r for r in caplog.records if r.message == "my_stage_end")
    assert end_record.request_id == "req-1"
    assert end_record.duration_ms >= 0


def test_traced_logs_error_and_still_raises(caplog):
    with caplog.at_level(logging.INFO, logger="adaptive_rag.observability"), pytest.raises(ValueError), traced("failing_stage", request_id="req-2"):
        raise ValueError("boom")
    error_record = next(r for r in caplog.records if r.message == "failing_stage_error")
    assert error_record.request_id == "req-2"


def test_traced_records_success_metric():
    with traced("stage_a"):
        pass
    snapshot = METRICS.snapshot()
    assert snapshot["counters"]["stage_a.success"] == 1
    assert "stage_a.duration_ms" in snapshot["histograms"]


def test_traced_records_error_metric_not_success_metric():
    with pytest.raises(ValueError), traced("stage_b"):
        raise ValueError("boom")
    snapshot = METRICS.snapshot()
    assert snapshot["counters"]["stage_b.error"] == 1
    assert "stage_b.success" not in snapshot["counters"]


# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------


def test_metrics_registry_increment_accumulates():
    registry = MetricsRegistry()
    registry.increment("x")
    registry.increment("x")
    registry.increment("x")
    assert registry.snapshot()["counters"]["x"] == 3


def test_metrics_registry_observe_computes_summary_stats():
    registry = MetricsRegistry()
    for v in (10.0, 20.0, 30.0):
        registry.observe("latency", v)
    summary = registry.snapshot()["histograms"]["latency"]
    assert summary == {"count": 3, "avg": 20.0, "min": 10.0, "max": 30.0}


def test_metrics_registry_reset_clears_state():
    registry = MetricsRegistry()
    registry.increment("x")
    registry.observe("y", 1.0)
    registry.reset()
    assert registry.snapshot() == {"counters": {}, "histograms": {}}


# ---------------------------------------------------------------------------
# Online eval collection (FR7) - Redis-backed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_and_get_eval_event_round_trip():
    redis = FakeRedis()
    await record_eval_event("req-1", {"route": "hybrid-rag", "grounded": True, "confidence": 0.9}, redis_client=redis)
    event = await get_eval_event("req-1", redis_client=redis)
    assert event == {"route": "hybrid-rag", "grounded": True, "confidence": 0.9}


@pytest.mark.asyncio
async def test_get_eval_event_returns_none_when_missing():
    redis = FakeRedis()
    event = await get_eval_event("nonexistent", redis_client=redis)
    assert event is None


@pytest.mark.asyncio
async def test_record_eval_event_sets_a_ttl():
    class SpyRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.set_calls = []

        async def set(self, key, value, ex=None):
            self.set_calls.append((key, value, ex))
            self.store[key] = value

    redis = SpyRedis()
    await record_eval_event("req-1", {"x": 1}, redis_client=redis, ttl_seconds=3600)
    assert redis.set_calls[0][2] == 3600


@pytest.mark.asyncio
async def test_record_eval_event_degrades_gracefully_on_redis_failure():
    class FailingRedis:
        async def set(self, key, value, ex=None):
            raise ConnectionError("redis down")

    # must not raise - a monitoring write failure must never fail the request
    await record_eval_event("req-1", {"x": 1}, redis_client=FailingRedis())


@pytest.mark.asyncio
async def test_get_eval_event_degrades_gracefully_on_redis_failure():
    class FailingRedis:
        async def get(self, key):
            raise ConnectionError("redis down")

    result = await get_eval_event("req-1", redis_client=FailingRedis())
    assert result is None


# ---------------------------------------------------------------------------
# Drift detection (index staleness)
# ---------------------------------------------------------------------------


def test_check_index_staleness_fresh_index_not_stale():
    client = FakeOpenSearchAgg(most_recent=datetime.now(UTC) - timedelta(hours=1))
    result = check_index_staleness(client=client, index="documents", max_age_hours=168.0)
    assert result["stale"] is False
    assert result["age_hours"] < 2


def test_check_index_staleness_old_index_is_stale():
    client = FakeOpenSearchAgg(most_recent=datetime.now(UTC) - timedelta(hours=200))
    result = check_index_staleness(client=client, index="documents", max_age_hours=168.0)
    assert result["stale"] is True
    assert result["age_hours"] > 168


def test_check_index_staleness_empty_index_is_stale():
    client = FakeOpenSearchAgg(most_recent=None)
    result = check_index_staleness(client=client, index="documents")
    assert result["stale"] is True
    assert result["reason"] == "no documents indexed"


def test_check_index_staleness_custom_threshold():
    client = FakeOpenSearchAgg(most_recent=datetime.now(UTC) - timedelta(hours=2))
    result = check_index_staleness(client=client, index="documents", max_age_hours=1.0)
    assert result["stale"] is True  # 2h old, but threshold is 1h
