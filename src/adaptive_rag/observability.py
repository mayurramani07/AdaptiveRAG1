"""Observability (Phase 8, FR22-26): lightweight, dependency-free tracing
and metrics, plus Redis-backed online eval collection.

Deliberately NOT a full OpenTelemetry SDK integration - SS2.3 names
"OpenTelemetry + free-tier log aggregator" as the intended eventual stack,
but no tracing/metrics backend has been chosen or provisioned yet (the same
kind of external-action gap OpenSearch/Neo4j/Groq/OpenSearch were before
the user signed up for those). This module satisfies the FUNCTIONAL
requirement - every stage's timing/success/failure is observable, and every
request's plan + grounding result is queryable by request_id (FR7's literal
acceptance: "within 5 minutes") - using only what's already provisioned:
structured JSON logs (Phase 1) and Redis (Phase 1). Swap `traced`'s
log-based span for a real OTel span later without changing call sites -
the context-manager shape stays the same either way.

`traced` produces per-request span logs (correlated by request_id, like
`planning.log_plan_decision` already does); `METRICS` is a separate
in-memory *aggregate* registry (no request_id tag - aggregating per-request
would defeat the point of a counter/histogram). `record_eval_event`/
`get_eval_event` reuse Redis (not a new service) for the online eval loop.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_EVAL_TTL_SECONDS = 86400  # 24h - well over FR7's "within 5 minutes" acceptance


@dataclass
class MetricsRegistry:
    """In-memory counters/histograms, tagged by stage name only - a
    process-local stand-in for a real metrics backend (Prometheus/OTel),
    not provisioned yet. `snapshot()` is what a future `/v1/metrics`
    endpoint or exporter would read from."""

    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: Lock = field(default_factory=Lock)

    def increment(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    name: {"count": len(values), "avg": round(sum(values) / len(values), 2), "min": min(values), "max": max(values)}
                    for name, values in self._histograms.items()
                    if values
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


METRICS = MetricsRegistry()


@contextmanager
def traced(stage: str, request_id: str | None = None, **extra):
    """FR22: wraps one pipeline stage with a structured start/end span log
    (correlated by request_id) plus aggregate latency + success/failure
    counts in `METRICS`. Usage: `with traced("dense_retrieval",
    request_id=request_id): ...`."""
    start = time.monotonic()
    logger.info(f"{stage}_start", extra={"stage": stage, "request_id": request_id, **extra})
    try:
        yield
    except Exception:
        duration_ms = round((time.monotonic() - start) * 1000, 1)
        METRICS.increment(f"{stage}.error")
        METRICS.observe(f"{stage}.duration_ms", duration_ms)
        logger.warning(f"{stage}_error", extra={"stage": stage, "request_id": request_id, "duration_ms": duration_ms}, exc_info=True)
        raise
    else:
        duration_ms = round((time.monotonic() - start) * 1000, 1)
        METRICS.increment(f"{stage}.success")
        METRICS.observe(f"{stage}.duration_ms", duration_ms)
        logger.info(f"{stage}_end", extra={"stage": stage, "request_id": request_id, "duration_ms": duration_ms})


class RedisLike(Protocol):
    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> Any: ...


def _eval_key(request_id: str) -> str:
    return f"eval:{request_id}"


async def record_eval_event(request_id: str, event: dict, redis_client: RedisLike | None = None, ttl_seconds: int = DEFAULT_EVAL_TTL_SECONDS) -> None:
    """FR7 acceptance, matched literally: 'Every request's plan + final
    grounding score queryable by request ID within 5 minutes.' Reuses
    Redis (already provisioned, Phase 1) instead of a new log-aggregation
    backend - a write followed immediately by `get_eval_event` returns the
    same data with no aggregation lag at all, well under the 5-minute bar.
    Best-effort, same degradation as `gateway.cache_set`: a write failure
    must not fail the request that triggered it."""
    from adaptive_rag.gateway import get_redis

    client = redis_client or get_redis()
    try:
        await client.set(_eval_key(request_id), json.dumps(event, default=str), ex=ttl_seconds)
    except ConnectionError:
        pass


async def get_eval_event(request_id: str, redis_client: RedisLike | None = None) -> dict | None:
    from adaptive_rag.gateway import get_redis

    client = redis_client or get_redis()
    try:
        raw = await client.get(_eval_key(request_id))
    except ConnectionError:
        return None
    return json.loads(raw) if raw else None


# ponytail: index staleness only (via `ingested_at`, set by
# retrieval.index_chunk) - the most concrete, directly-actionable form of
# "drift" this codebase can check without a production traffic history:
# NG4/SS3's re-ingestion cadence ("derived graph nodes/edges must not
# silently outlive a deleted or superseded source document past that
# cadence"). Embedding-distribution drift and graph-structure drift would
# need historical baselines that don't exist yet - not built here.
DEFAULT_STALENESS_THRESHOLD_HOURS = 168.0  # 7 days - ties to NG4's scheduled-reindex cadence, provisional (SS15 convention)


class OpenSearchAggLike(Protocol):
    def search(self, *, index: str, body: dict) -> dict: ...


def check_index_staleness(client: OpenSearchAggLike | None = None, index: str | None = None, max_age_hours: float = DEFAULT_STALENESS_THRESHOLD_HOURS) -> dict:
    """Flags when the most recently ingested chunk is older than
    `max_age_hours` - a signal that scheduled re-ingestion
    (`retrieval.sync_all`) may not be running, not a guarantee that any
    specific document's content is stale. Verified 2026-09-16 against the
    real Aiven OpenSearch instance: `value_as_string` on a `max` aggregation
    over a `date`-mapped field returns a Z-suffixed ISO timestamp, which
    Python 3.11+'s `datetime.fromisoformat` parses natively."""
    from datetime import UTC, datetime

    from adaptive_rag.retrieval import DEFAULT_INDEX, get_opensearch_client

    search_client = client or get_opensearch_client()
    response = search_client.search(index=index or DEFAULT_INDEX, body={"size": 0, "aggs": {"most_recent": {"max": {"field": "ingested_at"}}}})
    most_recent_str = response["aggregations"]["most_recent"]["value_as_string"]
    if most_recent_str is None:
        return {"stale": True, "reason": "no documents indexed", "most_recent_ingested_at": None, "age_hours": None}

    most_recent = datetime.fromisoformat(most_recent_str)
    age_hours = (datetime.now(UTC) - most_recent).total_seconds() / 3600
    return {"stale": age_hours > max_age_hours, "most_recent_ingested_at": most_recent.isoformat(), "age_hours": round(age_hours, 2)}
