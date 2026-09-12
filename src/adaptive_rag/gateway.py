"""Ingress & cache (Phase 1 / FR1-FR3): auth, rate limiting, exact-match cache.

Redis backs both rate limiting and the cache since it's the one ingress-layer
dependency the architecture already requires (SS4 locked stack) - no new
service for either concern.
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Any, Optional, Protocol

from fastapi import Header, HTTPException

from adaptive_rag.config import get_settings


class RedisLike(Protocol):
    async def get(self, key: str) -> Optional[bytes]: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> Any: ...
    async def incr(self, key: str) -> int: ...
    async def expire(self, key: str, seconds: int) -> Any: ...


@lru_cache
def get_redis() -> RedisLike:
    import redis.asyncio as redis

    settings = get_settings()
    # protocol=2: RESP3 (the redis-py default) sends HELLO on connect, which
    # older Redis builds (e.g. the Windows 5.x port) reject as unknown.
    return redis.from_url(settings.redis_url, decode_responses=True, protocol=2)


async def require_api_key(x_api_key: str = Header(default="")) -> str:
    settings = get_settings()
    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return x_api_key


async def enforce_rate_limit(api_key: str, redis_client: RedisLike | None = None) -> None:
    """Fixed-window counter per API key per minute.

    ponytail: fixed window, not sliding - can allow a short burst across a
    window boundary. Upgrade to a sliding window if that burst matters.
    """
    settings = get_settings()
    client = redis_client or get_redis()
    window = int(time.time() // 60)
    key = f"ratelimit:{api_key}:{window}"
    try:
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, 60)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail="rate limiter unavailable") from exc
    if count > settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="rate limit exceeded")


def _cache_key(query: str) -> str:
    normalized = query.strip().lower()
    return "cache:" + hashlib.sha256(normalized.encode()).hexdigest()


async def cache_get(query: str, redis_client: RedisLike | None = None) -> Optional[dict]:
    client = redis_client or get_redis()
    try:
        raw = await client.get(_cache_key(query))
    except ConnectionError:
        return None  # cache unavailable degrades to MISS, not a hard failure
    return json.loads(raw) if raw else None


async def cache_set(query: str, value: dict, redis_client: RedisLike | None = None) -> None:
    settings = get_settings()
    client = redis_client or get_redis()
    try:
        await client.set(_cache_key(query), json.dumps(value), ex=settings.cache_ttl_seconds)
    except ConnectionError:
        pass  # best-effort; a cache write failure must not fail the request


async def semantic_cache_get(query: str) -> Optional[dict]:
    """No-op until an embedding provider is chosen (project context SS9)."""
    settings = get_settings()
    if not settings.embedding_provider:
        return None
    raise NotImplementedError("semantic cache provider not yet wired")
