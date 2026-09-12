import time
import uuid

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from adaptive_rag.gateway import cache_get, cache_set, enforce_rate_limit, require_api_key
from adaptive_rag.logging import configure_logging

configure_logging()

app = FastAPI(title="Adaptive RAG")


@app.get("/v1/health")
def health() -> dict:
    # Real per-dependency reachability checks land in Phase 4 (retrievers)
    # and Phase 6 (Groq) as each client is built.
    return {"status": "ok"}


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str | None = None
    stream: bool = True


def _run_pipeline_stub(query: str) -> dict:
    # Placeholder until the planner/retrieval/generation stages exist
    # (Phase 2-6). Phase 1 only proves auth, rate limiting, and caching.
    return {
        "answer": "pipeline not yet implemented (Phase 2-6)",
        "sources": [],
        "route": "unimplemented",
        "recovery_used": False,
    }


@app.post("/v1/query")
async def query(req: QueryRequest, api_key: str = Depends(require_api_key)) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    await enforce_rate_limit(api_key)

    start = time.monotonic()
    cached = await cache_get(req.query)
    if cached is not None:
        cached["cache_hit"] = True
        cached["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return cached

    result = _run_pipeline_stub(req.query)
    result["request_id"] = str(uuid.uuid4())
    result["cache_hit"] = False
    await cache_set(req.query, {k: v for k, v in result.items() if k != "cache_hit"})
    result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    return result
