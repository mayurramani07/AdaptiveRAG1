from fastapi import FastAPI

from adaptive_rag.logging import configure_logging

configure_logging()

app = FastAPI(title="Adaptive RAG")


@app.get("/v1/health")
def health() -> dict:
    # Real per-dependency reachability checks land in Phase 1 (FR1-3) and
    # Phase 4 (retrievers) as each client is built.
    return {"status": "ok"}
