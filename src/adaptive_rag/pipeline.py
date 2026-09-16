"""Full pre-generation pipeline: wires Phases 3-5 together (understand ->
plan -> route -> retrieve -> fuse -> rerank -> grade -> recover). Lives
outside app.py so it's testable as plain Python via dependency injection
(fakes for retrievers/grader/LLM), the same separation gateway.py already
keeps from app.py's HTTP concerns - no FastAPI/TestClient needed to test
this module.

Blocking/synchronous throughout, matching every Phase 3-5 function it
calls (OpenSearch/Neo4j/Groq clients are all sync). The async `/v1/query`
handler in app.py must run `run_pipeline` via `run_in_threadpool` to avoid
blocking FastAPI's event loop - verified this actually matters: Starlette's
own `StreamingResponse` runs a sync generator's `next()` calls through
`iterate_in_threadpool` for exactly this reason (checked its source
2026-09-16), so the same discipline applies here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_rag.generation import build_context, extract_sources
from adaptive_rag.grading import grade_evidence, needs_recovery
from adaptive_rag.ingestion import LLMLike
from adaptive_rag.observability import traced
from adaptive_rag.planning import (
    CircuitBreaker,
    RetrievalPlan,
    Retrievers,
    clamp_top_k,
    execute_plan,
    log_plan_decision,
    plan_query,
    understand_query,
)
from adaptive_rag.recovery import (
    RecoveryStrategies,
    make_graph_expansion_strategy,
    make_internal_re_retrieval_strategy,
    make_query_rewrite_strategy,
    run_recovery,
)
from adaptive_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    GraphRetriever,
    RerankerLike,
    reciprocal_rank_fusion,
    rerank_candidates,
)

# Process-level circuit breaker state (planning.execute_plan's own
# docstring calls for this: "the caller (orchestration) should hold one
# dict for the process" - a fresh dict per call means breakers never
# actually accumulate failures across requests. This module IS that caller.
_BREAKERS: dict[str, CircuitBreaker] = {}


def _route_name(plan: RetrievalPlan) -> str:
    if not (plan.dense or plan.bm25 or plan.graph):
        return "chitchat"
    if plan.graph:
        return "graph-rag"
    if plan.dense and plan.bm25:
        return "hybrid-rag"
    if plan.dense:
        return "simple-rag"
    return "bm25-rag"


def default_retrievers() -> Retrievers:
    """Real retrievers - each lazily grabs its singleton OpenSearch/Neo4j
    client on first `.retrieve()` call, same pattern as `get_redis()`
    elsewhere. No client wiring needed here beyond instantiation."""
    return Retrievers(dense=DenseRetriever(), bm25=BM25Retriever(), graph=GraphRetriever())


def default_recovery_strategies(retrievers: Retrievers) -> RecoveryStrategies:
    """No `external_web_search` default - no provider is chosen or keyed
    yet (documented gap, Phase 5). Query Rewrite/Internal Re-retrieval/
    Graph Expansion all reuse the SAME retrievers as the main request."""
    return RecoveryStrategies(
        query_rewrite=make_query_rewrite_strategy(retrievers),
        internal_re_retrieval=make_internal_re_retrieval_strategy(retrievers),
        graph_expansion=make_graph_expansion_strategy(),
    )


@dataclass
class PipelineResult:
    plan: RetrievalPlan
    route: str
    evidence: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    recovery_used: bool = False
    insufficient_evidence: bool = False


def run_pipeline(
    query: str,
    request_id: str,
    retrievers: Retrievers | None = None,
    recovery_strategies: RecoveryStrategies | None = None,
    grader: RerankerLike | None = None,
    llm: LLMLike | None = None,
) -> PipelineResult:
    """FR4-FR14: the full pre-generation pipeline for one request.
    `insufficient_evidence=True` means the Recovery Planner exhausted its
    2-attempt cap without reaching Correct evidence (SS2.5) - the caller
    must return the fallback message and skip generation entirely, never
    generate from evidence already known to be inadequate."""
    with traced("understand_and_plan", request_id=request_id):
        understanding = understand_query(query)
        plan = plan_query(understanding, query)
        log_plan_decision(request_id, query, plan)
        route = _route_name(plan)

    if route == "chitchat":
        return PipelineResult(plan=plan, route=route, messages=build_context(query, []), sources=[])

    active_retrievers = retrievers if retrievers is not None else default_retrievers()
    with traced("retrieve_fuse_rerank", request_id=request_id, route=route):
        results_by_retriever = execute_plan(plan, active_retrievers, query, filters=understanding.filter_candidates, breakers=_BREAKERS)
        fused = reciprocal_rank_fusion(results_by_retriever, top_k=clamp_top_k(plan.top_k))
        reranked = rerank_candidates(query, fused)
    with traced("grade_evidence", request_id=request_id):
        graded = grade_evidence(query, reranked, grader=grader)

    recovery_used = False
    if needs_recovery(graded):
        with traced("recovery", request_id=request_id):
            strategies = recovery_strategies if recovery_strategies is not None else default_recovery_strategies(active_retrievers)
            recovery_result = run_recovery(query, graded, strategies, grader=grader, graph_was_used=plan.graph)
            graded = recovery_result.evidence
            recovery_used = recovery_result.attempts_used > 0
        if not recovery_result.recovered:
            return PipelineResult(plan=plan, route=route, evidence=graded, recovery_used=recovery_used, insufficient_evidence=True)

    return PipelineResult(
        plan=plan,
        route=route,
        evidence=graded,
        messages=build_context(query, graded),
        sources=extract_sources(graded),
        recovery_used=recovery_used,
    )
