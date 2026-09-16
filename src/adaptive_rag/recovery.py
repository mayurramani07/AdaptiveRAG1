"""CRAG Recovery Policy (Phase 5, SS2.5): when evidence grades Ambiguous or
Incorrect, tries recovery strategies in preference order - query rewrite,
internal re-retrieval, graph expansion, external web search - capped at 2
total attempts, with a trust-tier gate that keeps external web search from
ever running while an authoritative internal source still scores
moderately.

`run_recovery` is the testable core: a pure orchestration loop over
injected `RecoveryStrategyFn` callables (same dependency-injection pattern
as `planning.Retrievers`/`retrieval.EmbeddingLike` elsewhere in this
codebase). The `make_*_strategy` builders below are real, usable default
implementations wired to Phase 2-4 infra (Groq's small extraction model for
rewriting, the same retrievers/RRF fusion for re-retrieval, `GraphRetriever`
with `hops=2` for expansion) - but nothing calls them yet, since there's no
live request-orchestration layer in this codebase to wire them into (Phase
6/7's job). External Web Search has no real provider wired in at all - no
search API has been chosen or keyed, the same kind of external-action gap
OpenSearch/Neo4j provisioning were before those got signed up for.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from adaptive_rag.grading import (
    DEFAULT_AMBIGUOUS_THRESHOLD,
    DEFAULT_CORRECT_THRESHOLD,
    grade_evidence,
    needs_recovery,
)
from adaptive_rag.ingestion import LLMLike, Neo4jLike, get_llm_client
from adaptive_rag.pii import redact_pii
from adaptive_rag.planning import (
    TOP_K_DEFAULT,
    RetrievalPlan,
    Retrievers,
    clamp_top_k,
    execute_plan,
)
from adaptive_rag.retrieval import GraphRetriever, RerankerLike, reciprocal_rank_fusion

RECOVERY_STRATEGY_ORDER = ("query_rewrite", "internal_re_retrieval", "graph_expansion", "external_web_search")
MAX_RECOVERY_ATTEMPTS = 2  # FR14, SS2.5 - hard cap, enforced in code below, not just documented
# ponytail: provisional, SS15 lists this as an unset tunable. Placed between
# grading.py's AMBIGUOUS (-3.0) and CORRECT (2.0) thresholds - "moderate"
# sits between "clearly bad" and "clearly good" on the same measured scale.
DEFAULT_MODERATE_THRESHOLD = -1.0
INSUFFICIENT_EVIDENCE_MESSAGE = "I don't have enough reliable information to answer this confidently."

# SS2.5's trust-tier gate, adapted to this codebase's actual source
# taxonomy: the PRD's example keys ("internal_policy_docs", etc.) don't
# correspond to anything concrete here - Dense/BM25/Graph are ALL internal
# retrieval (OpenSearch/Neo4j), so every one of them is authoritative by
# construction. The only non-authoritative source is external web search,
# tagged "supplementary" at the point results are produced (RedactedWebSearch
# below) - so evidence with no explicit trust_tier defaults to authoritative.
TRUST_TIERS = {"dense": "authoritative", "bm25": "authoritative", "graph": "authoritative", "web": "supplementary"}

_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's search query to be clearer and more specific for a document search engine, "
    'without changing its meaning or adding new facts. Respond with strict JSON only: '
    '{"rewritten_query": "..."}. No prose, JSON only.'
)

class WebSearchLike(Protocol):
    def search(self, query: str) -> list[dict]: ...


class RedactedWebSearch:
    """Wraps a WebSearchLike, redacting the query (FR15) before it ever
    reaches the underlying implementation - the redaction guarantee lives
    here structurally, not as a convention callers must remember to apply.
    Also tags every result "supplementary" (SS2.5) at the point of
    creation, so trust-tier tagging can't be forgotten downstream either.
    No concrete WebSearchLike provider is wired in - no external search API
    has been chosen or keyed yet."""

    def __init__(self, backend: WebSearchLike):
        self._backend = backend

    def search(self, query: str) -> list[dict]:
        results = self._backend.search(redact_pii(query))
        return [{**r, "trust_tier": "supplementary"} for r in results]


class RecoveryStrategyFn(Protocol):
    def __call__(self, query: str, prior_evidence: list[dict]) -> tuple[str, list[dict]]: ...


@dataclass
class RecoveryStrategies:
    """Injected per-strategy implementations - mirrors `planning.Retrievers`'
    shape exactly (one optional slot per named strategy). A strategy left
    as None is simply skipped, not an error - same convention `execute_plan`
    uses for a flagged-but-uninjected retriever."""

    query_rewrite: RecoveryStrategyFn | None = None
    internal_re_retrieval: RecoveryStrategyFn | None = None
    graph_expansion: RecoveryStrategyFn | None = None
    external_web_search: RecoveryStrategyFn | None = None


@dataclass
class RecoveryResult:
    query: str
    evidence: list[dict]
    attempts_used: int
    strategies_tried: list[str] = field(default_factory=list)
    recovered: bool = False


def _authoritative_source_scored_moderately(graded_evidence: list[dict], moderate_threshold: float) -> bool:
    return any(
        e.get("trust_tier", "authoritative") == "authoritative" and e.get("grader_score", float("-inf")) >= moderate_threshold
        for e in graded_evidence
    )


def run_recovery(
    query: str,
    graded_evidence: list[dict],
    strategies: RecoveryStrategies,
    grader: RerankerLike | None = None,
    graph_was_used: bool = False,
    max_attempts: int = MAX_RECOVERY_ATTEMPTS,
    moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
    correct_threshold: float = DEFAULT_CORRECT_THRESHOLD,
    ambiguous_threshold: float = DEFAULT_AMBIGUOUS_THRESHOLD,
) -> RecoveryResult:
    """FR13/FR14: on Ambiguous/Incorrect evidence, tries strategies in
    RECOVERY_STRATEGY_ORDER, stopping as soon as evidence grades Correct or
    `max_attempts` real attempts have been made - whichever comes first. A
    strategy that isn't applicable (no callable injected, Graph Expansion
    when Graph wasn't used, or External Web Search while an authoritative
    source still scores >= moderate_threshold - the SS2.5 trust-tier gate)
    is SKIPPED without consuming an attempt; only an actually-executed
    strategy counts against the cap. If both attempts fail (or none were
    applicable), `recovered` is False - the caller should return
    INSUFFICIENT_EVIDENCE_MESSAGE, never the last, still-bad evidence."""
    if not needs_recovery(graded_evidence):
        return RecoveryResult(query=query, evidence=graded_evidence, attempts_used=0, recovered=True)

    current_query = query
    current_evidence = graded_evidence
    attempts_used = 0
    strategies_tried: list[str] = []

    for name in RECOVERY_STRATEGY_ORDER:
        if attempts_used >= max_attempts:
            break
        fn = getattr(strategies, name)
        if fn is None:
            continue
        if name == "graph_expansion" and not graph_was_used:
            continue
        if name == "external_web_search" and _authoritative_source_scored_moderately(current_evidence, moderate_threshold):
            continue

        current_query, raw_evidence = fn(current_query, current_evidence)
        attempts_used += 1
        strategies_tried.append(name)
        current_evidence = grade_evidence(current_query, raw_evidence, grader=grader, correct_threshold=correct_threshold, ambiguous_threshold=ambiguous_threshold)

        if not needs_recovery(current_evidence):
            return RecoveryResult(query=current_query, evidence=current_evidence, attempts_used=attempts_used, strategies_tried=strategies_tried, recovered=True)

    return RecoveryResult(
        query=current_query,
        evidence=current_evidence,
        attempts_used=attempts_used,
        strategies_tried=strategies_tried,
        recovered=not needs_recovery(current_evidence),
    )


def _run_and_fuse(retrievers: Retrievers, query: str, top_k: int) -> list[dict]:
    plan = RetrievalPlan(
        dense=retrievers.dense is not None,
        bm25=retrievers.bm25 is not None,
        graph=retrievers.graph is not None,
        freshness=False,
        apply_filters=False,
        top_k=clamp_top_k(top_k),
    )
    results_by_retriever = execute_plan(plan, retrievers, query)
    return reciprocal_rank_fusion(results_by_retriever)


def make_query_rewrite_strategy(retrievers: Retrievers, llm: LLMLike | None = None) -> RecoveryStrategyFn:
    """Cheapest strategy (SS2.5): asks a small model to reformulate the
    query, then re-runs the same retrievers with the new query text. Uses
    `get_llm_client()`/`GROQ_EXTRACTION_MODEL` - never the generation model
    (FR18/NFR4) - the same small-tier LLM already used for ingestion-time
    extraction, no new model added for this."""

    def strategy(query: str, prior_evidence: list[dict]) -> tuple[str, list[dict]]:
        client = llm or get_llm_client()
        try:
            data = json.loads(client.complete_json(_REWRITE_SYSTEM_PROMPT, query))
        except (json.JSONDecodeError, TypeError):
            data = None
        rewritten = data.get("rewritten_query") if isinstance(data, dict) else None
        new_query = rewritten if isinstance(rewritten, str) and rewritten.strip() else query
        return new_query, _run_and_fuse(retrievers, new_query, TOP_K_DEFAULT)

    return strategy


def make_internal_re_retrieval_strategy(retrievers: Retrievers, widen_factor: int = 2) -> RecoveryStrategyFn:
    """SS2.5: "widen internal search... larger candidate pool" - re-runs
    the same retrievers with a wider top_k (still clamped to
    TOP_K_HARD_CAP by `_run_and_fuse`/`clamp_top_k`, same defense-in-depth
    as the normal request path). Per-retriever weighting isn't implemented
    - RRF fusion already treats all fired retrievers equally; a weighted
    variant is a further refinement, not built here."""

    def strategy(query: str, prior_evidence: list[dict]) -> tuple[str, list[dict]]:
        return query, _run_and_fuse(retrievers, query, TOP_K_DEFAULT * widen_factor)

    return strategy


def make_graph_expansion_strategy(driver: Neo4jLike | None = None, hops: int = 2) -> RecoveryStrategyFn:
    """SS2.5: "expand traversal depth (e.g. 1-hop to 2-hop)" - a real
    `GraphRetriever(hops=2)` call (live-verified against Neo4j AuraDB
    2026-09-16: correctly reaches a second document through a RELATED_TO
    edge that 1-hop retrieval misses)."""

    def strategy(query: str, prior_evidence: list[dict]) -> tuple[str, list[dict]]:
        return query, GraphRetriever(driver=driver, hops=hops).retrieve(query, top_k=TOP_K_DEFAULT, filters=None)

    return strategy


def make_external_web_search_strategy(backend: WebSearchLike) -> RecoveryStrategyFn:
    """SS2.5: only reached if the trust-tier gate in `run_recovery` permits
    it. `backend` is caller-supplied - no web search provider is chosen or
    keyed in this codebase yet."""

    def strategy(query: str, prior_evidence: list[dict]) -> tuple[str, list[dict]]:
        return query, RedactedWebSearch(backend).search(query)

    return strategy
