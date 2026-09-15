"""Query understanding, planning, and routing (Phase 3, SS2.4).

The Query Planner (`plan_query`) is a rule-based multi-label classifier, not
free-text LLM generation (non-negotiable principle #2, FR5) - and
deliberately not an LLM call at all, unlike ingestion's extractors. Query
understanding + planning sit on the synchronous per-request hot path
(SS5/SS6 latency targets), while ingestion is offline batch - an LLM
round-trip here would blow every route's p50 target. `plan_query`'s rules
are a baseline pending real labeled training data (FR5 acceptance: "revisit
before go-live", same placeholder-threshold convention as the rest of this
codebase); swap it for a trained classifier later without touching
`understand_query` or the router.

Entity/date detection (`understand_query`) uses spaCy's local NER model
(en_core_web_sm) instead of regex - a real trained model, still local/free/
fast enough for the hot path (no network call, no cost). dateparser then
normalizes whatever spaCy tags as DATE/TIME into an actual date, rather than
running dateparser's free-text search over the whole query directly - that
mode is too noisy (e.g. it matched the word "me" as a date in testing).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Protocol

import dateparser
import spacy

TOP_K_DEFAULT = 20
TOP_K_HARD_CAP = 30

logger = logging.getLogger(__name__)

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "FAC", "LOC", "EVENT"}
_DATE_LABELS = {"DATE", "TIME"}
_FRESHNESS_WORDS = {"recent", "recently", "latest", "newest", "current", "today", "now", "this week", "this month", "this year", "up to date", "up-to-date"}
_RELATIONAL_WORDS = {"relationship", "related", "relates", "connected", "connection", "between", "works with", "reports to", "compare", "versus", "vs"}
# ponytail: static keyword lists as filter-candidate categories - no real
# document taxonomy exists yet. Replace with the actual field/value schema
# once the document metadata model is decided. Tuples, not sets: the first
# match wins below, and set iteration order isn't stable across runs
# (string hash randomization) - a query matching two keywords at once would
# otherwise pick a different "first" one on every process invocation.
_DOC_TYPE_WORDS = ("policy", "report", "contract", "memo", "guideline", "manual", "invoice")
_DEPARTMENT_WORDS = ("hr", "finance", "engineering", "legal", "sales", "marketing", "operations")
_CHITCHAT_PATTERNS = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye", "test"}


@lru_cache
def _get_nlp():
    return spacy.load("en_core_web_sm")


@dataclass
class QueryUnderstanding:
    entities: list[str] = field(default_factory=list)
    filter_candidates: dict[str, str] = field(default_factory=dict)
    freshness_signal: bool = False
    is_chitchat: bool = False


@dataclass
class RetrievalPlan:
    dense: bool
    bm25: bool
    graph: bool
    freshness: bool
    apply_filters: bool
    top_k: int = TOP_K_DEFAULT


def understand_query(query: str) -> QueryUnderstanding:
    normalized = query.strip().lower().rstrip(".!?")
    if normalized in _CHITCHAT_PATTERNS or len(normalized) < 3:
        return QueryUnderstanding(is_chitchat=True)

    doc = _get_nlp()(query)
    entities = sorted({ent.text.strip() for ent in doc.ents if ent.label_ in _ENTITY_LABELS and len(ent.text.strip()) > 1})
    date_texts = [ent.text for ent in doc.ents if ent.label_ in _DATE_LABELS]

    filter_candidates: dict[str, str] = {}
    for text in date_texts:
        parsed = dateparser.parse(text)
        if parsed:
            filter_candidates["date"] = parsed.date().isoformat()
            break
    for word in _DOC_TYPE_WORDS:
        if word in normalized:
            filter_candidates["doc_type"] = word
            break
    for word in _DEPARTMENT_WORDS:
        if word in normalized:
            filter_candidates["department"] = word
            break

    freshness_signal = bool(date_texts) or any(w in normalized for w in _FRESHNESS_WORDS)

    return QueryUnderstanding(entities=entities, filter_candidates=filter_candidates, freshness_signal=freshness_signal)


def plan_query(understanding: QueryUnderstanding, query: str) -> RetrievalPlan:
    """SS2.4: if all retrieval flags are false, the query is treated as
    chitchat/meta and retrieval is skipped entirely."""
    if understanding.is_chitchat:
        return RetrievalPlan(dense=False, bm25=False, graph=False, freshness=False, apply_filters=False, top_k=TOP_K_DEFAULT)

    normalized = query.lower()
    graph = len(understanding.entities) >= 2 or any(w in normalized for w in _RELATIONAL_WORDS)

    return RetrievalPlan(
        dense=True,
        bm25=True,
        graph=graph,
        freshness=understanding.freshness_signal,
        apply_filters=bool(understanding.filter_candidates),
        top_k=TOP_K_DEFAULT,
    )


def clamp_top_k(top_k: int, hard_cap: int = TOP_K_HARD_CAP) -> int:
    """NFR6: the plan's top_k is never trusted blindly - always clamped
    downstream regardless of what the planner (or a malicious/buggy caller)
    produced."""
    return max(1, min(top_k, hard_cap))


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int, filters: dict[str, str] | None) -> list[dict]: ...


@dataclass
class Retrievers:
    """Injected per-retriever implementations. All optional/None here in
    Phase 3 - Phase 4 (FR8) wires in real OpenSearch/Neo4j clients and adds
    parallel execution; this router's job is only correct selection +
    clamping, tested against fakes until then."""

    dense: Retriever | None = None
    bm25: Retriever | None = None
    graph: Retriever | None = None


def execute_plan(plan: RetrievalPlan, retrievers: Retrievers, query: str, filters: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """FR6: fires only the retrievers flagged true, applies filters only if
    apply_filters is true, and clamps top_k to the hard cap regardless of
    the planner's output."""
    top_k = clamp_top_k(plan.top_k)
    applied_filters = filters if plan.apply_filters else None
    results: dict[str, list[dict]] = {}
    if plan.dense and retrievers.dense:
        results["dense"] = retrievers.dense.retrieve(query, top_k, applied_filters)
    if plan.bm25 and retrievers.bm25:
        results["bm25"] = retrievers.bm25.retrieve(query, top_k, applied_filters)
    if plan.graph and retrievers.graph:
        results["graph"] = retrievers.graph.retrieve(query, top_k, applied_filters)
    return results


def log_plan_decision(request_id: str, query: str, plan: RetrievalPlan) -> None:
    """FR7: plan decisions logged with the request ID so they're queryable
    for retraining. Grounding/quality scores get attached later (Phase 7/8
    - they don't exist yet in this pipeline)."""
    logger.info("plan_decision", extra={"request_id": request_id, "query": query, "plan": asdict(plan)})
