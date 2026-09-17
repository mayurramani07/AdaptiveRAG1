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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Protocol

import dateparser
import spacy

TOP_K_DEFAULT = 20
TOP_K_HARD_CAP = 30
DEFAULT_RETRIEVAL_TIMEOUT_SECONDS = 3.0

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


def _one_edit_away(a: str, b: str) -> bool:
    """True if `a` can become `b` via at most one character insert/delete/
    substitute - catches common typos ("hii"/"hi", "heyy"/"hey") without a
    fuzzy-matching dependency. Real-world gap found 2026-09-17: "hii" fell
    through exact matching, ran a full (pointless) retrieval+CRAG-recovery
    cycle, and correctly-but-confusingly reported insufficient evidence for
    what was obviously just a greeting."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    seen_diff = False
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
            continue
        if seen_diff:
            return False
        seen_diff = True
        j += 1
    return True


def _collapse_repeated_chars(s: str) -> str:
    """Collapses runs of 2+ identical consecutive characters to one -
    normalizes key-mashed emphasis typing ("hiiii", "heyyyy", "okkkkk", any
    number of repeats) to its canonical form. Applied to both the query and
    the patterns before comparing, so "hello" and a mashed "hellllooo" both
    collapse to the same "helo" and compare equal. Real-world gap found
    2026-09-17: "HIIII" (4 repeated letters) was still beyond the original
    single-typo tolerance."""
    if not s:
        return s
    result = [s[0]]
    for ch in s[1:]:
        if ch != result[-1]:
            result.append(ch)
    return "".join(result)


_COLLAPSED_CHITCHAT_PATTERNS = {_collapse_repeated_chars(p) for p in _CHITCHAT_PATTERNS}
_MAX_COLLAPSED_CHITCHAT_PATTERN_LEN = max(len(p) for p in _COLLAPSED_CHITCHAT_PATTERNS)


def _is_chitchat_like(normalized: str) -> bool:
    if normalized in _CHITCHAT_PATTERNS:
        return True
    collapsed = _collapse_repeated_chars(normalized)
    if collapsed in _COLLAPSED_CHITCHAT_PATTERNS:
        return True
    if len(collapsed) > _MAX_COLLAPSED_CHITCHAT_PATTERN_LEN + 1:
        return False  # too long to be a one-typo-away greeting even after collapsing repeats
    return any(_one_edit_away(collapsed, pattern) for pattern in _COLLAPSED_CHITCHAT_PATTERNS)


@lru_cache
def _get_nlp():
    return spacy.load("en_core_web_sm")


@lru_cache
def _get_retrieval_pool() -> ThreadPoolExecutor:
    # ponytail: one shared pool for the process, not one per call - thread
    # creation has real OS-level cost (measured ~150-200ms per pool on
    # Windows), so a fresh pool per request was needlessly slow. Sized for
    # a few requests' worth of concurrent 3-way fan-out; revisit under real
    # load (Phase 9 measurement).
    return ThreadPoolExecutor(max_workers=6)


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
    if _is_chitchat_like(normalized) or len(normalized) < 3:
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


@dataclass
class CircuitBreaker:
    """NFR2: opens after `failure_threshold` consecutive failures, stays
    open for `cooldown_seconds`, then allows one more attempt. One instance
    per dependency (dense/bm25/graph) - pass the same `breakers` dict across
    calls to actually accumulate state; a fresh dict per call (the default)
    means no memory between requests, which is fine for tests but the
    caller (Phase 6 orchestration) should hold one dict for the process."""

    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    _failures: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            self._opened_at = None
            self._failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


def execute_plan(
    plan: RetrievalPlan,
    retrievers: Retrievers,
    query: str,
    filters: dict[str, str] | None = None,
    breakers: dict[str, CircuitBreaker] | None = None,
    timeout: float = DEFAULT_RETRIEVAL_TIMEOUT_SECONDS,
) -> dict[str, list[dict]]:
    """FR6/FR8: fires only the retrievers flagged true, in parallel, applies
    filters only if apply_filters is true, and clamps top_k to the hard cap
    regardless of the planner's output.

    NFR1/NFR2: each retriever call gets its own timeout and circuit breaker.
    A slow/failing retriever degrades to "no results from that retriever"
    (SS13's documented fallback, e.g. "Neo4j down -> degrade to
    OpenSearch-only") - it never fails the whole request.

    ponytail: a genuinely hung synchronous retriever call can't be killed
    in Python, only abandoned - the shared pool below just stops waiting on
    it, but the thread itself leaks until it eventually returns or errors
    on its own, permanently occupying one worker slot. This is the outer
    safety net; retriever implementations should still set their own
    client-level timeouts (Phase 4's OpenSearch/Neo4j calls) so hangs are
    rare and bounded at the source. A circuit breaker naturally throttles
    how often a stuck dependency gets sent more work, which limits (but
    doesn't eliminate) pool exhaustion from repeated hangs - revisit if
    that turns out to matter under real load (Phase 9 measurement).
    """
    top_k = clamp_top_k(plan.top_k)
    applied_filters = filters if plan.apply_filters else None
    breakers = breakers if breakers is not None else {}

    tasks = [
        (name, retriever)
        for name, retriever, flag in (
            ("dense", retrievers.dense, plan.dense),
            ("bm25", retrievers.bm25, plan.bm25),
            ("graph", retrievers.graph, plan.graph),
        )
        if flag and retriever is not None
    ]
    if not tasks:
        return {}

    pool = _get_retrieval_pool()
    results: dict[str, list[dict]] = {}
    futures = {}
    for name, retriever in tasks:
        breaker = breakers.setdefault(name, CircuitBreaker())
        if breaker.is_open():
            logger.warning("circuit_open", extra={"retriever": name})
            continue
        futures[name] = (pool.submit(retriever.retrieve, query, top_k, applied_filters), breaker)

    for name, (future, breaker) in futures.items():
        try:
            results[name] = future.result(timeout=timeout)
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            logger.warning("retriever_failed", extra={"retriever": name}, exc_info=True)
    return results


def log_plan_decision(request_id: str, query: str, plan: RetrievalPlan) -> None:
    """FR7: plan decisions logged with the request ID so they're queryable
    for retraining. Grounding/quality scores get attached later (Phase 7/8
    - they don't exist yet in this pipeline)."""
    logger.info("plan_decision", extra={"request_id": request_id, "query": query, "plan": asdict(plan)})
