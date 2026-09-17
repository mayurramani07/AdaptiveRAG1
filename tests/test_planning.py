import itertools
import logging

import pytest

from adaptive_rag.planning import (
    TOP_K_HARD_CAP,
    CircuitBreaker,
    QueryUnderstanding,
    RetrievalPlan,
    Retrievers,
    clamp_top_k,
    execute_plan,
    log_plan_decision,
    plan_query,
    understand_query,
)


class FakeRetriever:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def retrieve(self, query, top_k, filters):
        self.calls.append((query, top_k, filters))
        return self.results


# ---------------------------------------------------------------------------
# Query understanding (FR4)
# ---------------------------------------------------------------------------


def test_understand_query_detects_chitchat():
    u = understand_query("hello")
    assert u.is_chitchat is True
    assert u.entities == [] and u.filter_candidates == {}


@pytest.mark.parametrize("greeting_typo", ["hii", "heyy", "helo", "okk", "byee", "thnks", "HIIII", "hiiiiiiii", "heyyyyy", "okkkkk"])
def test_understand_query_detects_common_greeting_typos_as_chitchat(greeting_typo):
    # Real gap found 2026-09-17: "hii" fell through exact matching, ran a
    # full retrieval+recovery cycle, and confusingly reported insufficient
    # evidence for what was obviously a greeting.
    assert understand_query(greeting_typo).is_chitchat is True


@pytest.mark.parametrize(
    "real_query",
    ["what is the refund policy", "who is the CEO of Acme Corporation", "how do I file an expense report"],
)
def test_understand_query_does_not_misclassify_real_short_questions_as_chitchat(real_query):
    assert understand_query(real_query).is_chitchat is False


def test_understand_query_does_not_treat_a_greeting_prefix_on_a_real_question_as_chitchat():
    # A mashed greeting followed by an actual question must still run
    # retrieval - only a message that IS (a typo/repeat of) a greeting,
    # nothing else, counts as chitchat.
    assert understand_query("hiii, what is the refund policy").is_chitchat is False


def test_understand_query_rejects_words_too_different_from_any_greeting():
    # "no" was never a supported chitchat pattern - collapsing repeats
    # doesn't invent new matches beyond the existing pattern set.
    assert understand_query("nooo").is_chitchat is False


def test_understand_query_extracts_entities():
    u = understand_query("What did Steve Jobs say about Apple?")
    assert "Steve Jobs" in u.entities
    assert "Apple" in u.entities


def test_understand_query_detects_date_and_freshness():
    u = understand_query("Show me the 2023 policy report")
    assert u.filter_candidates["date"].startswith("2023")  # normalized to an ISO date, not the raw "2023" text
    assert u.filter_candidates["doc_type"] == "policy"
    assert u.freshness_signal is True


def test_understand_query_detects_freshness_keyword_without_year():
    u = understand_query("What is the latest guideline?")
    assert u.freshness_signal is True


def test_understand_query_detects_relative_dates_a_regex_would_miss():
    # This is the concrete capability a plain year-regex never had -
    # spaCy's NER model recognizes "last month" as a DATE entity on its own.
    u = understand_query("What happened last month?")
    assert u.freshness_signal is True
    assert "date" in u.filter_candidates


def test_understand_query_plain_question_has_no_filters():
    u = understand_query("What is the refund policy timeline")
    # "policy" is a doc-type keyword but no date/freshness cue present
    assert u.freshness_signal is False


# ---------------------------------------------------------------------------
# Query planner (FR5)
# ---------------------------------------------------------------------------


def test_plan_query_chitchat_disables_all_retrieval():
    plan = plan_query(QueryUnderstanding(is_chitchat=True), "hi")
    assert (plan.dense, plan.bm25, plan.graph) == (False, False, False)


def test_plan_query_normal_query_enables_dense_and_bm25():
    understanding = understand_query("What is the refund policy?")
    plan = plan_query(understanding, "What is the refund policy?")
    assert plan.dense is True
    assert plan.bm25 is True


def test_plan_query_multiple_entities_enables_graph():
    query = "How are Steve Jobs and Apple connected to Beats?"
    understanding = understand_query(query)
    plan = plan_query(understanding, query)
    assert plan.graph is True


def test_plan_query_filters_and_freshness_propagate():
    query = "Show me the 2023 finance report"
    understanding = understand_query(query)
    plan = plan_query(understanding, query)
    assert plan.apply_filters is True
    assert plan.freshness is True


# ---------------------------------------------------------------------------
# top_k clamping (NFR6)
# ---------------------------------------------------------------------------


def test_clamp_top_k_caps_oversized_value():
    assert clamp_top_k(500) == TOP_K_HARD_CAP


def test_clamp_top_k_leaves_normal_value_alone():
    assert clamp_top_k(10) == 10


def test_clamp_top_k_floors_at_one():
    assert clamp_top_k(0) == 1


# ---------------------------------------------------------------------------
# Adaptive Router (FR6) - all 8 reachable flag combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dense,bm25,graph", list(itertools.product([True, False], repeat=3)))
def test_router_invokes_exactly_the_flagged_retrievers(dense, bm25, graph):
    dense_r, bm25_r, graph_r = FakeRetriever(), FakeRetriever(), FakeRetriever()
    retrievers = Retrievers(dense=dense_r, bm25=bm25_r, graph=graph_r)
    plan = RetrievalPlan(dense=dense, bm25=bm25, graph=graph, freshness=False, apply_filters=False, top_k=10)

    results = execute_plan(plan, retrievers, "some query")

    assert bool(dense_r.calls) == dense
    assert bool(bm25_r.calls) == bm25
    assert bool(graph_r.calls) == graph
    assert set(results.keys()) == {name for name, flag in (("dense", dense), ("bm25", bm25), ("graph", graph)) if flag}


def test_router_clamps_oversized_top_k_before_calling_retrievers():
    dense_r = FakeRetriever()
    plan = RetrievalPlan(dense=True, bm25=False, graph=False, freshness=False, apply_filters=False, top_k=500)
    execute_plan(plan, Retrievers(dense=dense_r), "q")
    assert dense_r.calls[0][1] == TOP_K_HARD_CAP  # (query, top_k, filters)


def test_router_only_applies_filters_when_plan_says_so():
    dense_r = FakeRetriever()
    filters = {"doc_type": "policy"}

    plan_no_filters = RetrievalPlan(dense=True, bm25=False, graph=False, freshness=False, apply_filters=False, top_k=10)
    execute_plan(plan_no_filters, Retrievers(dense=dense_r), "q", filters=filters)
    assert dense_r.calls[0][2] is None

    plan_with_filters = RetrievalPlan(dense=True, bm25=False, graph=False, freshness=False, apply_filters=True, top_k=10)
    execute_plan(plan_with_filters, Retrievers(dense=dense_r), "q", filters=filters)
    assert dense_r.calls[1][2] == filters


def test_router_skips_a_flagged_retriever_that_was_never_injected():
    # Phase 4 hasn't wired real retrievers in yet - a flagged-but-missing
    # retriever must not crash the router.
    plan = RetrievalPlan(dense=True, bm25=True, graph=True, freshness=False, apply_filters=False, top_k=10)
    results = execute_plan(plan, Retrievers(), "q")
    assert results == {}


class RaisingRetriever:
    """Simulates a dependency that is down (e.g. Neo4j unreachable)."""

    def retrieve(self, query, top_k, filters):
        raise ConnectionError("neo4j down")


# ---------------------------------------------------------------------------
# Phase 9 chaos test: "kill Neo4j mid-request" (SS13's documented fallback -
# graph failure degrades to dense+bm25-only, the whole request never fails)
# ---------------------------------------------------------------------------


def test_execute_plan_degrades_when_graph_retriever_raises():
    dense_r = FakeRetriever(results=[{"id": "d1"}])
    plan = RetrievalPlan(dense=True, bm25=False, graph=True, freshness=False, apply_filters=False, top_k=10)
    retrievers = Retrievers(dense=dense_r, graph=RaisingRetriever())

    results = execute_plan(plan, retrievers, "q")  # must not raise

    assert results == {"dense": [{"id": "d1"}]}  # graph silently dropped, dense result still returned


def test_execute_plan_opens_circuit_breaker_after_repeated_graph_failures():
    plan = RetrievalPlan(dense=False, bm25=False, graph=True, freshness=False, apply_filters=False, top_k=10)
    retrievers = Retrievers(graph=RaisingRetriever())
    breakers: dict[str, CircuitBreaker] = {}

    for _ in range(3):
        execute_plan(plan, retrievers, "q", breakers=breakers)
    assert breakers["graph"].is_open()

    # once open, a 4th request skips calling the dead retriever entirely
    calls_before = getattr(retrievers.graph, "calls", None)  # RaisingRetriever has no .calls, just confirm no crash
    results = execute_plan(plan, retrievers, "q", breakers=breakers)
    assert results == {}
    assert calls_before is None


# ---------------------------------------------------------------------------
# Plan decision logging (FR7)
# ---------------------------------------------------------------------------


def test_log_plan_decision_includes_request_id_and_plan(caplog):
    plan = RetrievalPlan(dense=True, bm25=True, graph=False, freshness=False, apply_filters=False, top_k=20)
    with caplog.at_level(logging.INFO, logger="adaptive_rag.planning"):
        log_plan_decision("req-123", "some query", plan)
    record = caplog.records[0]
    assert record.request_id == "req-123"
    assert record.plan["dense"] is True
    assert record.query == "some query"
