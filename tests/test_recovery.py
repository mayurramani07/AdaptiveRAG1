from adaptive_rag.grading import Grade
from adaptive_rag.recovery import (
    MAX_RECOVERY_ATTEMPTS,
    RecoveryStrategies,
    RedactedWebSearch,
    make_external_web_search_strategy,
    make_graph_expansion_strategy,
    make_internal_re_retrieval_strategy,
    make_query_rewrite_strategy,
    redact_pii,
    run_recovery,
)


class ScriptedGrader:
    """Grades by text, not a real model - lets tests script exact grades
    without depending on FastEmbed's real score distribution."""

    def __init__(self, score_by_text: dict[str, float], default: float = -100.0):
        self.score_by_text = score_by_text
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return [self.score_by_text.get(doc, self.default) for doc in documents]


def _hit(chunk_id, text, trust_tier=None):
    hit = {"id": chunk_id, "doc_id": "doc1", "chunk_id": chunk_id, "text": text}
    if trust_tier:
        hit["trust_tier"] = trust_tier
    return hit


def _graded(chunk_id, text, grade, score=None, trust_tier=None):
    hit = _hit(chunk_id, text, trust_tier)
    hit["grade"] = grade
    hit["grader_score"] = score if score is not None else (5.0 if grade == Grade.CORRECT else -10.0)
    return hit


def _strategy(new_query, new_evidence):
    """Builds a fixed RecoveryStrategyFn that ignores its inputs and always
    returns the given (query, evidence) - the standard fake for run_recovery
    orchestration tests."""

    def fn(query, prior_evidence):
        return new_query, new_evidence

    return fn


def _counting_strategy(evidence_text="still bad"):
    """A strategy that always produces fresh-but-still-bad evidence and
    counts how many times it was actually invoked - used by the chaos test
    to prove the cap stops real work, not just that the loop exits."""
    calls = []

    def fn(query, prior_evidence):
        calls.append(query)
        return query, [_hit("x", evidence_text)]

    fn.calls = calls
    return fn


# ---------------------------------------------------------------------------
# run_recovery core orchestration (FR13/FR14)
# ---------------------------------------------------------------------------


def test_run_recovery_skips_entirely_when_evidence_already_correct():
    graded = [_graded("a", "good", Grade.CORRECT)]
    result = run_recovery("query", graded, RecoveryStrategies())
    assert result.recovered is True
    assert result.attempts_used == 0
    assert result.strategies_tried == []


def test_run_recovery_tries_strategies_in_preference_order():
    grader = ScriptedGrader({"still bad": -10.0, "good now": 5.0})
    strategies = RecoveryStrategies(
        query_rewrite=_strategy("q2", [_hit("a", "still bad")]),
        internal_re_retrieval=_strategy("q2", [_hit("b", "good now")]),
    )
    graded = [_graded("orig", "bad", Grade.INCORRECT)]
    result = run_recovery("q1", graded, strategies, grader=grader)

    assert result.strategies_tried == ["query_rewrite", "internal_re_retrieval"]
    assert result.recovered is True
    assert result.attempts_used == 2


def test_run_recovery_stops_as_soon_as_evidence_grades_correct():
    grader = ScriptedGrader({"good immediately": 5.0})
    strategies = RecoveryStrategies(
        query_rewrite=_strategy("q2", [_hit("a", "good immediately")]),
        internal_re_retrieval=_strategy("q2", [_hit("b", "should never be tried")]),
    )
    graded = [_graded("orig", "bad", Grade.INCORRECT)]
    result = run_recovery("q1", graded, strategies, grader=grader)

    assert result.strategies_tried == ["query_rewrite"]
    assert result.attempts_used == 1


def test_run_recovery_skips_uninjected_strategies_without_consuming_attempts():
    grader = ScriptedGrader({"good": 5.0})
    # only internal_re_retrieval injected - query_rewrite (earlier in
    # preference order) must be skipped silently, not counted as an attempt
    strategies = RecoveryStrategies(internal_re_retrieval=_strategy("q1", [_hit("a", "good")]))
    graded = [_graded("orig", "bad", Grade.INCORRECT)]
    result = run_recovery("q1", graded, strategies, grader=grader)

    assert result.strategies_tried == ["internal_re_retrieval"]
    assert result.attempts_used == 1


def test_run_recovery_skips_graph_expansion_when_graph_was_not_used():
    grader = ScriptedGrader({"never reached": 5.0}, default=-10.0)
    strategies = RecoveryStrategies(graph_expansion=_strategy("q1", [_hit("a", "never reached")]))
    graded = [_graded("orig", "bad", Grade.INCORRECT)]
    result = run_recovery("q1", graded, strategies, grader=grader, graph_was_used=False)

    assert result.strategies_tried == []
    assert result.attempts_used == 0
    assert result.recovered is False


def test_run_recovery_tries_graph_expansion_when_graph_was_used():
    grader = ScriptedGrader({"expanded": 5.0})
    strategies = RecoveryStrategies(graph_expansion=_strategy("q1", [_hit("a", "expanded")]))
    graded = [_graded("orig", "bad", Grade.INCORRECT)]
    result = run_recovery("q1", graded, strategies, grader=grader, graph_was_used=True)

    assert result.strategies_tried == ["graph_expansion"]
    assert result.recovered is True


# ---------------------------------------------------------------------------
# Trust-tier gate (FR13 acceptance: web search never attempted while an
# authoritative internal source scores >= moderate threshold)
# ---------------------------------------------------------------------------


def test_web_search_never_attempted_while_authoritative_source_scores_moderately():
    web_search = _strategy("q1", [_hit("web", "should never be called")])
    strategies = RecoveryStrategies(external_web_search=web_search)
    # authoritative (default trust_tier), scored above the moderate threshold
    graded = [_graded("orig", "moderate", Grade.AMBIGUOUS, score=0.0)]
    result = run_recovery("q1", graded, strategies, grader=ScriptedGrader({}))

    assert "external_web_search" not in result.strategies_tried
    assert result.attempts_used == 0


def test_web_search_attempted_when_no_authoritative_source_scores_moderately():
    grader = ScriptedGrader({"web result": 5.0})
    strategies = RecoveryStrategies(external_web_search=_strategy("q1", [_hit("web", "web result", trust_tier="supplementary")]))
    # scored well below the moderate threshold
    graded = [_graded("orig", "very bad", Grade.INCORRECT, score=-20.0)]
    result = run_recovery("q1", graded, strategies, grader=grader)

    assert result.strategies_tried == ["external_web_search"]
    assert result.recovered is True


def test_web_search_attempted_when_only_supplementary_sources_scored_moderately():
    # A moderately-scoring SUPPLEMENTARY (web) source must not block a later
    # web search attempt - only an AUTHORITATIVE source blocks it (SS2.5).
    grader = ScriptedGrader({"web result": 5.0})
    strategies = RecoveryStrategies(external_web_search=_strategy("q1", [_hit("web2", "web result", trust_tier="supplementary")]))
    graded = [_graded("orig", "supplementary moderate", Grade.AMBIGUOUS, score=0.0, trust_tier="supplementary")]
    result = run_recovery("q1", graded, strategies, grader=grader)

    assert result.strategies_tried == ["external_web_search"]


# ---------------------------------------------------------------------------
# Hard cap (FR14) + chaos test
# ---------------------------------------------------------------------------


def test_run_recovery_never_exceeds_max_attempts():
    strategies = RecoveryStrategies(
        query_rewrite=_counting_strategy(),
        internal_re_retrieval=_counting_strategy(),
        graph_expansion=_counting_strategy(),
        external_web_search=_counting_strategy(),
    )
    grader = ScriptedGrader({}, default=-100.0)  # every attempt grades Incorrect, forever
    graded = [_graded("orig", "bad", Grade.INCORRECT, score=-100.0)]

    result = run_recovery("q1", graded, strategies, grader=grader, graph_was_used=True)

    assert result.attempts_used == MAX_RECOVERY_ATTEMPTS == 2
    assert result.recovered is False
    assert len(result.strategies_tried) == 2


def test_chaos_grader_always_incorrect_exactly_two_attempts_then_fallback_no_crash_no_loop():
    """FR14 chaos test: forces the grader to always return Incorrect and
    confirms exactly 2 recovery attempts occur, then a (non-recovered)
    fallback result is returned - no infinite loop, no crash, regardless of
    how many strategies are available to try."""
    counting = [_counting_strategy(f"attempt-{i}") for i in range(4)]
    strategies = RecoveryStrategies(
        query_rewrite=counting[0],
        internal_re_retrieval=counting[1],
        graph_expansion=counting[2],
        external_web_search=counting[3],
    )
    always_incorrect_grader = ScriptedGrader({}, default=-100.0)
    graded = [_graded("orig", "bad", Grade.INCORRECT, score=-100.0)]

    result = run_recovery("q1", graded, strategies, grader=always_incorrect_grader, graph_was_used=True)

    assert result.attempts_used == 2
    assert result.recovered is False
    # exactly the first two strategies in preference order were actually invoked
    assert len(counting[0].calls) == 1
    assert len(counting[1].calls) == 1
    assert len(counting[2].calls) == 0
    assert len(counting[3].calls) == 0


# ---------------------------------------------------------------------------
# PII redaction (FR15)
# ---------------------------------------------------------------------------


def test_redact_pii_scrubs_email():
    assert redact_pii("contact me at john.doe@example.com please") == "contact me at [REDACTED_EMAIL] please"


def test_redact_pii_scrubs_phone_number():
    assert "[REDACTED_PHONE]" in redact_pii("call me at 415-555-2671 tomorrow")


def test_redact_pii_scrubs_ssn():
    assert "[REDACTED_SSN]" in redact_pii("my ssn is 123-45-6789")


def test_redact_pii_scrubs_credit_card():
    assert "[REDACTED_CARD]" in redact_pii("card number 4111 1111 1111 1111 expires soon")


def test_redact_pii_leaves_clean_text_untouched():
    assert redact_pii("what is the refund policy") == "what is the refund policy"


def test_redacted_web_search_never_sends_raw_pii_to_backend():
    class SpyBackend:
        def __init__(self):
            self.received_queries = []

        def search(self, query):
            self.received_queries.append(query)
            return [{"id": "r1", "text": "a result"}]

    backend = SpyBackend()
    wrapper = RedactedWebSearch(backend)
    results = wrapper.search("email me at leak@example.com")

    assert backend.received_queries == ["email me at [REDACTED_EMAIL]"]
    assert "leak@example.com" not in backend.received_queries[0]
    assert results[0]["trust_tier"] == "supplementary"


def test_make_external_web_search_strategy_redacts_and_tags_results():
    class FakeBackend:
        def search(self, query):
            assert "@" not in query or "REDACTED" in query
            return [{"id": "r1", "text": "web fact"}]

    strategy = make_external_web_search_strategy(FakeBackend())
    new_query, evidence = strategy("contact ceo@company.com", [])

    assert evidence[0]["trust_tier"] == "supplementary"
    assert new_query == "contact ceo@company.com"  # web search doesn't rewrite the tracked query, only what it sends out


# ---------------------------------------------------------------------------
# Default strategy builders (real Phase 2-4 wiring, tested against fakes)
# ---------------------------------------------------------------------------


def test_make_query_rewrite_strategy_uses_rewritten_query_for_retrieval():
    from adaptive_rag.planning import Retrievers

    class FakeLLM:
        def complete_json(self, system, user):
            return '{"rewritten_query": "better query"}'

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k, filters):
            self.calls.append(query)
            return [{"id": "a", "doc_id": "d", "chunk_id": "a", "text": "result"}]

    dense = FakeRetriever()
    strategy = make_query_rewrite_strategy(Retrievers(dense=dense), llm=FakeLLM())
    new_query, evidence = strategy("original query", [])

    assert new_query == "better query"
    assert dense.calls == ["better query"]
    assert len(evidence) == 1


def test_make_query_rewrite_strategy_falls_back_to_original_query_on_invalid_llm_output():
    from adaptive_rag.planning import Retrievers

    class FakeLLM:
        def complete_json(self, system, user):
            return "not json"

    class FakeRetriever:
        def retrieve(self, query, top_k, filters):
            return []

    strategy = make_query_rewrite_strategy(Retrievers(dense=FakeRetriever()), llm=FakeLLM())
    new_query, _evidence = strategy("original query", [])
    assert new_query == "original query"


def test_make_internal_re_retrieval_strategy_widens_top_k():
    from adaptive_rag.planning import TOP_K_DEFAULT, Retrievers

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k, filters):
            self.calls.append(top_k)
            return []

    dense = FakeRetriever()
    # widen_factor=2 asks for TOP_K_DEFAULT*2 (40), but NFR6's clamp
    # (_run_and_fuse -> clamp_top_k) still caps it at TOP_K_HARD_CAP (30) -
    # the same defense-in-depth the normal request path gets, not bypassed
    # just because this is a recovery attempt.
    strategy = make_internal_re_retrieval_strategy(Retrievers(dense=dense), widen_factor=2)
    strategy("query", [])

    assert dense.calls[0] > TOP_K_DEFAULT  # actually widened...
    assert dense.calls[0] == 30  # ...but still clamped to TOP_K_HARD_CAP


def test_make_graph_expansion_strategy_uses_hops_2_by_default():
    from dataclasses import dataclass

    @dataclass
    class FakeEagerResult:
        records: list

    class FakeDriver:
        def __init__(self):
            self.queries = []

        def execute_query(self, query, **params):
            self.queries.append(query)
            return FakeEagerResult(records=[])

    driver = FakeDriver()
    strategy = make_graph_expansion_strategy(driver=driver, hops=2)
    strategy("Steve Jobs", [])

    assert driver.queries
    assert "RELATED_TO*1.." in driver.queries[0]
