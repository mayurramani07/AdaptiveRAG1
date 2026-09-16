import logging

from adaptive_rag.pipeline import PipelineResult, _route_name, run_pipeline
from adaptive_rag.planning import RetrievalPlan, Retrievers
from adaptive_rag.recovery import RecoveryStrategies


class FakeRetriever:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def retrieve(self, query, top_k, filters):
        self.calls.append((query, top_k, filters))
        return self.results


class ScriptedGrader:
    def __init__(self, score_by_text: dict[str, float], default: float = -100.0):
        self.score_by_text = score_by_text
        self.default = default
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return [self.score_by_text.get(doc, self.default) for doc in documents]


def _hit(chunk_id, text):
    return {"id": chunk_id, "doc_id": "doc1", "chunk_id": chunk_id, "text": text}


# ---------------------------------------------------------------------------
# _route_name
# ---------------------------------------------------------------------------


def _plan(dense=False, bm25=False, graph=False):
    return RetrievalPlan(dense=dense, bm25=bm25, graph=graph, freshness=False, apply_filters=False, top_k=20)


def test_route_name_chitchat_when_all_flags_false():
    assert _route_name(_plan()) == "chitchat"


def test_route_name_simple_rag_dense_only():
    assert _route_name(_plan(dense=True)) == "simple-rag"


def test_route_name_hybrid_rag_dense_and_bm25():
    assert _route_name(_plan(dense=True, bm25=True)) == "hybrid-rag"


def test_route_name_graph_rag_whenever_graph_is_flagged():
    assert _route_name(_plan(dense=True, bm25=True, graph=True)) == "graph-rag"
    assert _route_name(_plan(graph=True)) == "graph-rag"


def test_route_name_bm25_rag_bm25_only():
    assert _route_name(_plan(bm25=True)) == "bm25-rag"


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def test_run_pipeline_chitchat_skips_retrieval_entirely():
    dense = FakeRetriever()
    result = run_pipeline("hi", "req-1", retrievers=Retrievers(dense=dense))

    assert result.route == "chitchat"
    assert dense.calls == []  # FR3-style guarantee: no downstream retriever call on chitchat
    assert result.messages  # still a valid prompt (empty context + the query)


def test_run_pipeline_correct_evidence_skips_recovery():
    dense = FakeRetriever(results=[_hit("a", "Apple released a new phone")])
    grader = ScriptedGrader({"Apple released a new phone": 10.0})  # well above CORRECT threshold
    result = run_pipeline("Tell me about Apple", "req-1", retrievers=Retrievers(dense=dense), grader=grader)

    assert isinstance(result, PipelineResult)
    assert result.recovery_used is False
    assert result.insufficient_evidence is False
    assert len(result.sources) == 1
    assert result.sources[0]["id"] == "a"


def test_run_pipeline_recovers_via_injected_strategy():
    dense = FakeRetriever(results=[_hit("a", "weak evidence")])
    grader = ScriptedGrader({"weak evidence": -100.0, "strong evidence": 10.0})

    def rewrite_strategy(query, prior_evidence):
        return query, [_hit("b", "strong evidence")]

    strategies = RecoveryStrategies(query_rewrite=rewrite_strategy)
    result = run_pipeline("query", "req-1", retrievers=Retrievers(dense=dense), recovery_strategies=strategies, grader=grader)

    assert result.recovery_used is True
    assert result.insufficient_evidence is False
    assert result.sources[0]["id"] == "b"


def test_run_pipeline_returns_insufficient_evidence_when_recovery_exhausted():
    dense = FakeRetriever(results=[_hit("a", "bad evidence")])
    grader = ScriptedGrader({}, default=-100.0)  # everything always grades Incorrect

    result = run_pipeline("query", "req-1", retrievers=Retrievers(dense=dense), recovery_strategies=RecoveryStrategies(), grader=grader)

    assert result.insufficient_evidence is True
    assert result.messages == []  # never builds a generation prompt from known-bad evidence


def test_run_pipeline_logs_plan_decision(caplog):
    dense = FakeRetriever(results=[_hit("a", "text")])
    grader = ScriptedGrader({"text": 10.0})
    with caplog.at_level(logging.INFO, logger="adaptive_rag.planning"):
        run_pipeline("some query", "req-42", retrievers=Retrievers(dense=dense), grader=grader)
    record = next(r for r in caplog.records if r.message == "plan_decision")
    assert record.request_id == "req-42"


def test_run_pipeline_accumulates_circuit_breaker_state_across_calls():
    from adaptive_rag import pipeline as pipeline_module

    class FailingRetriever:
        def retrieve(self, query, top_k, filters):
            raise ConnectionError("dependency down")

    pipeline_module._BREAKERS.clear()
    grader = ScriptedGrader({}, default=-100.0)
    for _ in range(3):
        run_pipeline("query", "req-1", retrievers=Retrievers(dense=FailingRetriever()), recovery_strategies=RecoveryStrategies(), grader=grader)

    assert pipeline_module._BREAKERS["dense"].is_open() is True
    pipeline_module._BREAKERS.clear()
