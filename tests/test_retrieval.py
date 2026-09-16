from dataclasses import dataclass

from adaptive_rag.retrieval import (
    DEFAULT_INDEX,
    DENSE_FILTER_OVERSAMPLE,
    EMBEDDING_DIM,
    RERANK_TOP_N_CAP,
    BM25Retriever,
    DenseRetriever,
    FastEmbedProvider,
    FastEmbedReranker,
    GraphRetriever,
    index_chunk,
    ingest_and_index,
    reciprocal_rank_fusion,
    rerank_candidates,
    sync_all,
)


class FakeEmbeddingProvider:
    def __init__(self, dim=EMBEDDING_DIM):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(i) for i in range(self.dim)] for _ in texts]


class FakeOpenSearch:
    """In-memory stand-in matching opensearch-py's real keyword-only
    `index(*, index, body, id)` / `search(*, index, body)` signatures
    (verified against the installed client) - no live OpenSearch needed."""

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.search_calls: list[tuple[str, dict]] = []

    def index(self, *, index, body, id):
        self.docs[id] = {"_index": index, **body}

    def search(self, *, index, body):
        self.search_calls.append((index, body))
        query = body["query"]
        if "knn" in query:
            size = query["knn"]["embedding"]["k"]
            candidates = list(self.docs.items())
        elif "bool" in query:
            size = body["size"]
            filter_clauses = query["bool"].get("filter", [])
            candidates = [
                (doc_id, doc)
                for doc_id, doc in self.docs.items()
                if all(doc.get(next(iter(f["term"]))) == next(iter(f["term"].values())) for f in filter_clauses)
            ]
        else:  # plain "match" (BM25, no filters)
            size = body["size"]
            candidates = list(self.docs.items())
        hits = [
            {"_id": doc_id, "_score": 1.0 - i * 0.01, "_source": {key: val for key, val in doc.items() if key != "_index"}}
            for i, (doc_id, doc) in enumerate(candidates)
        ][:size]
        return {"hits": {"hits": hits}}

    def delete_by_query(self, *, index, body):
        query = body["query"]
        if "bool" in query:  # must_not terms doc_id - "delete anything not in this set"
            keep_ids = set(query["bool"]["must_not"][0]["terms"]["doc_id"])
            self.docs = {doc_id: doc for doc_id, doc in self.docs.items() if doc["doc_id"] in keep_ids}
        else:  # term doc_id - "delete this one document's chunks"
            target = query["term"]["doc_id"]
            self.docs = {doc_id: doc for doc_id, doc in self.docs.items() if doc["doc_id"] != target}


class FakeRecord(dict):
    """Mimics neo4j.Record's mapping-style access (record["field"]) - real
    Record is a collections.abc.Mapping subclass, verified against the
    installed neo4j package; a plain dict satisfies the same access pattern
    used in GraphRetriever.retrieve."""


@dataclass
class FakeEagerResult:
    records: list[FakeRecord]


class FakeNeo4jReader:
    """In-memory stand-in for reads (GraphRetriever), separate from
    ingestion.py's FakeNeo4j (which only covers the write-shaped queries
    ingestion emits) - matches execute_query's real return shape
    (`.records`, each record accessible via `record["field"]`)."""

    def __init__(self, entity_chunks: dict[str, list[dict]], doc_metadata: dict[str, dict] | None = None):
        # entity name (lowercase) -> list of {chunk_id, text, doc_id} it's mentioned in
        self._entity_chunks = entity_chunks
        self._doc_metadata = doc_metadata or {}
        self.queries: list[tuple[str, dict]] = []

    def execute_query(self, query, **params):
        self.queries.append((query, params))
        matched_chunks: dict[str, dict] = {}
        for entity_name, chunks in self._entity_chunks.items():
            if any(entity_name in name.lower() or name.lower() in entity_name for name in params["entities"]):
                for chunk in chunks:
                    matched_chunks[chunk["chunk_id"]] = chunk

        from adaptive_rag.retrieval import FILTERABLE_FIELDS

        for field in FILTERABLE_FIELDS:
            if field in params:
                matched_chunks = {
                    cid: c for cid, c in matched_chunks.items() if self._doc_metadata.get(c["doc_id"], {}).get(field) == params[field]
                }

        records = [FakeRecord(chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"]) for c in matched_chunks.values()]
        return FakeEagerResult(records=records[: params["top_k"]])


class EmptyLLM:
    """Satisfies ingestion.LLMLike but always reports zero entities - keeps
    `ingest_and_index`/`sync_all` tests focused on the OpenSearch-wiring
    behavior they're actually testing, not re-covering entity/relationship
    extraction (already fully covered in tests/test_ingestion.py)."""

    def complete_json(self, system, user):
        return '{"entities": []}'


class FakeNeo4jWriter:
    """Minimal write-shaped fake for `ingest_and_index`/`sync_all` tests -
    covers the query shapes exercised when the LLM (EmptyLLM) returns no
    entities, plus the document-metadata SET. Deliberately smaller than
    ingestion.py's own FakeNeo4j (test_ingestion.py), which already covers
    full entity/relationship writes - no need to duplicate that here."""

    def __init__(self):
        self.documents: set[str] = set()
        self.doc_metadata: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}

    def execute_query(self, query, **p):
        if query.startswith("MERGE (:Document"):
            self.documents.add(p["doc_id"])
        elif "MERGE (c:Chunk" in query:
            self.chunks[p["chunk_id"]] = {"doc_id": p["doc_id"], "text": p["text"]}
        elif "SET d += $metadata" in query:
            self.doc_metadata.setdefault(p["doc_id"], {}).update(p["metadata"])
        elif "DETACH DELETE d, c" in query:
            keep = set(p["keep_ids"])
            self.chunks = {cid: c for cid, c in self.chunks.items() if c["doc_id"] in keep}
            self.documents = {d for d in self.documents if d in keep}
        elif "DETACH DELETE c" in query:
            self.chunks = {cid: c for cid, c in self.chunks.items() if c["doc_id"] != p["doc_id"]}
        elif "MATCH (e:Entity) WHERE NOT" in query:
            pass  # no entities ever created in these tests (EmptyLLM)


class ScriptedReranker:
    def __init__(self, score_by_text: dict[str, float]):
        self.score_by_text = score_by_text
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return [self.score_by_text.get(doc, 0.0) for doc in documents]


def test_index_chunk_embeds_and_stores_in_opensearch():
    client = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    index_chunk("doc1", "doc1:0", "Some chunk text", client=client, embedding_provider=embedder)

    assert embedder.calls == [["Some chunk text"]]
    stored = client.docs["doc1:0"]
    assert stored["doc_id"] == "doc1"
    assert stored["chunk_id"] == "doc1:0"
    assert stored["text"] == "Some chunk text"
    assert len(stored["embedding"]) == EMBEDDING_DIM


def test_index_chunk_uses_default_index_name():
    client = FakeOpenSearch()
    index_chunk("doc1", "doc1:0", "text", client=client, embedding_provider=FakeEmbeddingProvider())
    assert client.docs["doc1:0"]["_index"] == DEFAULT_INDEX


def test_dense_retriever_embeds_query_and_returns_hits():
    client = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    index_chunk("doc1", "doc1:0", "Apple released a new phone", client=client, embedding_provider=embedder)
    index_chunk("doc1", "doc1:1", "Apple also makes laptops", client=client, embedding_provider=embedder)

    retriever = DenseRetriever(client=client, embedding_provider=embedder)
    results = retriever.retrieve("Tell me about Apple", top_k=5, filters=None)

    assert embedder.calls[-1] == ["Tell me about Apple"]  # the query itself got embedded
    assert len(results) == 2
    assert {r["chunk_id"] for r in results} == {"doc1:0", "doc1:1"}
    assert all("score" in r and "text" in r for r in results)


def test_dense_retriever_respects_top_k():
    client = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    for i in range(5):
        index_chunk("doc1", f"doc1:{i}", f"chunk {i}", client=client, embedding_provider=embedder)

    retriever = DenseRetriever(client=client, embedding_provider=embedder)
    results = retriever.retrieve("query", top_k=2, filters=None)
    assert len(results) == 2


def test_dense_retriever_conforms_to_planning_retriever_protocol():
    # Proves the "keep Phase 3's router interface unchanged" claim is real,
    # not just a docstring assertion - DenseRetriever plugs into the
    # existing execute_plan router with zero changes to planning.py.
    from adaptive_rag.planning import RetrievalPlan, Retrievers, execute_plan

    client = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    index_chunk("doc1", "doc1:0", "some text", client=client, embedding_provider=embedder)

    retriever = DenseRetriever(client=client, embedding_provider=embedder)
    plan = RetrievalPlan(dense=True, bm25=False, graph=False, freshness=False, apply_filters=False, top_k=5)
    results = execute_plan(plan, Retrievers(dense=retriever), "query")

    assert "dense" in results
    assert len(results["dense"]) == 1


def test_fastembed_provider_produces_correct_dimension_vectors():
    # Real model, not a fake - FastEmbed is free/local (no API key, no
    # network call after the one-time model download), same approach
    # already used for spaCy in Phase 3.
    provider = FastEmbedProvider()
    vectors = provider.embed(["a short sentence", "another one"])
    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIM
    assert all(isinstance(v, float) for v in vectors[0])


# ---------------------------------------------------------------------------
# BM25 (FR8)
# ---------------------------------------------------------------------------


def test_bm25_retriever_returns_hits_without_embedding_the_query():
    client = FakeOpenSearch()
    index_chunk("doc1", "doc1:0", "Apple released a new phone", client=client, embedding_provider=FakeEmbeddingProvider())

    retriever = BM25Retriever(client=client)
    results = retriever.retrieve("Apple phone", top_k=5, filters=None)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc1:0"
    query_body = client.search_calls[-1][1]
    assert query_body["query"] == {"match": {"text": "Apple phone"}}  # lexical, not knn


def test_bm25_retriever_respects_top_k():
    client = FakeOpenSearch()
    for i in range(5):
        index_chunk("doc1", f"doc1:{i}", f"chunk {i}", client=client, embedding_provider=FakeEmbeddingProvider())
    retriever = BM25Retriever(client=client)
    assert len(retriever.retrieve("query", top_k=2, filters=None)) == 2


def test_bm25_retriever_conforms_to_planning_retriever_protocol():
    from adaptive_rag.planning import RetrievalPlan, Retrievers, execute_plan

    client = FakeOpenSearch()
    index_chunk("doc1", "doc1:0", "some text", client=client, embedding_provider=FakeEmbeddingProvider())
    plan = RetrievalPlan(dense=False, bm25=True, graph=False, freshness=False, apply_filters=False, top_k=5)
    results = execute_plan(plan, Retrievers(bm25=BM25Retriever(client=client)), "query")
    assert "bm25" in results


# ---------------------------------------------------------------------------
# Graph (FR8)
# ---------------------------------------------------------------------------


def test_graph_retriever_finds_chunks_via_entity_mention():
    driver = FakeNeo4jReader(
        {
            "apple": [{"chunk_id": "doc1:0", "text": "Apple released a new phone.", "doc_id": "doc1"}],
            "beats": [{"chunk_id": "doc1:1", "text": "Apple acquired Beats.", "doc_id": "doc1"}],
        }
    )
    retriever = GraphRetriever(driver=driver)
    results = retriever.retrieve("Tell me about Apple", top_k=10, filters=None)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc1:0"
    assert results[0]["doc_id"] == "doc1"
    assert results[0]["text"]  # resolves to real source text, not a synthesized label


def test_graph_retriever_returns_empty_when_no_entities_in_query():
    driver = FakeNeo4jReader({"apple": [{"chunk_id": "doc1:0", "text": "x", "doc_id": "doc1"}]})
    retriever = GraphRetriever(driver=driver)
    results = retriever.retrieve("hello there", top_k=10, filters=None)
    assert results == []
    assert driver.queries == []  # short-circuits before ever touching Neo4j


def test_graph_retriever_respects_top_k():
    driver = FakeNeo4jReader(
        {"apple": [{"chunk_id": f"doc1:{i}", "text": f"Apple chunk {i}", "doc_id": "doc1"} for i in range(5)]}
    )
    retriever = GraphRetriever(driver=driver)
    results = retriever.retrieve("Apple", top_k=2, filters=None)
    assert len(results) == 2


def test_graph_retriever_conforms_to_planning_retriever_protocol():
    from adaptive_rag.planning import RetrievalPlan, Retrievers, execute_plan

    driver = FakeNeo4jReader({"apple": [{"chunk_id": "doc1:0", "text": "Apple text", "doc_id": "doc1"}]})
    plan = RetrievalPlan(dense=False, bm25=False, graph=True, freshness=False, apply_filters=False, top_k=5)
    results = execute_plan(plan, Retrievers(graph=GraphRetriever(driver=driver)), "Apple")
    assert "graph" in results


# ---------------------------------------------------------------------------
# RRF fusion (FR9)
# ---------------------------------------------------------------------------


def _hit(chunk_id, text="text"):
    return {"id": chunk_id, "doc_id": "doc1", "chunk_id": chunk_id, "text": text}


def test_rrf_fusion_combines_and_boosts_docs_ranked_in_multiple_retrievers():
    results_by_retriever = {
        "dense": [_hit("a"), _hit("b"), _hit("c")],
        "bm25": [_hit("b"), _hit("a"), _hit("d")],
    }
    fused = reciprocal_rank_fusion(results_by_retriever)
    fused_ids = [h["chunk_id"] for h in fused]

    # "a" and "b" each appear in both lists (rank 1+2 combined) - both must
    # outrank "c"/"d", which only appear in one list each.
    assert fused_ids.index("a") < fused_ids.index("c")
    assert fused_ids.index("b") < fused_ids.index("d")
    assert {h["chunk_id"] for h in fused} == {"a", "b", "c", "d"}


def test_rrf_fusion_deterministic_across_repeated_calls():
    results_by_retriever = {
        "dense": [_hit("a"), _hit("b")],
        "bm25": [_hit("b"), _hit("a")],
        "graph": [_hit("c")],
    }
    first = [h["chunk_id"] for h in reciprocal_rank_fusion(results_by_retriever)]
    for _ in range(20):
        assert [h["chunk_id"] for h in reciprocal_rank_fusion(results_by_retriever)] == first


def test_rrf_fusion_respects_top_k():
    results_by_retriever = {"dense": [_hit("a"), _hit("b"), _hit("c")]}
    fused = reciprocal_rank_fusion(results_by_retriever, top_k=2)
    assert len(fused) == 2


def test_rrf_fusion_empty_input_returns_empty():
    assert reciprocal_rank_fusion({}) == []


# ---------------------------------------------------------------------------
# Reranker (FR10)
# ---------------------------------------------------------------------------


def test_rerank_candidates_never_exceeds_cap():
    reranker = ScriptedReranker({})
    candidates = [_hit(str(i), text=f"doc {i}") for i in range(RERANK_TOP_N_CAP + 10)]
    rerank_candidates("query", candidates, reranker=reranker)
    assert len(reranker.calls[0][1]) == RERANK_TOP_N_CAP  # never handed more than the cap, even given more


def test_rerank_candidates_reorders_by_rerank_score():
    candidates = [_hit("a", text="irrelevant"), _hit("b", text="highly relevant")]
    reranker = ScriptedReranker({"irrelevant": 0.1, "highly relevant": 0.9})
    reranked = rerank_candidates("query", candidates, reranker=reranker)
    assert [h["chunk_id"] for h in reranked] == ["b", "a"]


def test_rerank_candidates_empty_input_returns_empty():
    assert rerank_candidates("query", [], reranker=ScriptedReranker({})) == []


def test_fastembed_reranker_produces_scores_for_each_document():
    # Real model, not a fake - same free/local reasoning as the embedding
    # provider test above.
    reranker = FastEmbedReranker()
    scores = reranker.rerank("What is the capital of France?", ["Paris is the capital of France.", "Bananas are yellow."])
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    assert scores[0] > scores[1]  # the relevant document should score higher


# ---------------------------------------------------------------------------
# Filters (closes the "filters accepted but not applied" gap)
# ---------------------------------------------------------------------------


def test_dense_retriever_applies_filters_client_side():
    client = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    index_chunk("doc1", "doc1:0", "policy text", client=client, embedding_provider=embedder, metadata={"doc_type": "policy"})
    index_chunk("doc1", "doc1:1", "report text", client=client, embedding_provider=embedder, metadata={"doc_type": "report"})

    retriever = DenseRetriever(client=client, embedding_provider=embedder)
    results = retriever.retrieve("query", top_k=5, filters={"doc_type": "policy"})

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc1:0"


def test_dense_retriever_oversamples_before_filtering():
    client = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    for i in range(3):
        index_chunk("doc1", f"doc1:{i}", f"chunk {i}", client=client, embedding_provider=embedder)

    retriever = DenseRetriever(client=client, embedding_provider=embedder)
    retriever.retrieve("query", top_k=2, filters={"doc_type": "policy"})

    requested_k = client.search_calls[-1][1]["query"]["knn"]["embedding"]["k"]
    assert requested_k == 2 * DENSE_FILTER_OVERSAMPLE  # over-fetched, not just top_k


def test_dense_retriever_no_oversample_without_filters():
    client = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    index_chunk("doc1", "doc1:0", "text", client=client, embedding_provider=embedder)
    DenseRetriever(client=client, embedding_provider=embedder).retrieve("query", top_k=5, filters=None)
    assert client.search_calls[-1][1]["query"]["knn"]["embedding"]["k"] == 5


def test_bm25_retriever_applies_filters_server_side():
    client = FakeOpenSearch()
    index_chunk("doc1", "doc1:0", "policy text", client=client, embedding_provider=FakeEmbeddingProvider(), metadata={"department": "finance"})
    index_chunk("doc1", "doc1:1", "policy text", client=client, embedding_provider=FakeEmbeddingProvider(), metadata={"department": "hr"})

    retriever = BM25Retriever(client=client)
    results = retriever.retrieve("policy", top_k=5, filters={"department": "finance"})

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc1:0"
    query_body = client.search_calls[-1][1]["query"]
    assert query_body["bool"]["filter"] == [{"term": {"department": "finance"}}]


def test_graph_retriever_applies_filters_via_cypher_where():
    driver = FakeNeo4jReader(
        {
            "apple": [
                {"chunk_id": "doc1:0", "text": "Apple finance chunk", "doc_id": "doc1"},
                {"chunk_id": "doc2:0", "text": "Apple hr chunk", "doc_id": "doc2"},
            ]
        },
        doc_metadata={"doc1": {"department": "finance"}, "doc2": {"department": "hr"}},
    )
    retriever = GraphRetriever(driver=driver)
    results = retriever.retrieve("Apple", top_k=10, filters={"department": "finance"})

    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"
    _, params = driver.queries[-1]
    assert params["department"] == "finance"  # value passed as a query param, never string-interpolated


def test_graph_retriever_no_filter_clause_without_filters():
    driver = FakeNeo4jReader({"apple": [{"chunk_id": "doc1:0", "text": "Apple chunk", "doc_id": "doc1"}]})
    GraphRetriever(driver=driver).retrieve("Apple", top_k=10, filters=None)
    query_str, params = driver.queries[-1]
    assert "department" not in params
    assert "WHERE true " in query_str


# ---------------------------------------------------------------------------
# Ingestion wiring (closes the "index_chunk never wired in" gap)
# ---------------------------------------------------------------------------


def test_ingest_and_index_writes_to_both_stores():
    driver = FakeNeo4jWriter()
    opensearch = FakeOpenSearch()
    ingest_and_index(driver, "doc1", "hello world", llm=EmptyLLM(), opensearch_client=opensearch, embedding_provider=FakeEmbeddingProvider())

    assert "doc1" in driver.documents
    assert driver.chunks  # reached Neo4j
    assert opensearch.docs  # reached OpenSearch too - this was the actual gap
    assert set(driver.chunks.keys()) == set(opensearch.docs.keys())


def test_ingest_and_index_stores_metadata_in_both_stores():
    driver = FakeNeo4jWriter()
    opensearch = FakeOpenSearch()
    metadata = {"doc_type": "policy", "department": "finance"}
    ingest_and_index(driver, "doc1", "hello", llm=EmptyLLM(), opensearch_client=opensearch, embedding_provider=FakeEmbeddingProvider(), metadata=metadata)

    assert driver.doc_metadata["doc1"] == metadata
    stored_chunk = next(iter(opensearch.docs.values()))
    assert stored_chunk["doc_type"] == "policy"
    assert stored_chunk["department"] == "finance"


def test_sync_all_removes_deleted_document_from_both_stores():
    driver = FakeNeo4jWriter()
    opensearch = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    sync_all(driver, {"doc1": "hello world"}, llm=EmptyLLM(), opensearch_client=opensearch, embedding_provider=embedder)
    assert "doc1" in driver.documents
    assert opensearch.docs

    sync_all(driver, {}, llm=EmptyLLM(), opensearch_client=opensearch, embedding_provider=embedder)
    assert driver.documents == set()
    assert opensearch.docs == {}


def test_sync_all_rebuilds_opensearch_chunks_on_content_update():
    # Mirrors ingestion.py's own regression test for the same bug class,
    # now verifying OpenSearch (which the original Neo4j-only fix never
    # touched) also doesn't leave stale chunks behind after an edit.
    driver = FakeNeo4jWriter()
    opensearch = FakeOpenSearch()
    embedder = FakeEmbeddingProvider()
    old_text = "word " * 500  # spans multiple 200-word chunks
    sync_all(driver, {"doc1": old_text}, llm=EmptyLLM(), opensearch_client=opensearch, embedding_provider=embedder)
    assert len(opensearch.docs) > 1

    new_text = "short text"
    sync_all(driver, {"doc1": new_text}, llm=EmptyLLM(), opensearch_client=opensearch, embedding_provider=embedder)

    assert set(opensearch.docs.keys()) == {"doc1:0"}
    assert opensearch.docs["doc1:0"]["text"] == new_text
