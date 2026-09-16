"""Retrieval & Fusion (Phase 4, SS2.3/SS2.4): Dense (OpenSearch k-NN), BM25
(OpenSearch), Graph (Neo4j) retrievers, RRF fusion, and reranking.

All three retrievers conform to `planning.Retriever` (query, top_k, filters)
-> list[dict] and plug into Phase 3's `execute_plan` router unchanged - no
router/Protocol changes were needed to add them.

`filters` (doc_type/department/date, from Phase 3's `QueryUnderstanding`) IS
applied per-backend: BM25 and Graph filter server-side (a standard OpenSearch
bool `filter` clause; a Cypher WHERE clause against Document properties).
Dense filters client-side by over-fetching candidates from the ANN search
then discarding non-matches in Python (see `DenseRetriever`'s docstring for
why - OpenSearch k-NN filter support is engine-dependent and unverified
against a real cluster). None of this has any effect yet on documents
ingested without metadata: nothing in this codebase currently decides or
extracts document-level doc_type/department/date (SS9 - document taxonomy
is still an open PRD decision) - `ingest_and_index`/`sync_all` below accept
an optional `metadata` dict for callers who already know it, but nothing
populates one automatically.

`index_chunk` (OpenSearch write path) is wired into ingestion via
`ingest_and_index`/`sync_all` - thin wrappers living here, not in
ingestion.py, to avoid a circular import (this module already imports
`Neo4jLike`/`get_neo4j_driver` from `ingestion.py`).

Dense retrieval's embedding provider is a swappable abstraction
(`EmbeddingLike`) - same dependency-injection pattern as
`RedisLike`/`Neo4jLike`/`LLMLike` elsewhere in this codebase - so the model
or service can change later without touching `DenseRetriever` or
`index_chunk`. FastEmbed (ONNX Runtime, no torch) was chosen over
sentence-transformers after measuring: torch alone added ~500MB RSS just
importing sentence-transformers, over Render's free-tier 512MB limit;
FastEmbed measured ~180MB for the same embedding model. The reranker
(`RerankerLike`) follows the same reasoning - FastEmbed's ONNX cross-encoder,
not sentence-transformers', to stay off torch entirely.

Graph retrieval reuses Phase 3's NER (`planning.understand_query`) to find
candidate entity names in the query, rather than reimplementing entity
extraction inside a Cypher query - then traverses
Entity -MENTIONED_IN-> Chunk -PART_OF-> Document. Per SS3's "the graph is
an index, not evidence" rule, this returns real chunk text pulled from
Neo4j, never a synthesized relationship-edge label.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from adaptive_rag.config import get_settings
from adaptive_rag.ingestion import LLMLike, Neo4jLike, get_neo4j_driver
from adaptive_rag.planning import understand_query

DEFAULT_INDEX = "documents"
EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5 output size
DEFAULT_RRF_K = 60
RERANK_TOP_N_CAP = 30  # FR10: reranker never receives more than this many candidates
DENSE_FILTER_OVERSAMPLE = 5  # DenseRetriever over-fetches this factor before post-filtering by metadata
FILTERABLE_FIELDS = ("doc_type", "department", "date")

INDEX_MAPPING = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "text": {"type": "text"},
            "doc_type": {"type": "keyword"},
            "department": {"type": "keyword"},
            "date": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIM,
                # faiss, not nmslib: nmslib is deprecated since OpenSearch
                # 2.16 and blocked for new indices entirely from 3.0+
                # (verified against OpenSearch's own docs 2026-09-16) -
                # Aiven's free OpenSearch tier runs 3.3.2, so nmslib would
                # have failed index creation there. faiss is OpenSearch's
                # own default since 2.18 and supports the same space_type.
                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss"},
            },
        }
    },
}


class EmbeddingLike(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedProvider:
    """Local ONNX embedding model - no torch, no API key, no network call
    at inference time (one-time model download from HuggingFace on first
    use, same as spaCy's model download in Phase 3)."""

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        settings = get_settings()
        self._model = TextEmbedding(model_name or settings.embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]


@lru_cache
def get_embedding_provider() -> EmbeddingLike:
    return FastEmbedProvider()


class OpenSearchLike(Protocol):
    def index(self, *, index: str, body: dict[str, Any], id: str) -> Any: ...
    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]: ...
    def delete_by_query(self, *, index: str, body: dict[str, Any]) -> Any: ...


@lru_cache
def get_opensearch_client() -> OpenSearchLike:
    from opensearchpy import OpenSearch

    settings = get_settings()
    auth = (settings.opensearch_user, settings.opensearch_password) if settings.opensearch_user else None
    # pool_maxsize: urllib3's default of 1 gets exceeded as soon as Dense
    # and BM25 fire concurrently (planning.execute_plan's ThreadPoolExecutor
    # fan-out) through this one shared client - observed live 2026-09-16 as
    # a "Connection pool is full, discarding connection" warning during the
    # first real end-to-end /v1/query request. Sized to the retrieval
    # pool's own concurrency (planning._get_retrieval_pool's max_workers=6),
    # not a guess.
    return OpenSearch(hosts=[settings.opensearch_url], http_auth=auth, use_ssl=True, verify_certs=True, pool_maxsize=6)


def index_chunk(
    doc_id: str,
    chunk_id: str,
    text: str,
    client: OpenSearchLike | None = None,
    embedding_provider: EmbeddingLike | None = None,
    index: str = DEFAULT_INDEX,
    metadata: dict[str, str] | None = None,
) -> None:
    """Ingestion-time embedding generation: embeds `text` once and upserts
    it into OpenSearch keyed by chunk_id, so Dense/BM25Retriever can find it
    later. `metadata` (doc_type/department/date) is stored on the chunk doc
    if given, enabling filtered retrieval later - see module docstring for
    why nothing currently supplies it automatically."""
    search_client = client or get_opensearch_client()
    provider = embedding_provider or get_embedding_provider()
    [vector] = provider.embed([text])
    body = {"doc_id": doc_id, "chunk_id": chunk_id, "text": text, "embedding": vector}
    if metadata:
        body.update({k: v for k, v in metadata.items() if k in FILTERABLE_FIELDS})
    search_client.index(index=index, id=chunk_id, body=body)


def _map_opensearch_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "id": hit["_id"],
            "doc_id": hit["_source"]["doc_id"],
            "chunk_id": hit["_source"]["chunk_id"],
            "text": hit["_source"]["text"],
            "score": hit["_score"],
        }
        for hit in hits
    ]


def _matches_filters(source: dict[str, Any], filters: dict[str, str]) -> bool:
    return all(source.get(key) == value for key, value in filters.items())


class DenseRetriever:
    """Conforms to `planning.Retriever` (query, top_k, filters) -> list[dict].
    Embeds the query, then asks OpenSearch for the nearest chunks by cosine
    similarity. `filters` is applied client-side: over-fetches
    `top_k * DENSE_FILTER_OVERSAMPLE` candidates from the ANN search, then
    discards non-matches in Python before truncating to top_k - deliberately
    NOT passed as a native k-NN "filter" clause, since that OpenSearch
    feature's support varies by k-NN engine (lucene/faiss vs. nmslib, the
    engine this index uses) and hasn't been verified against a real cluster
    (none provisioned yet). Post-filtering is slower and can under-return if
    very few of the oversampled candidates match, but it's guaranteed
    correct regardless of engine - a documented, reasonable tradeoff for a
    first cut. Revisit once a real OpenSearch cluster + engine choice let
    native filtering be tested."""

    def __init__(self, client: OpenSearchLike | None = None, embedding_provider: EmbeddingLike | None = None, index: str = DEFAULT_INDEX):
        self._client = client
        self._embedding_provider = embedding_provider
        self._index = index

    def retrieve(self, query: str, top_k: int, filters: dict[str, str] | None) -> list[dict]:
        client = self._client or get_opensearch_client()
        provider = self._embedding_provider or get_embedding_provider()
        [vector] = provider.embed([query])
        fetch_k = top_k * DENSE_FILTER_OVERSAMPLE if filters else top_k
        body = {"size": fetch_k, "query": {"knn": {"embedding": {"vector": vector, "k": fetch_k}}}}
        response = client.search(index=self._index, body=body)
        hits = response["hits"]["hits"]
        if filters:
            hits = [h for h in hits if _matches_filters(h["_source"], filters)]
        return _map_opensearch_hits(hits[:top_k])


class BM25Retriever:
    """Conforms to `planning.Retriever`. Plain OpenSearch lexical match
    query on `text` - complements DenseRetriever's semantic k-NN search
    (SS2.3 row 7). `filters` applied server-side as a standard bool `filter`
    clause (well-supported OpenSearch DSL, no engine-specific concerns
    unlike Dense's k-NN filtering)."""

    def __init__(self, client: OpenSearchLike | None = None, index: str = DEFAULT_INDEX):
        self._client = client
        self._index = index

    def retrieve(self, query: str, top_k: int, filters: dict[str, str] | None) -> list[dict]:
        client = self._client or get_opensearch_client()
        es_query: dict[str, Any] = {"match": {"text": query}}
        if filters:
            es_query = {"bool": {"must": [{"match": {"text": query}}], "filter": [{"term": {k: v}} for k, v in filters.items()]}}
        body = {"size": top_k, "query": es_query}
        response = client.search(index=self._index, body=body)
        return _map_opensearch_hits(response["hits"]["hits"])


class GraphRetriever:
    """Conforms to `planning.Retriever`. Finds candidate entity names in the
    query by reusing Phase 3's NER (`planning.understand_query`) rather than
    reimplementing entity extraction inside Cypher, then traverses
    Entity -MENTIONED_IN-> Chunk -PART_OF-> Document to find relevant
    passages. Per SS3, this returns real chunk text - never a synthesized
    relationship-edge label - so it's safe to use as generation evidence
    downstream, same as Dense/BM25 hits. `filters` applied as a Cypher WHERE
    clause against Document node properties, restricted to
    `FILTERABLE_FIELDS` (a fixed allowlist, not user-controlled field names
    - values are always passed as query parameters, never interpolated, so
    this can't become a Cypher injection point). No ranking signal exists
    yet beyond "was this entity mentioned" - every hit scores 1.0 (ponytail:
    flat scoring, fine as an input to RRF which only cares about rank order
    per retriever, not raw score comparability across retrievers; revisit
    if a real relevance signal - traversal distance, mention frequency - is
    needed later)."""

    def __init__(self, driver: Neo4jLike | None = None, hops: int = 1):
        # hops=1 (default): only entities the query directly names - exactly
        # Phase 4's original query, unchanged for existing callers. hops>1:
        # also follow RELATED_TO edges out from those entities before
        # collecting MENTIONED_IN chunks - this is what Phase 5's Recovery
        # Planner uses for the "Graph Expansion" strategy (SS2.5: "expand
        # traversal depth, e.g. 1-hop to 2-hop, before concluding the graph
        # has nothing").
        self._driver = driver
        self._hops = hops

    def retrieve(self, query: str, top_k: int, filters: dict[str, str] | None) -> list[dict]:
        entities = understand_query(query).entities
        if not entities:
            return []
        driver = self._driver or get_neo4j_driver()
        params: dict[str, Any] = {"entities": entities, "top_k": top_k}
        where_extra = ""
        if filters:
            active = [field for field in FILTERABLE_FIELDS if field in filters]
            if active:
                where_extra = " AND " + " AND ".join(f"d.{field} = ${field}" for field in active)
                params.update({field: filters[field] for field in active})

        entity_match = "MATCH (e:Entity) WHERE any(name IN $entities WHERE toLower(e.name) CONTAINS toLower(name) OR toLower(name) CONTAINS toLower(e.name)) "
        if self._hops > 1:
            entity_match += (
                f"OPTIONAL MATCH (e)-[:RELATED_TO*1..{self._hops - 1}]-(expanded:Entity) "
                "WITH collect(DISTINCT e) + collect(DISTINCT expanded) AS matched "
                "UNWIND matched AS ent WITH DISTINCT ent WHERE ent IS NOT NULL "
            )
            traversal_source = "ent"
        else:
            traversal_source = "e"

        result = driver.execute_query(
            entity_match
            + f"MATCH ({traversal_source})-[:MENTIONED_IN]->(c:Chunk)-[:PART_OF]->(d:Document) "
            "WHERE true" + where_extra + " "
            "RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, d.doc_id AS doc_id "
            "LIMIT $top_k",
            **params,
        )
        return [
            {"id": r["chunk_id"], "doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "text": r["text"], "score": 1.0}
            for r in result.records
        ]


def reciprocal_rank_fusion(results_by_retriever: dict[str, list[dict]], k: int = DEFAULT_RRF_K, top_k: int | None = None) -> list[dict]:
    """FR9: combines ranked lists from multiple retrievers via Reciprocal
    Rank Fusion - rank position within each retriever's list matters, not
    the raw scores, which aren't comparable across dense/bm25/graph (cosine
    similarity vs. BM25 score vs. a flat 1.0). Deterministic given identical
    inputs (FR9 acceptance): addition is commutative regardless of
    dict/retrieval order, and ties are broken explicitly by chunk_id, not
    by relying on dict/insertion order - the same non-determinism class
    already found and fixed once in Phase 3's keyword matching."""
    scores: dict[str, float] = {}
    doc_by_id: dict[str, dict] = {}
    for hits in results_by_retriever.values():
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            doc_by_id.setdefault(chunk_id, hit)

    fused = sorted(doc_by_id.values(), key=lambda hit: (-scores[hit["chunk_id"]], hit["chunk_id"]))
    if top_k is not None:
        fused = fused[:top_k]
    return [{**hit, "rrf_score": scores[hit["chunk_id"]]} for hit in fused]


class RerankerLike(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """Local ONNX cross-encoder - no torch, same reasoning as
    FastEmbedProvider (module docstring)."""

    def __init__(self, model_name: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        settings = get_settings()
        self._model = TextCrossEncoder(model_name or settings.reranker_model)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return list(self._model.rerank(query, documents))


@lru_cache
def get_reranker() -> RerankerLike:
    return FastEmbedReranker()


def rerank_candidates(query: str, candidates: list[dict], reranker: RerankerLike | None = None, top_n_cap: int = RERANK_TOP_N_CAP) -> list[dict]:
    """FR10: the reranker never receives more than `top_n_cap` candidates,
    regardless of how many were fused - enforced here in code (the slice
    below), not left as a convention for the caller to remember. Expects
    RRF-fused input (rrf_score present) but works on any list[dict] with a
    "text" field."""
    capped = candidates[:top_n_cap]
    if not capped:
        return []
    model = reranker or get_reranker()
    scores = model.rerank(query, [c["text"] for c in capped])
    reranked = [{**candidate, "rerank_score": score} for candidate, score in zip(capped, scores)]
    return sorted(reranked, key=lambda c: -c["rerank_score"])


def _set_document_metadata(driver: Neo4jLike, doc_id: str, metadata: dict[str, str]) -> None:
    active = {k: v for k, v in metadata.items() if k in FILTERABLE_FIELDS}
    if active:
        driver.execute_query("MATCH (d:Document {doc_id: $doc_id}) SET d += $metadata", doc_id=doc_id, metadata=active)


def ingest_and_index(
    driver: Neo4jLike,
    doc_id: str,
    text: str,
    llm: LLMLike | None = None,
    opensearch_client: OpenSearchLike | None = None,
    embedding_provider: EmbeddingLike | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    """Closes the gap where `index_chunk` (OpenSearch) was implemented but
    never wired into ingestion - previously chunks reached Neo4j but not
    OpenSearch, so Dense/BM25 had nothing to search. Lives here, not in
    ingestion.py, to avoid a circular import (this module already imports
    from ingestion.py). Calls `chunk_document` twice (once inside
    `ingest_document`, once here) - a small, deliberate redundancy
    (`chunk_document` is a pure function, cheap, no side effects) rather
    than restructuring Phase 2's `ingest_document` internals for one
    caller. `metadata` (doc_type/department/date), if given, is stored on
    both the Neo4j Document node and every OpenSearch chunk doc, enabling
    filtered retrieval for this document - see module docstring for why
    nothing supplies this automatically yet."""
    from adaptive_rag.ingestion import chunk_document, ingest_document

    ingest_document(driver, doc_id, text, llm=llm)
    if metadata:
        _set_document_metadata(driver, doc_id, metadata)
    for chunk in chunk_document(doc_id, text):
        index_chunk(doc_id, chunk.chunk_id, chunk.text, client=opensearch_client, embedding_provider=embedding_provider, metadata=metadata)


def sync_all(
    driver: Neo4jLike,
    documents: dict[str, str],
    llm: LLMLike | None = None,
    opensearch_client: OpenSearchLike | None = None,
    embedding_provider: EmbeddingLike | None = None,
    metadata_by_doc: dict[str, dict[str, str]] | None = None,
    index: str = DEFAULT_INDEX,
) -> None:
    """Scheduled re-ingestion across BOTH stores (extends FR-ING4 to
    OpenSearch) - mirrors `ingestion.sync_graph`'s semantics (wipe + rebuild
    per document, so an edit never leaves stale data behind) but for
    OpenSearch too, which `sync_graph` alone never touched. Deletes, in one
    `delete_by_query` call, every OpenSearch chunk whose doc_id isn't in
    `documents` (same `WHERE NOT doc_id IN keep_ids` shape as
    `sync_graph`'s Neo4j cleanup), then wipes and rebuilds each given
    document's chunks - same reasoning as Phase 2's fix for the "edited
    document leaves stale chunks behind" bug, now applied to OpenSearch."""
    from adaptive_rag.ingestion import chunk_document, sync_graph

    client = opensearch_client or get_opensearch_client()
    metadata_by_doc = metadata_by_doc or {}
    keep_ids = list(documents.keys())

    client.delete_by_query(index=index, body={"query": {"bool": {"must_not": [{"terms": {"doc_id": keep_ids}}]}}})

    sync_graph(driver, documents, llm=llm)

    for doc_id, text in documents.items():
        client.delete_by_query(index=index, body={"query": {"term": {"doc_id": doc_id}}})
        metadata = metadata_by_doc.get(doc_id)
        if metadata:
            _set_document_metadata(driver, doc_id, metadata)
        for chunk in chunk_document(doc_id, text):
            index_chunk(doc_id, chunk.chunk_id, chunk.text, client=client, embedding_provider=embedding_provider, index=index, metadata=metadata)
