from adaptive_rag.ingestion import (
    Chunk,
    Relationship,
    extract_entities,
    extract_relationships,
    ingest_document,
    sync_graph,
    validate_relationships,
)


class FakeNeo4j:
    """In-memory stand-in covering only the query shapes ingestion.py emits -
    enough to verify graph-level effects (dedup, pointers, cleanup) without a
    real Neo4j instance (none provisioned yet - see project context)."""

    def __init__(self):
        self.documents: set[str] = set()
        self.chunks: dict[str, dict] = {}  # chunk_id -> {doc_id, text}
        self.entities: dict[str, set[str]] = {}  # name -> set of chunk_ids
        self.relationships: set[tuple] = set()

    def _drop_chunks(self, chunk_ids):
        for cid in chunk_ids:
            del self.chunks[cid]
            for mentioned_chunks in self.entities.values():
                mentioned_chunks.discard(cid)

    def execute_query(self, query, **p):
        if query.startswith("MERGE (:Document"):
            self.documents.add(p["doc_id"])
        elif "MERGE (c:Chunk" in query:
            self.chunks[p["chunk_id"]] = {"doc_id": p["doc_id"], "text": p["text"]}
        elif "MERGE (e:Entity" in query and "MENTIONED_IN" in query:
            self.entities.setdefault(p["name"], set()).add(p["chunk_id"])
        elif "RELATED_TO" in query:
            self.relationships.add((p["source"], p["target"], p["doc_id"], p["chunk_id"]))
        elif "DETACH DELETE d, c" in query:
            keep = set(p["keep_ids"])
            self._drop_chunks([cid for cid, c in self.chunks.items() if c["doc_id"] not in keep])
            self.documents = {d for d in self.documents if d in keep}
        elif "DETACH DELETE c" in query:
            self._drop_chunks([cid for cid, c in self.chunks.items() if c["doc_id"] == p["doc_id"]])
        elif "MATCH (e:Entity) WHERE NOT" in query:
            self.entities = {name: chunks for name, chunks in self.entities.items() if chunks}


DUPLICATE_MENTION_TEXT = "Apple Inc. released a new phone. Apple continues to lead the market."


def test_duplicate_entity_resolves_to_one_node():
    driver = FakeNeo4j()
    ingest_document(driver, "doc1", DUPLICATE_MENTION_TEXT)
    # regex-heuristic extraction drops the trailing "." (word-boundary quirk);
    # what matters here is dedup, not exact surface spelling.
    assert list(driver.entities.keys()) == ["Apple Inc"]


def test_entity_resolves_back_to_source_text():
    driver = FakeNeo4j()
    ingest_document(driver, "doc1", DUPLICATE_MENTION_TEXT)
    mentioned_chunks = driver.entities["Apple Inc"]
    assert mentioned_chunks, "entity must carry at least one MENTIONED_IN pointer"
    for chunk_id in mentioned_chunks:
        assert driver.chunks[chunk_id]["text"]  # resolves to real passage text
        assert driver.chunks[chunk_id]["doc_id"] == "doc1"


def test_low_confidence_relationship_rejected():
    relationships = [
        Relationship(source="a", target="b", doc_id="d", chunk_id="c", confidence=0.9),
        Relationship(source="a", target="c", doc_id="d", chunk_id="c", confidence=0.2),
    ]
    validated = validate_relationships(relationships)
    assert [r.target for r in validated] == ["b"]


def test_same_sentence_relationship_scored_above_threshold():
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Apple met Google today. Amazon was not mentioned there.")
    mentions = extract_entities(chunk)
    validated = validate_relationships(extract_relationships(chunk, mentions))
    validated_pairs = {frozenset((r.source, r.target)) for r in validated}
    assert frozenset(("apple", "google")) in validated_pairs
    assert not any("amazon" in pair for pair in validated_pairs)


def test_sync_graph_removes_deleted_document_and_orphaned_entities():
    driver = FakeNeo4j()
    sync_graph(driver, {"doc1": "Apple Inc. is a company."})
    assert "doc1" in driver.documents
    assert driver.entities

    sync_graph(driver, {})
    assert driver.documents == set()
    assert driver.entities == {}


def test_sync_graph_drops_stale_chunks_and_entities_on_content_update():
    driver = FakeNeo4j()
    old_text = "Apple Inc released the iPhone. Google announced Pixel. Amazon shipped Kindle. " * 60
    sync_graph(driver, {"doc1": old_text})
    assert len(driver.chunks) > 1
    assert {"Apple Inc", "Google", "Amazon"} <= driver.entities.keys()

    new_text = "Apple Inc released the iPhone."
    sync_graph(driver, {"doc1": new_text})

    # doc_id is unchanged, so the old chunks/entities must be gone, not just
    # left stale (this was the FR-ING2/FR-ING4 bug: old sync_graph only
    # cleaned up documents that vanished entirely, not ones that shrank).
    assert set(driver.chunks.keys()) == {"doc1:0"}
    assert driver.chunks["doc1:0"]["text"] == new_text
    assert driver.entities.keys() == {"Apple Inc"}
