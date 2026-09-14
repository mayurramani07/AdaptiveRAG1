import json

import pytest

from adaptive_rag.ingestion import (
    _DISAMBIGUATION_SYSTEM_PROMPT,
    _ENTITY_SYSTEM_PROMPT,
    _RELATIONSHIP_SYSTEM_PROMPT,
    Chunk,
    EntityMention,
    Relationship,
    extract_entities,
    extract_relationships,
    ingest_document,
    resolve_entities,
    sync_graph,
    validate_relationships,
)


class ScriptedLLM:
    """Fake LLM client keyed by system prompt (each extraction step uses its
    own fixed prompt, so this is enough to distinguish entity vs relationship
    vs disambiguation calls). Records every call for assertions."""

    def __init__(self, by_system: dict[str, str], default: str = "{}"):
        self.by_system = by_system
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.by_system.get(system, self.default)


class FakeNeo4j:
    """In-memory stand-in covering only the query shapes ingestion.py emits -
    enough to verify graph-level effects (dedup, pointers, cleanup) without a
    real Neo4j instance (none provisioned yet - see project context)."""

    def __init__(self):
        self.documents: set[str] = set()
        self.chunks: dict[str, dict] = {}  # chunk_id -> {doc_id, text}
        self.entities: dict[str, set[str]] = {}  # name -> set of chunk_ids
        self.entity_types: dict[str, str] = {}
        self.relationships: list[dict] = []

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
            self.entity_types[p["name"]] = p["type"]
        elif "RELATED_TO" in query:
            self.relationships.append(dict(p))
        elif "DETACH DELETE d, c" in query:
            keep = set(p["keep_ids"])
            self._drop_chunks([cid for cid, c in self.chunks.items() if c["doc_id"] not in keep])
            self.documents = {d for d in self.documents if d in keep}
        elif "DETACH DELETE c" in query:
            self._drop_chunks([cid for cid, c in self.chunks.items() if c["doc_id"] == p["doc_id"]])
        elif "MATCH (e:Entity) WHERE NOT" in query:
            self.entities = {name: chunks for name, chunks in self.entities.items() if chunks}


def entities_response(*surfaces_and_types):
    return json.dumps({"entities": [{"surface": s, "type": t} for s, t in surfaces_and_types]})


def relationships_response(*rels):
    return json.dumps(
        {
            "relationships": [
                {"source": s, "relationship": rel, "target": t, "confidence": c, "evidence": e}
                for s, rel, t, c, e in rels
            ]
        }
    )


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def test_extract_entities_grounds_against_source_text_and_rejects_hallucinations():
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Apple released a new phone.")
    llm = ScriptedLLM({_ENTITY_SYSTEM_PROMPT: entities_response(("Apple", "ORG"), ("Tim Cook", "PERSON"))})
    mentions = extract_entities(chunk, llm=llm)
    assert [m.surface for m in mentions] == ["Apple"]  # "Tim Cook" never appears in the text - rejected


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "{}",
        json.dumps({"entities": "not a list"}),
        json.dumps({"entities": [123, {"no_surface_field": True}]}),
    ],
)
def test_extract_entities_handles_invalid_llm_output(raw):
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Apple released a new phone.")
    llm = ScriptedLLM({_ENTITY_SYSTEM_PROMPT: raw})
    assert extract_entities(chunk, llm=llm) == []


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


def test_apple_inc_and_apple_resolve_to_one_canonical_entity():
    mentions = [
        EntityMention(surface="Apple Inc", canonical="apple", doc_id="d", chunk_id="d:0", entity_type="ORG"),
        EntityMention(surface="Apple", canonical="apple", doc_id="d", chunk_id="d:1", entity_type="ORG"),
    ]
    llm = ScriptedLLM({})
    registry = resolve_entities(mentions, llm=llm)
    assert len({e.key for e in registry.values()}) == 1
    assert llm.calls == []  # identical normalized key -> exact match, no LLM needed


def test_ambiguous_alias_candidate_merges_when_llm_confirms_same_entity():
    mentions = [
        EntityMention(surface="Google", canonical="google", doc_id="d", chunk_id="d:0", entity_type="ORG"),
        EntityMention(surface="Google Maps", canonical="google maps", doc_id="d", chunk_id="d:1", entity_type="ORG"),
    ]
    llm = ScriptedLLM({_DISAMBIGUATION_SYSTEM_PROMPT: json.dumps({"same_entity": True})})
    registry = resolve_entities(mentions, llm=llm)
    assert registry["google"].key == registry["google maps"].key
    assert len(llm.calls) == 1


def test_ambiguous_alias_candidate_stays_separate_when_llm_denies_same_entity():
    mentions = [
        EntityMention(surface="Google", canonical="google", doc_id="d", chunk_id="d:0", entity_type="ORG"),
        EntityMention(surface="Google Maps", canonical="google maps", doc_id="d", chunk_id="d:1", entity_type="ORG"),
    ]
    llm = ScriptedLLM({_DISAMBIGUATION_SYSTEM_PROMPT: json.dumps({"same_entity": False})})
    registry = resolve_entities(mentions, llm=llm)
    assert registry["google"].key != registry["google maps"].key


def test_different_entity_types_never_auto_alias_without_disambiguation():
    mentions = [
        EntityMention(surface="Amazon", canonical="amazon", doc_id="d", chunk_id="d:0", entity_type="ORG"),
        EntityMention(surface="Amazon River", canonical="amazon river", doc_id="d", chunk_id="d:1", entity_type="GPE"),
    ]
    llm = ScriptedLLM({})  # no disambiguation answer scripted - must not be needed
    registry = resolve_entities(mentions, llm=llm)
    assert registry["amazon"].key != registry["amazon river"].key
    assert llm.calls == []  # confidently different types short-circuit before any LLM call


# ---------------------------------------------------------------------------
# Relationship extraction
# ---------------------------------------------------------------------------


def test_steve_jobs_founded_apple():
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Steve Jobs founded Apple in a garage.")
    mentions = [
        EntityMention(surface="Steve Jobs", canonical="steve jobs", doc_id="d", chunk_id="d:0", entity_type="PERSON"),
        EntityMention(surface="Apple", canonical="apple", doc_id="d", chunk_id="d:0", entity_type="ORG"),
    ]
    llm = ScriptedLLM(
        {_RELATIONSHIP_SYSTEM_PROMPT: relationships_response(("Steve Jobs", "founded", "Apple", 0.95, "Steve Jobs founded Apple in a garage."))}
    )
    rels = extract_relationships(chunk, mentions, llm=llm)
    assert len(rels) == 1
    assert (rels[0].source, rels[0].label, rels[0].target) == ("steve jobs", "FOUNDED", "apple")
    assert rels[0].evidence == "Steve Jobs founded Apple in a garage."
    assert rels[0].confidence == 0.95


def test_apple_acquired_beats():
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Apple acquired Beats for its audio business.")
    mentions = [
        EntityMention(surface="Apple", canonical="apple", doc_id="d", chunk_id="d:0", entity_type="ORG"),
        EntityMention(surface="Beats", canonical="beats", doc_id="d", chunk_id="d:0", entity_type="ORG"),
    ]
    llm = ScriptedLLM(
        {_RELATIONSHIP_SYSTEM_PROMPT: relationships_response(("Apple", "acquired", "Beats", 0.9, "Apple acquired Beats for its audio business."))}
    )
    rels = extract_relationships(chunk, mentions, llm=llm)
    assert len(rels) == 1
    assert (rels[0].source, rels[0].label, rels[0].target) == ("apple", "ACQUIRED", "beats")


def test_passive_phrasing_direction_is_preserved_from_llm_output():
    # This validates that our code trusts/passes through whatever direction
    # the LLM states (per the prompt's explicit active-voice-normalization
    # instruction) rather than re-deriving or reversing it - it does not
    # test the live model's own reasoning, which needs a real eval set later
    # (same "baseline, revisit in Phase 9" caveat as the rest of this repo).
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Apple was founded by Steve Jobs.")
    mentions = [
        EntityMention(surface="Apple", canonical="apple", doc_id="d", chunk_id="d:0", entity_type="ORG"),
        EntityMention(surface="Steve Jobs", canonical="steve jobs", doc_id="d", chunk_id="d:0", entity_type="PERSON"),
    ]
    llm = ScriptedLLM(
        {_RELATIONSHIP_SYSTEM_PROMPT: relationships_response(("Steve Jobs", "FOUNDED", "Apple", 0.9, "Apple was founded by Steve Jobs."))}
    )
    rels = extract_relationships(chunk, mentions, llm=llm)
    assert (rels[0].source, rels[0].target) == ("steve jobs", "apple")


@pytest.mark.parametrize(
    "raw",
    ["not json", "{}", json.dumps({"relationships": "nope"}), json.dumps({"relationships": [{"source": "Apple"}]})],
)
def test_extract_relationships_handles_invalid_llm_output(raw):
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Steve Jobs founded Apple.")
    mentions = [
        EntityMention(surface="Steve Jobs", canonical="steve jobs", doc_id="d", chunk_id="d:0", entity_type="PERSON"),
        EntityMention(surface="Apple", canonical="apple", doc_id="d", chunk_id="d:0", entity_type="ORG"),
    ]
    llm = ScriptedLLM({_RELATIONSHIP_SYSTEM_PROMPT: raw})
    assert extract_relationships(chunk, mentions, llm=llm) == []


def test_relationship_rejects_entity_not_present_in_chunk():
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Apple released a new phone. Samsung reacted quickly.")
    mentions = [EntityMention(surface="Apple", canonical="apple", doc_id="d", chunk_id="d:0", entity_type="ORG")]
    # only one resolved mention, so extract_relationships short-circuits before calling the LLM
    assert extract_relationships(chunk, mentions, llm=ScriptedLLM({})) == []


def test_relationship_rejects_target_outside_resolved_entities():
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Apple competes with Samsung in phones.")
    mentions = [
        EntityMention(surface="Apple", canonical="apple", doc_id="d", chunk_id="d:0", entity_type="ORG"),
        EntityMention(surface="phones", canonical="phones", doc_id="d", chunk_id="d:0", entity_type="OTHER"),
    ]
    llm = ScriptedLLM(
        {_RELATIONSHIP_SYSTEM_PROMPT: relationships_response(("Apple", "COMPETES_WITH", "Samsung", 0.8, "Apple competes with Samsung in phones."))}
    )
    # "Samsung" was never a resolved entity in this chunk - must be dropped
    assert extract_relationships(chunk, mentions, llm=llm) == []


def test_relationship_rejects_ungrounded_evidence():
    chunk = Chunk(doc_id="d", chunk_id="d:0", text="Steve Jobs founded Apple.")
    mentions = [
        EntityMention(surface="Steve Jobs", canonical="steve jobs", doc_id="d", chunk_id="d:0", entity_type="PERSON"),
        EntityMention(surface="Apple", canonical="apple", doc_id="d", chunk_id="d:0", entity_type="ORG"),
    ]
    llm = ScriptedLLM(
        {_RELATIONSHIP_SYSTEM_PROMPT: relationships_response(("Steve Jobs", "FOUNDED", "Apple", 0.9, "this sentence is not in the source text"))}
    )
    assert extract_relationships(chunk, mentions, llm=llm) == []


def test_low_confidence_relationship_rejected():
    relationships = [
        Relationship(source="a", target="b", doc_id="d", chunk_id="c", confidence=0.9),
        Relationship(source="a", target="c", doc_id="d", chunk_id="c", confidence=0.2),
    ]
    validated = validate_relationships(relationships)
    assert [r.target for r in validated] == ["b"]


# ---------------------------------------------------------------------------
# End-to-end provenance
# ---------------------------------------------------------------------------


def test_provenance_preserved_end_to_end():
    driver = FakeNeo4j()
    text = "Steve Jobs founded Apple. Apple later acquired Beats."
    llm = ScriptedLLM(
        {
            _ENTITY_SYSTEM_PROMPT: entities_response(("Steve Jobs", "PERSON"), ("Apple", "ORG"), ("Beats", "ORG")),
            _RELATIONSHIP_SYSTEM_PROMPT: relationships_response(
                ("Steve Jobs", "founded", "Apple", 0.95, "Steve Jobs founded Apple."),
                ("Apple", "acquired", "Beats", 0.9, "Apple later acquired Beats."),
            ),
        }
    )
    ingest_document(driver, "doc1", text, llm=llm)

    assert set(driver.entities.keys()) == {"Steve Jobs", "Apple", "Beats"}
    assert driver.entity_types == {"Steve Jobs": "PERSON", "Apple": "ORG", "Beats": "ORG"}
    for name, chunk_ids in driver.entities.items():
        assert chunk_ids, f"{name} must carry at least one MENTIONED_IN pointer"
        for cid in chunk_ids:
            assert driver.chunks[cid]["doc_id"] == "doc1"
            assert driver.chunks[cid]["text"]  # resolves to real source passage

    rel_triples = {(r["source"], r["type"], r["target"]) for r in driver.relationships}
    assert ("Steve Jobs", "FOUNDED", "Apple") in rel_triples
    assert ("Apple", "ACQUIRED", "Beats") in rel_triples
    for r in driver.relationships:
        assert r["doc_id"] == "doc1"
        assert r["chunk_id"] in driver.chunks  # every relationship traces back to a real chunk
        assert r["evidence"] and r["confidence"] > 0


# ---------------------------------------------------------------------------
# Scheduled re-ingestion (unchanged behavior, now threading llm through)
# ---------------------------------------------------------------------------


def test_sync_graph_removes_deleted_document_and_orphaned_entities():
    driver = FakeNeo4j()
    llm = ScriptedLLM({_ENTITY_SYSTEM_PROMPT: entities_response(("Apple Inc.", "ORG"))})
    sync_graph(driver, {"doc1": "Apple Inc. is a company."}, llm=llm)
    assert "doc1" in driver.documents
    assert driver.entities

    sync_graph(driver, {}, llm=llm)
    assert driver.documents == set()
    assert driver.entities == {}


def test_sync_graph_drops_stale_chunks_and_entities_on_content_update():
    driver = FakeNeo4j()
    llm = ScriptedLLM(
        {
            _ENTITY_SYSTEM_PROMPT: entities_response(("Apple Inc", "ORG"), ("Google", "ORG"), ("Amazon", "ORG")),
            _RELATIONSHIP_SYSTEM_PROMPT: relationships_response(),
        }
    )
    old_text = "Apple Inc released the iPhone. Google announced Pixel. Amazon shipped Kindle. " * 60
    sync_graph(driver, {"doc1": old_text}, llm=llm)
    assert len(driver.chunks) > 1
    assert {"Apple Inc", "Google", "Amazon"} <= driver.entities.keys()

    new_text = "Apple Inc released the iPhone."
    sync_graph(driver, {"doc1": new_text}, llm=llm)

    # doc_id is unchanged, so the old chunks/entities must be gone, not just
    # left stale (this was the FR-ING2/FR-ING4 bug: old sync_graph only
    # cleaned up documents that vanished entirely, not ones that shrank).
    # "Google"/"Amazon" also correctly drop out here because they no longer
    # appear verbatim in the new chunk text (grounding check, not just the
    # chunk-wipe).
    assert set(driver.chunks.keys()) == {"doc1:0"}
    assert driver.chunks["doc1:0"]["text"] == new_text
    assert driver.entities.keys() == {"Apple Inc"}
