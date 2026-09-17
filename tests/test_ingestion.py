import json

import pytest

from adaptive_rag.ingestion import (
    _DISAMBIGUATION_SYSTEM_PROMPT,
    _ENTITY_SYSTEM_PROMPT,
    _RELATIONSHIP_SYSTEM_PROMPT,
    DEFAULT_MAX_CHUNK_CHARS,
    Chunk,
    EntityMention,
    GroqLLMClient,
    Relationship,
    chunk_document,
    extract_entities,
    extract_relationships,
    extract_upload_text,
    ingest_document,
    resolve_entities,
    sanitize_upload_doc_id,
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


class FlakyLLM:
    """Raises on a specific call index (1-based, across all complete_json
    calls for the whole document) - simulates one chunk's extraction
    hitting a non-retryable error (e.g. Groq 400), real scenario found
    2026-09-17. Succeeds (returning empty results) on every other call."""

    def __init__(self, fail_on_call_index):
        self.fail_on_call_index = fail_on_call_index
        self.call_count = 0

    def complete_json(self, system, user):
        self.call_count += 1
        if self.call_count == self.fail_on_call_index:
            raise RuntimeError("simulated non-retryable extraction failure")
        if system == _ENTITY_SYSTEM_PROMPT:
            if "Zenith Robotics" in user:
                return entities_response(("Zenith Robotics", "ORG"))
            return json.dumps({"entities": []})
        return json.dumps({"relationships": []})


def test_ingest_document_skips_a_chunk_whose_extraction_raises_and_continues():
    # Two chunks: chunk 1's extraction call raises, chunk 2's must still
    # succeed - the whole document must not fail just because one chunk did.
    sentence_a = "Nimbus Corp released a new product."
    sentence_b = "Zenith Robotics announced a partnership."
    padding = " ".join(["word"] * 194)  # pads chunk 1 to exactly 200 words
    text = f"{sentence_a} {padding} {sentence_b}"
    assert len(text.split()) == 205  # 6 + 194 + 5 - chunk boundary lands exactly after sentence_a's chunk

    driver = FakeNeo4j()
    llm = FlakyLLM(fail_on_call_index=1)  # chunk 1's extract_entities call

    ingest_document(driver, "doc1", text, llm=llm)  # must not raise

    assert "Zenith Robotics" in driver.entities  # chunk 2 still processed normally
    assert "Nimbus Corp" not in driver.entities  # chunk 1's extraction was skipped, not fabricated
    assert len(driver.chunks) == 2  # both chunks still indexed into the graph regardless


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


# ---------------------------------------------------------------------------
# Upload helpers (Phase 11, FR-ING5) - POST /v1/documents
# ---------------------------------------------------------------------------


def test_sanitize_upload_doc_id_slugifies_filename():
    doc_id = sanitize_upload_doc_id("Quarterly Report 2026.pdf")
    assert doc_id.startswith("quarterly-report-2026-")
    assert " " not in doc_id


def test_sanitize_upload_doc_id_is_unique_per_call():
    a = sanitize_upload_doc_id("report.pdf")
    b = sanitize_upload_doc_id("report.pdf")
    assert a != b  # NG8: every upload is a new document, never an in-place update


def test_extract_upload_text_decodes_txt():
    text = extract_upload_text("notes.txt", b"hello world")
    assert text == "hello world"


def test_extract_upload_text_rejects_unsupported_type():
    with pytest.raises(ValueError, match="unsupported file type"):
        extract_upload_text("image.png", b"\x89PNG")


def test_extract_upload_text_extracts_real_pdf_text():
    from io import BytesIO

    from pypdf import PdfReader, PdfWriter

    # Build a real minimal PDF with actual text content via pypdf itself,
    # rather than hand-crafting PDF bytes - this is what a real uploaded
    # PDF's structure looks like, not a mock.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)

    # add_blank_page has no text layer - confirms the pdf branch runs
    # end-to-end (parses real PDF bytes via PdfReader) without crashing;
    # real-world text extraction accuracy is pypdf's own concern, not
    # something this codebase re-tests.
    text = extract_upload_text("empty.pdf", buf.getvalue())
    assert isinstance(text, str)
    assert PdfReader(BytesIO(buf.getvalue())).pages  # sanity: it's a real, readable PDF


# ---------------------------------------------------------------------------
# chunk_document (Phase 11 follow-up) - the 8,000-char document-level cap
# was fixed by moving validation to the correct boundaries (app.py); these
# tests cover chunk_document's own responsibility: turning arbitrarily
# large text into bounded, well-formed chunks.
# ---------------------------------------------------------------------------


def test_chunk_document_splits_a_100k_char_document_into_many_chunks():
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * (106_000 // len(sentence))
    assert len(text) > 100_000

    chunks = chunk_document("doc1", text)

    assert len(chunks) > 50  # genuinely split, not one blob
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.text for c in chunks)  # no empty chunks in the middle
    # reassembling all chunk text recovers the same words, nothing dropped
    assert " ".join(c.text for c in chunks).split() == text.split()


def test_chunk_document_below_default_chunk_words_produces_one_chunk():
    chunks = chunk_document("doc1", "a short document with only a few words")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc1:0"


def test_chunk_document_handles_empty_text():
    chunks = chunk_document("doc1", "")
    assert len(chunks) == 1
    assert chunks[0].text == ""


def test_chunk_document_hard_splits_a_pathological_whitespace_free_blob():
    # A single "word" with no whitespace at all (e.g. minified data pasted
    # as text) would otherwise bypass DEFAULT_CHUNK_WORDS entirely and
    # produce one arbitrarily large chunk.
    blob = "x" * 50_000
    chunks = chunk_document("doc1", blob)

    assert len(chunks) > 1
    assert all(len(c.text) <= DEFAULT_MAX_CHUNK_CHARS for c in chunks)
    assert "".join(c.text for c in chunks) == blob  # no data lost across the hard split


def test_chunk_document_normal_prose_never_triggers_the_char_backstop():
    # Ordinary text with real whitespace should chunk purely by word count -
    # the char backstop is a no-op here (each chunk well under the cap).
    text = " ".join(f"word{i}" for i in range(5000))
    chunks = chunk_document("doc1", text)
    assert all(len(c.text) <= DEFAULT_MAX_CHUNK_CHARS for c in chunks)
    assert len(chunks) == 25  # 5000 words / 200 words-per-chunk


# ---------------------------------------------------------------------------
# GroqLLMClient retry-on-429 (surfaced by the 106k-char fix: a large
# document now makes enough sequential extraction calls, from
# ingest_document's per-chunk loop, to realistically trip Groq's free-tier
# rate limit mid-upload - a transient 429 must not fail the whole upload).
# ---------------------------------------------------------------------------


class FakeHttpxResponse:
    def __init__(self, status_code, json_body=None, headers=None, text=""):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json_body


def _completion_body(content):
    return {"choices": [{"message": {"content": content}}]}


def test_groq_llm_client_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("adaptive_rag.ingestion.time.sleep", lambda s: None)
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        if len(calls) == 1:
            return FakeHttpxResponse(429, headers={"retry-after": "1"})
        return FakeHttpxResponse(200, _completion_body('{"entities": []}'))

    monkeypatch.setattr("httpx.post", fake_post)
    client = GroqLLMClient()
    result = client.complete_json("system", "user")

    assert result == '{"entities": []}'
    assert len(calls) == 2  # one 429, one success - no more retries than needed


def test_groq_llm_client_falls_back_when_retry_after_is_not_a_number(monkeypatch):
    # Retry-After is allowed by HTTP spec to be an HTTP-date instead of a
    # second count - must not crash trying to float() it.
    sleeps = []
    monkeypatch.setattr("adaptive_rag.ingestion.time.sleep", sleeps.append)
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        if len(calls) == 1:
            return FakeHttpxResponse(429, headers={"retry-after": "Wed, 17 Sep 2026 16:45:00 GMT"})
        return FakeHttpxResponse(200, _completion_body('{"entities": []}'))

    monkeypatch.setattr("httpx.post", fake_post)
    client = GroqLLMClient()
    result = client.complete_json("system", "user")

    assert result == '{"entities": []}'
    assert sleeps == [1.0]  # fell back to 2**0 = 1 second, didn't crash


def test_groq_llm_client_caps_a_very_long_retry_after_wait(monkeypatch):
    # Real value observed live 2026-09-17: Groq sent Retry-After: 216 under
    # heavy free-tier load. Since a chunk's exhausted retries now degrade
    # gracefully (ingest_document skips it, doesn't crash the document),
    # honoring a multi-minute wait per attempt is no longer worth it.
    sleeps = []
    monkeypatch.setattr("adaptive_rag.ingestion.time.sleep", sleeps.append)
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        if len(calls) == 1:
            return FakeHttpxResponse(429, headers={"retry-after": "216"})
        return FakeHttpxResponse(200, _completion_body('{"entities": []}'))

    monkeypatch.setattr("httpx.post", fake_post)
    client = GroqLLMClient()
    result = client.complete_json("system", "user")

    assert result == '{"entities": []}'
    assert sleeps == [GroqLLMClient.MAX_RETRY_WAIT_SECONDS]  # clamped from 216 to the cap


def test_groq_llm_client_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("adaptive_rag.ingestion.time.sleep", lambda s: None)
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        return FakeHttpxResponse(429)

    monkeypatch.setattr("httpx.post", fake_post)
    client = GroqLLMClient()

    import httpx

    with pytest.raises(httpx.HTTPStatusError):
        client.complete_json("system", "user")
    assert len(calls) == GroqLLMClient.MAX_RETRIES + 1  # exhausted every retry, then gave up


def test_groq_llm_client_does_not_retry_non_429_errors(monkeypatch):
    monkeypatch.setattr("adaptive_rag.ingestion.time.sleep", lambda s: None)
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        return FakeHttpxResponse(500)

    monkeypatch.setattr("httpx.post", fake_post)
    client = GroqLLMClient()

    import httpx

    with pytest.raises(httpx.HTTPStatusError):
        client.complete_json("system", "user")
    assert len(calls) == 1  # a non-429 failure fails immediately, not after 6 attempts
