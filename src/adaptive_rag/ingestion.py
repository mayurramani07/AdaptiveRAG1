"""Ingestion pipeline (Phase 2, SS3): chunk -> extract -> resolve -> relate -> validate -> load.

Entity and relationship extraction use a small hosted LLM (Groq, via
`settings.groq_extraction_model` - deliberately NOT `settings.groq_model`,
which is reserved for the one expensive generation call, FR18/NFR4). Every
extracted entity/relationship is grounded against the source chunk text
before it's trusted (hallucination guard) - the LLM proposes, the chunk text
disposes.

Entity resolution follows a multi-signal pipeline: normalize -> exact match
-> alias/rule candidate -> embedding/context similarity (no-op until an
embedding provider is chosen, SS9 - mirrors gateway.semantic_cache_get's
convention) -> LLM tie-break on the remaining ambiguous cases only. This
replaces plain string-normalization dedup, which could silently merge
unrelated entities that happen to share a surface form.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from adaptive_rag.config import get_settings

DEFAULT_CHUNK_WORDS = 200
DEFAULT_MIN_RELATIONSHIP_CONFIDENCE = 0.5

_SUFFIXES = ("inc.", "inc", "corp.", "corp", "llc", "ltd.", "ltd", "co.", "co")
_LABEL_RE = re.compile(r"[^A-Z0-9_]")

_ENTITY_SYSTEM_PROMPT = (
    "Extract named entities (people, organizations, products, or places) from the user's text. "
    'Respond with strict JSON only: {"entities": [{"surface": "exact substring from the text", '
    '"type": "PERSON|ORG|PRODUCT|GPE|OTHER"}]}. Only include a surface if it appears verbatim in '
    "the text. Do not invent entities that are not there. No prose, JSON only."
)

_RELATIONSHIP_SYSTEM_PROMPT = (
    "Given a text passage and a list of named entities found in it, extract factual relationships "
    "strictly between those entities. Respond with strict JSON only: "
    '{"relationships": [{"source": "entity surface exactly as given", "relationship": '
    '"SHORT_UPPER_SNAKE_CASE_LABEL", "target": "entity surface exactly as given", "confidence": '
    '0.0-1.0, "evidence": "exact quoted sentence from the text supporting this"}]}. Use only '
    "entities from the provided list, spelled exactly as given. Always resolve the relationship "
    "direction to active voice regardless of passive phrasing in the source text (e.g. 'X was "
    "founded by Y' -> source=Y, relationship=FOUNDED, target=X). If no relationship is stated "
    "between a pair, omit it. No prose, JSON only."
)

_DISAMBIGUATION_SYSTEM_PROMPT = (
    "You resolve entity mentions in a knowledge graph. Given two entity names and short context "
    'for each, decide if they refer to the same real-world entity. Respond with strict JSON only: '
    '{"same_entity": true|false}. When unsure, answer false.'
)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


@dataclass
class EntityMention:
    surface: str
    canonical: str
    doc_id: str
    chunk_id: str
    entity_type: str = "OTHER"


@dataclass
class CanonicalEntity:
    key: str
    display_name: str
    entity_type: str = "OTHER"
    aliases: set[str] = field(default_factory=set)


@dataclass
class Relationship:
    source: str  # canonical entity key
    target: str
    doc_id: str
    chunk_id: str
    confidence: float
    label: str = "RELATED_TO"
    evidence: str = ""


class Neo4jLike(Protocol):
    def execute_query(self, query: str, **params: Any) -> Any: ...


class LLMLike(Protocol):
    def complete_json(self, system: str, user: str) -> str: ...


@lru_cache
def get_neo4j_driver() -> Neo4jLike:
    from neo4j import GraphDatabase

    settings = get_settings()
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


class GroqLLMClient:
    """Calls Groq's OpenAI-compatible chat completions endpoint with the
    small extraction model - never `settings.groq_model` (FR18/NFR4)."""

    def complete_json(self, system: str, user: str) -> str:
        import httpx

        settings = get_settings()
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_extraction_model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


@lru_cache
def get_llm_client() -> LLMLike:
    return GroqLLMClient()


def _safe_json(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def chunk_document(doc_id: str, text: str, chunk_words: int = DEFAULT_CHUNK_WORDS) -> list[Chunk]:
    # ponytail: fixed word-count chunks, no overlap/sentence-awareness.
    # Add overlap if answers start citing boundary-split context wrong.
    words = text.split()
    chunks = [
        Chunk(doc_id=doc_id, chunk_id=f"{doc_id}:{i // chunk_words}", text=" ".join(words[i : i + chunk_words]))
        for i in range(0, len(words), chunk_words)
    ]
    return chunks or [Chunk(doc_id=doc_id, chunk_id=f"{doc_id}:0", text="")]


def _canonicalize(surface: str) -> str:
    normalized = surface.strip().lower()
    for suffix in _SUFFIXES:
        if normalized.endswith(" " + suffix):
            return normalized[: -(len(suffix) + 1)].strip()
    return normalized


def extract_entities(chunk: Chunk, llm: LLMLike | None = None) -> list[EntityMention]:
    """LLM-proposed entities, kept only if grounded verbatim in chunk.text -
    the LLM can mistype/hallucinate a name; the source text is authoritative."""
    client = llm or get_llm_client()
    data = _safe_json(client.complete_json(_ENTITY_SYSTEM_PROMPT, chunk.text))
    candidates = data.get("entities") if isinstance(data, dict) else None
    if not isinstance(candidates, list):
        return []  # invalid/unparseable LLM output degrades to "no entities", not a crash

    seen: set[str] = set()
    mentions = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        surface = item.get("surface")
        if not isinstance(surface, str):
            continue
        surface = surface.strip()
        if len(surface) < 2 or surface not in chunk.text:
            continue  # anti-hallucination: must appear verbatim in the source chunk
        norm = surface.lower()
        if norm in seen:
            continue
        seen.add(norm)
        entity_type = item.get("type")
        mentions.append(
            EntityMention(
                surface=surface,
                canonical=_canonicalize(surface),
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                entity_type=entity_type if isinstance(entity_type, str) and entity_type else "OTHER",
            )
        )
    return mentions


def _find_alias_candidate(raw_key: str, entity_type: str, registry: dict[str, CanonicalEntity]) -> str | None:
    """Rule-based candidate lookup: an existing canonical entity whose
    normalized words overlap enough with `raw_key` to warrant disambiguation
    (e.g. "Google" vs "Google Maps"), short of an outright exact match
    (handled separately, before this is ever called). Confidently
    different-typed entities (PERSON vs GPE, etc.) never alias to each
    other - exact-key matches still bypass this check by design, per the
    given Normalization -> Exact match -> Alias/rule ordering."""
    tokens = set(raw_key.split())
    if not tokens:
        return None
    for key, entity in registry.items():
        if entity.entity_type != "OTHER" and entity_type != "OTHER" and entity.entity_type != entity_type:
            continue
        existing_tokens = set(key.split())
        if not existing_tokens or tokens == existing_tokens:
            continue
        if tokens.issubset(existing_tokens) or existing_tokens.issubset(tokens):
            return key
    return None


def _embedding_similarity_merge(mention: EntityMention, candidate: CanonicalEntity) -> bool | None:
    """Returns True/False when an embedding similarity check can confidently
    decide, or None to fall through to the LLM tie-break - including when no
    embedding provider is configured yet (SS9 open decision; no-op until
    then, mirroring gateway.semantic_cache_get)."""
    if not get_settings().embedding_provider:
        return None
    raise NotImplementedError("embedding-based entity resolution not yet wired (SS9)")


def _llm_confirms_same_entity(mention: EntityMention, candidate: CanonicalEntity, llm: LLMLike | None) -> bool:
    client = llm or get_llm_client()
    user = (
        f'Entity A: "{mention.surface}"\n'
        f'Entity B: "{candidate.display_name}" (known aliases: {sorted(candidate.aliases)})\n'
        "Are A and B the same real-world entity?"
    )
    data = _safe_json(client.complete_json(_DISAMBIGUATION_SYSTEM_PROMPT, user))
    # ponytail: fail toward "different entities" on any unparseable/missing
    # answer - a false merge silently corrupts the graph, a missed merge
    # just leaves two nodes. Upgrade to a stricter retry if duplication
    # turns out to matter more than that in practice.
    return isinstance(data, dict) and data.get("same_entity") is True


def resolve_entities(mentions: Iterable[EntityMention], llm: LLMLike | None = None) -> dict[str, CanonicalEntity]:
    """Normalization -> exact match -> alias/rule candidate -> embedding
    similarity -> LLM tie-break on ambiguous cases -> canonical entity."""
    registry: dict[str, CanonicalEntity] = {}
    mapping: dict[str, str] = {}

    for mention in mentions:
        raw_key = mention.canonical
        if raw_key in mapping:
            continue  # exact match on the already-normalized key - free, no LLM call

        candidate_key = _find_alias_candidate(raw_key, mention.entity_type, registry)
        if candidate_key is None:
            registry[raw_key] = CanonicalEntity(key=raw_key, display_name=mention.surface, entity_type=mention.entity_type, aliases={raw_key})
            mapping[raw_key] = raw_key
            continue

        candidate = registry[candidate_key]
        same = _embedding_similarity_merge(mention, candidate)
        if same is None:
            same = _llm_confirms_same_entity(mention, candidate, llm)

        if same:
            candidate.aliases.add(raw_key)
            mapping[raw_key] = candidate_key
        else:
            registry[raw_key] = CanonicalEntity(key=raw_key, display_name=mention.surface, entity_type=mention.entity_type, aliases={raw_key})
            mapping[raw_key] = raw_key

    return {raw_key: registry[reg_key] for raw_key, reg_key in mapping.items()}


def extract_relationships(chunk: Chunk, mentions: list[EntityMention], llm: LLMLike | None = None) -> list[Relationship]:
    """Structured {source, relationship, target, confidence, evidence}
    extraction, restricted to entities actually resolved in this chunk and
    grounded by requiring evidence to appear verbatim in the chunk text."""
    if len(mentions) < 2:
        return []
    client = llm or get_llm_client()
    surface_to_canonical = {m.surface: m.canonical for m in mentions}
    user = f"Entities: {sorted(surface_to_canonical)}\n\nText: {chunk.text}"
    data = _safe_json(client.complete_json(_RELATIONSHIP_SYSTEM_PROMPT, user))
    candidates = data.get("relationships") if isinstance(data, dict) else None
    if not isinstance(candidates, list):
        return []

    relationships = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        source_surface, target_surface = item.get("source"), item.get("target")
        label, evidence, confidence = item.get("relationship"), item.get("evidence"), item.get("confidence")
        if not all(isinstance(v, str) and v.strip() for v in (source_surface, target_surface, label, evidence)):
            continue
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            continue
        if source_surface not in surface_to_canonical or target_surface not in surface_to_canonical:
            continue  # only relationships between entities actually resolved in this chunk
        source_key, target_key = surface_to_canonical[source_surface], surface_to_canonical[target_surface]
        if source_key == target_key:
            continue
        evidence = evidence.strip()
        if evidence not in chunk.text:
            continue  # anti-hallucination: evidence must be grounded in the source chunk
        clean_label = _LABEL_RE.sub("", label.strip().upper().replace(" ", "_")) or "RELATED_TO"
        relationships.append(
            Relationship(
                source=source_key,
                target=target_key,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                confidence=max(0.0, min(1.0, float(confidence))),
                label=clean_label,
                evidence=evidence,
            )
        )
    return relationships


def validate_relationships(relationships: Iterable[Relationship], min_confidence: float = DEFAULT_MIN_RELATIONSHIP_CONFIDENCE) -> list[Relationship]:
    return [r for r in relationships if r.confidence >= min_confidence and r.source != r.target]


def load_into_neo4j(
    driver: Neo4jLike,
    doc_id: str,
    chunks: list[Chunk],
    mentions: list[EntityMention],
    registry: dict[str, CanonicalEntity],
    relationships: list[Relationship],
) -> None:
    """Loads validated evidence into Neo4j. Every Entity node reaches source
    text only via MENTIONED_IN -> Chunk.text, and every Chunk carries its
    doc_id - satisfies FR-ING2's "pointer back to source document + passage"
    for both nodes and edges. Relationship edges additionally carry their
    semantic type/confidence/evidence (`type` is part of the MERGE key so
    two distinct relationship types between the same pair in the same chunk
    don't collapse onto one edge)."""
    driver.execute_query("MERGE (:Document {doc_id: $doc_id})", doc_id=doc_id)
    for chunk in chunks:
        driver.execute_query(
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (c:Chunk {chunk_id: $chunk_id}) SET c.text = $text, c.doc_id = $doc_id "
            "MERGE (c)-[:PART_OF]->(d)",
            doc_id=doc_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
        )
    for mention in mentions:
        entity = registry[mention.canonical]
        driver.execute_query(
            "MERGE (e:Entity {name: $name}) SET e.type = $type "
            "WITH e MATCH (c:Chunk {chunk_id: $chunk_id}) MERGE (e)-[:MENTIONED_IN]->(c)",
            name=entity.display_name,
            type=entity.entity_type,
            chunk_id=mention.chunk_id,
        )
    for rel in relationships:
        driver.execute_query(
            "MATCH (a:Entity {name: $source}), (b:Entity {name: $target}) "
            "MERGE (a)-[r:RELATED_TO {doc_id: $doc_id, chunk_id: $chunk_id, type: $type}]->(b) "
            "SET r.confidence = $confidence, r.evidence = $evidence",
            source=registry[rel.source].display_name,
            target=registry[rel.target].display_name,
            doc_id=rel.doc_id,
            chunk_id=rel.chunk_id,
            type=rel.label,
            confidence=rel.confidence,
            evidence=rel.evidence,
        )


def ingest_document(driver: Neo4jLike, doc_id: str, text: str, llm: LLMLike | None = None) -> None:
    """Runs the full pipeline for one document."""
    chunks = chunk_document(doc_id, text)
    all_mentions: list[EntityMention] = []
    all_relationships: list[Relationship] = []
    for chunk in chunks:
        mentions = extract_entities(chunk, llm=llm)
        all_mentions.extend(mentions)
        all_relationships.extend(extract_relationships(chunk, mentions, llm=llm))
    registry = resolve_entities(all_mentions, llm=llm)
    validated = validate_relationships(all_relationships)
    load_into_neo4j(driver, doc_id, chunks, all_mentions, registry, validated)


def sync_graph(driver: Neo4jLike, documents: dict[str, str], llm: LLMLike | None = None) -> None:
    """Scheduled re-ingestion entrypoint (FR-ING4, NG4 cadence): re-loads
    `documents` and removes graph nodes/edges derived from a deleted OR
    changed source document. Wiring this to an actual scheduler (cron/Render
    job) is deferred - Render provisioning is still blocked (Phase 0).

    A document's Chunk subgraph is wiped and rebuilt on every sync rather
    than diffed against its previous chunk boundaries, so editing a
    document's content never leaves old chunks/entity mentions stranded,
    pointing at text that no longer exists in the source."""
    keep_ids = list(documents.keys())
    driver.execute_query(
        "MATCH (d:Document) WHERE NOT d.doc_id IN $keep_ids OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk) DETACH DELETE d, c",
        keep_ids=keep_ids,
    )
    for doc_id in documents:
        driver.execute_query(
            "MATCH (d:Document {doc_id: $doc_id})<-[:PART_OF]-(c:Chunk) DETACH DELETE c",
            doc_id=doc_id,
        )
    driver.execute_query("MATCH (e:Entity) WHERE NOT (e)-[:MENTIONED_IN]->() DETACH DELETE e")
    for doc_id, text in documents.items():
        ingest_document(driver, doc_id, text, llm=llm)
