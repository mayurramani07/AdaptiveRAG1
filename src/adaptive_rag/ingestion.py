"""Ingestion pipeline (Phase 2, SS3): chunk -> extract -> resolve -> relate -> validate -> load.

Entity/relationship extraction here is a regex/heuristic placeholder, not a
trained NER model - the extraction-model choice is still an open decision
(project context SS9). Swap extract_entities/extract_relationships for a real
model later without touching chunking, resolution, validation, or the Neo4j
loading shape - that shape is what guarantees every node/edge resolves back
to source text (SS3, FR-ING2), not the extraction quality.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from adaptive_rag.config import get_settings

DEFAULT_CHUNK_WORDS = 200
DEFAULT_MIN_RELATIONSHIP_CONFIDENCE = 0.5

_SUFFIXES = ("inc.", "inc", "corp.", "corp", "llc", "ltd.", "ltd", "co.", "co")
_CAPWORD_RE = re.compile(r"\b[A-Z][\w&]*(?:\s+(?:[A-Z][\w&.]*|of|the|and))*\b")
_SENTENCE_RE = re.compile(r"[.!?]")


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


@dataclass
class Relationship:
    source: str  # canonical entity key
    target: str
    doc_id: str
    chunk_id: str
    confidence: float
    label: str = "related_to"


class Neo4jLike(Protocol):
    def execute_query(self, query: str, **params: Any) -> Any: ...


@lru_cache
def get_neo4j_driver() -> Neo4jLike:
    from neo4j import GraphDatabase

    settings = get_settings()
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


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


def extract_entities(chunk: Chunk) -> list[EntityMention]:
    seen: set[str] = set()
    mentions = []
    for match in _CAPWORD_RE.finditer(chunk.text):
        surface = match.group().strip()
        if len(surface) < 2 or surface.lower() in seen:
            continue
        seen.add(surface.lower())
        mentions.append(EntityMention(surface=surface, canonical=_canonicalize(surface), doc_id=chunk.doc_id, chunk_id=chunk.chunk_id))
    return mentions


def resolve_entities(mentions: Iterable[EntityMention]) -> dict[str, str]:
    """canonical key -> display name (first surface form seen for that key)."""
    display_names: dict[str, str] = {}
    for mention in mentions:
        display_names.setdefault(mention.canonical, mention.surface)
    return display_names


def extract_relationships(chunk: Chunk, mentions: list[EntityMention]) -> list[Relationship]:
    """Naive co-occurrence: entities sharing a sentence score higher than
    entities merely sharing a chunk. Real relationship typing ("reports to",
    etc.) is deferred until an extraction model is chosen (SS3)."""
    sentences = _SENTENCE_RE.split(chunk.text)
    surface_by_canonical = {m.canonical: m.surface for m in mentions}
    canonicals = sorted(surface_by_canonical)
    relationships = []
    for i, source in enumerate(canonicals):
        for target in canonicals[i + 1 :]:
            same_sentence = any(surface_by_canonical[source] in s and surface_by_canonical[target] in s for s in sentences)
            confidence = 0.75 if same_sentence else 0.35
            relationships.append(Relationship(source=source, target=target, doc_id=chunk.doc_id, chunk_id=chunk.chunk_id, confidence=confidence))
    return relationships


def validate_relationships(relationships: Iterable[Relationship], min_confidence: float = DEFAULT_MIN_RELATIONSHIP_CONFIDENCE) -> list[Relationship]:
    return [r for r in relationships if r.confidence >= min_confidence and r.source != r.target]


def load_into_neo4j(
    driver: Neo4jLike,
    doc_id: str,
    chunks: list[Chunk],
    mentions: list[EntityMention],
    display_names: dict[str, str],
    relationships: list[Relationship],
) -> None:
    """Loads validated evidence into Neo4j. Every Entity node reaches source
    text only via MENTIONED_IN -> Chunk.text, and every Chunk carries its
    doc_id - satisfies FR-ING2's "pointer back to source document + passage"
    for both nodes and edges."""
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
        driver.execute_query(
            "MERGE (e:Entity {name: $name}) WITH e MATCH (c:Chunk {chunk_id: $chunk_id}) MERGE (e)-[:MENTIONED_IN]->(c)",
            name=display_names[mention.canonical],
            chunk_id=mention.chunk_id,
        )
    for rel in relationships:
        driver.execute_query(
            "MATCH (a:Entity {name: $source}), (b:Entity {name: $target}) "
            "MERGE (a)-[r:RELATED_TO {doc_id: $doc_id, chunk_id: $chunk_id}]->(b) SET r.confidence = $confidence",
            source=display_names[rel.source],
            target=display_names[rel.target],
            doc_id=rel.doc_id,
            chunk_id=rel.chunk_id,
            confidence=rel.confidence,
        )


def ingest_document(driver: Neo4jLike, doc_id: str, text: str) -> None:
    """Runs the full pipeline for one document."""
    chunks = chunk_document(doc_id, text)
    all_mentions: list[EntityMention] = []
    all_relationships: list[Relationship] = []
    for chunk in chunks:
        mentions = extract_entities(chunk)
        all_mentions.extend(mentions)
        all_relationships.extend(extract_relationships(chunk, mentions))
    display_names = resolve_entities(all_mentions)
    validated = validate_relationships(all_relationships)
    load_into_neo4j(driver, doc_id, chunks, all_mentions, display_names, validated)


def sync_graph(driver: Neo4jLike, documents: dict[str, str]) -> None:
    """Scheduled re-ingestion entrypoint (FR-ING4, NG4 cadence): re-loads
    `documents` and removes graph nodes/edges derived from a deleted OR
    changed source document. Wiring this to an actual scheduler (cron/Render
    job) is deferred - Render provisioning is still blocked (Phase 0).

    A document's Chunk subgraph is wiped and rebuilt on every sync rather
    than diffed against its previous chunk boundaries - the previous version
    only deleted a document's chunks when the doc_id vanished entirely, so
    editing a document's content (same doc_id, fewer/renumbered chunks) left
    old chunks and their entity mentions stranded forever, pointing at text
    that no longer exists in the source (violated FR-ING2/FR-ING4)."""
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
        ingest_document(driver, doc_id, text)
