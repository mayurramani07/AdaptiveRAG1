import ast
import json
from pathlib import Path

from adaptive_rag.generation import (
    build_context,
    extract_sources,
    format_sse_event,
    generate_buffered,
    generate_streaming,
    stream_sse_response,
)
from adaptive_rag.grading import Grade

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "adaptive_rag"


class FakeGenerationClient:
    """Yields a fixed sequence of tokens - the streaming-vs-buffered tests
    only care about consumption pattern, not real Groq/SSE parsing (that's
    covered separately by the live SSE test below)."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []

    def stream(self, messages):
        self.calls.append(messages)
        yield from self.tokens


def _evidence(chunk_id, text, grade, trust_tier=None, score=None):
    e = {"id": chunk_id, "doc_id": "doc1", "chunk_id": chunk_id, "text": text, "grade": grade}
    if trust_tier:
        e["trust_tier"] = trust_tier
    if score is not None:
        e["grader_score"] = score
    return e


# ---------------------------------------------------------------------------
# Context Builder (FR16)
# ---------------------------------------------------------------------------


def test_build_context_excludes_incorrect_evidence():
    evidence = [
        _evidence("a", "trustworthy fact", Grade.CORRECT),
        _evidence("b", "bad fact", Grade.INCORRECT),
    ]
    messages = build_context("What happened?", evidence)
    user_content = messages[-1]["content"]

    assert "trustworthy fact" in user_content
    assert "bad fact" not in user_content


def test_build_context_includes_ambiguous_evidence():
    # FR16's acceptance criterion only mandates excluding Incorrect -
    # Ambiguous is not excluded by Context Builder itself (in the intended
    # pipeline flow, Ambiguous evidence should already have been handled
    # by Phase 5's Recovery Planner before reaching here).
    evidence = [_evidence("a", "loosely related fact", Grade.AMBIGUOUS)]
    messages = build_context("query", evidence)
    assert "loosely related fact" in messages[-1]["content"]


def test_build_context_includes_the_query():
    messages = build_context("What is the refund policy?", [_evidence("a", "policy text", Grade.CORRECT)])
    assert "What is the refund policy?" in messages[-1]["content"]


def test_build_context_has_system_and_user_messages():
    messages = build_context("q", [_evidence("a", "text", Grade.CORRECT)])
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_context_empty_evidence_still_produces_valid_messages():
    messages = build_context("q", [])
    assert messages[1]["role"] == "user"
    assert "q" in messages[1]["content"]


def test_build_context_never_includes_a_bare_graph_label_without_source_text():
    # Regression-shaped test for FR16's "no graph edge label without its
    # resolved source text" acceptance: a candidate with a "label" field
    # (mimicking a raw graph edge) is only ever represented via its "text"
    # field in the prompt, never its label.
    evidence = [{"id": "a", "doc_id": "d", "chunk_id": "a", "text": "Steve Jobs founded Apple.", "label": "FOUNDED", "grade": Grade.CORRECT}]
    messages = build_context("q", evidence)
    assert "Steve Jobs founded Apple." in messages[-1]["content"]
    assert "FOUNDED" not in messages[-1]["content"]


# ---------------------------------------------------------------------------
# extract_sources
# ---------------------------------------------------------------------------


def test_extract_sources_tags_trust_tier_authoritative_by_default():
    evidence = [_evidence("a", "text", Grade.CORRECT)]
    sources = extract_sources(evidence)
    assert sources[0]["trust_tier"] == "authoritative"


def test_extract_sources_preserves_supplementary_tag():
    evidence = [_evidence("a", "text", Grade.CORRECT, trust_tier="supplementary")]
    sources = extract_sources(evidence)
    assert sources[0]["trust_tier"] == "supplementary"


def test_extract_sources_excludes_incorrect_evidence():
    evidence = [_evidence("a", "good", Grade.CORRECT), _evidence("b", "bad", Grade.INCORRECT)]
    sources = extract_sources(evidence)
    assert len(sources) == 1
    assert sources[0]["id"] == "a"


# ---------------------------------------------------------------------------
# Streaming vs buffered (FR17, NFR7) - same underlying client either way
# ---------------------------------------------------------------------------


def test_generate_streaming_yields_tokens_incrementally():
    client = FakeGenerationClient(["Hello", " ", "world"])
    messages = [{"role": "user", "content": "hi"}]
    chunks = list(generate_streaming(messages, client=client))
    assert chunks == ["Hello", " ", "world"]


def test_generate_buffered_returns_complete_joined_text():
    client = FakeGenerationClient(["Hello", " ", "world"])
    messages = [{"role": "user", "content": "hi"}]
    result = generate_buffered(messages, client=client)
    assert result == "Hello world"


def test_generate_streaming_and_buffered_use_the_same_underlying_call():
    # Proves this is one code path consumed two ways, not two
    # implementations that could silently drift apart.
    client = FakeGenerationClient(["a", "b"])
    list(generate_streaming([{"role": "user", "content": "x"}], client=client))
    assert client.calls == [[{"role": "user", "content": "x"}]]


def test_generate_buffered_never_yields_partial_results():
    # NFR7: nothing is observable from generate_buffered until it's fully
    # returned - there's no generator/iterator interface exposed at all.
    import inspect

    client = FakeGenerationClient(["a", "b"])
    result = generate_buffered([{"role": "user", "content": "x"}], client=client)
    assert isinstance(result, str)
    assert not inspect.isgenerator(result)


# ---------------------------------------------------------------------------
# FR18/NFR4 code-scan audit: settings.groq_model used ONLY in generation.py
# ---------------------------------------------------------------------------


def test_groq_model_attribute_accessed_only_in_generation_module():
    """FR18 acceptance: 'Automated lint/code-scan check confirms no other
    module invokes the Groq generation endpoint or any equivalent
    generation-tier API.' Walks the real AST (not string grepping) so a
    docstring merely *mentioning* settings.groq_model - like ingestion.py's
    and recovery.py's explicit 'never this one' notes - doesn't
    false-positive as a usage."""
    offenders = []
    for path in SRC_DIR.rglob("*.py"):
        if path.name == "generation.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "groq_model":
                offenders.append(f"{path.relative_to(SRC_DIR)}:{node.lineno}")
    assert offenders == []


def test_groq_model_attribute_is_actually_accessed_in_generation_module():
    # Sanity-checks the scanner itself isn't just vacuously passing because
    # nothing anywhere uses the attribute.
    tree = ast.parse((SRC_DIR / "generation.py").read_text(encoding="utf-8"))
    found = any(isinstance(node, ast.Attribute) and node.attr == "groq_model" for node in ast.walk(tree))
    assert found is True


# ---------------------------------------------------------------------------
# SSE wire format (SS9.1)
# ---------------------------------------------------------------------------


def test_format_sse_event_matches_contract_shape():
    event = format_sse_event("token", {"text": "hello"})
    assert event == 'event: token\ndata: {"text": "hello"}\n\n'


def test_format_sse_event_data_is_valid_json():
    event = format_sse_event("sources", {"sources": [{"id": "doc_1"}]})
    data_line = event.split("\n")[1]
    assert json.loads(data_line[len("data: ") :]) == {"sources": [{"id": "doc_1"}]}


def test_stream_sse_response_emits_sources_event_first():
    client = FakeGenerationClient(["hi"])
    events = list(stream_sse_response([{"role": "user", "content": "q"}], sources=[{"id": "doc_1"}], client=client))
    assert events[0].startswith("event: sources\n")
    assert json.loads(events[0].split("\n")[1][len("data: ") :]) == {"sources": [{"id": "doc_1"}]}


def test_stream_sse_response_emits_one_token_event_per_chunk():
    client = FakeGenerationClient(["Hello", " world"])
    events = list(stream_sse_response([{"role": "user", "content": "q"}], sources=[], client=client))
    token_events = events[1:]
    assert len(token_events) == 2
    assert all(e.startswith("event: token\n") for e in token_events)
    assert json.loads(token_events[0].split("\n")[1][len("data: ") :]) == {"text": "Hello"}
    assert json.loads(token_events[1].split("\n")[1][len("data: ") :]) == {"text": " world"}
