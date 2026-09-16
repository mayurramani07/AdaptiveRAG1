import time

from adaptive_rag.grounding import (
    DEFAULT_MODE1_PASS_THRESHOLD,
    DEFAULT_MODE2_PASS_THRESHOLD,
    GroundingCheck,
    GroundingStatus,
    check_grounding,
    check_grounding_async,
    detect_risk_category,
    generate_with_grounding_gate,
    is_passed,
    select_mode,
)
from adaptive_rag.recovery import INSUFFICIENT_EVIDENCE_MESSAGE


class ScriptedLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system, user):
        self.calls.append((system, user))
        return self.response


class SlowLLM:
    """Simulates real network latency - used to prove Mode 1 never blocks
    on it (FR19's load-test-shaped acceptance)."""

    def __init__(self, response: str, delay: float):
        self.response = response
        self.delay = delay

    def complete_json(self, system, user):
        time.sleep(self.delay)
        return self.response


class FakeGenerationClient:
    def __init__(self, tokens):
        self.tokens = tokens

    def stream(self, messages):
        yield from self.tokens


def _evidence(text):
    return {"id": "a", "doc_id": "d", "chunk_id": "a", "text": text}


# ---------------------------------------------------------------------------
# check_grounding (core)
# ---------------------------------------------------------------------------


def test_check_grounding_passes_a_grounded_answer():
    llm = ScriptedLLM('{"grounded": true, "confidence": 0.95}')
    check = check_grounding("supported answer", [_evidence("source text")], llm=llm)
    assert check.grounded is True
    assert check.confidence == 0.95


def test_check_grounding_flags_an_ungrounded_answer():
    llm = ScriptedLLM('{"grounded": false, "confidence": 0.9}')
    check = check_grounding("hallucinated answer", [_evidence("unrelated source text")], llm=llm)
    assert check.grounded is False


def test_check_grounding_sends_resolved_evidence_text_to_the_judge():
    llm = ScriptedLLM('{"grounded": true, "confidence": 1.0}')
    check_grounding("answer", [_evidence("the actual source passage")], llm=llm)
    assert "the actual source passage" in llm.calls[0][1]


def test_check_grounding_fails_safe_on_unparseable_llm_output():
    llm = ScriptedLLM("not json at all")
    check = check_grounding("answer", [_evidence("text")], llm=llm)
    assert check.grounded is False
    assert check.confidence == 0.0


def test_check_grounding_fails_safe_on_missing_grounded_key():
    llm = ScriptedLLM('{"confidence": 0.9}')
    check = check_grounding("answer", [_evidence("text")], llm=llm)
    assert check.grounded is False


def test_check_grounding_clamps_out_of_range_confidence():
    llm = ScriptedLLM('{"grounded": true, "confidence": 1.5}')
    check = check_grounding("answer", [_evidence("text")], llm=llm)
    assert check.confidence == 1.0


# ---------------------------------------------------------------------------
# is_passed thresholding
# ---------------------------------------------------------------------------


def test_is_passed_requires_both_grounded_and_confidence():
    assert is_passed(GroundingCheck(grounded=True, confidence=0.9), threshold=0.5) is True
    assert is_passed(GroundingCheck(grounded=True, confidence=0.1), threshold=0.5) is False
    assert is_passed(GroundingCheck(grounded=False, confidence=0.99), threshold=0.5) is False


# ---------------------------------------------------------------------------
# Mode 1 - async, non-blocking (FR19)
# ---------------------------------------------------------------------------


def test_check_grounding_async_returns_immediately_regardless_of_llm_latency():
    """FR19's load-test-shaped acceptance: response latency for Mode 1 is
    unaffected by grounding check completion time. Uses an LLM that sleeps
    for 2 seconds - if check_grounding_async blocked on it, this test
    would take >2s; it must return near-instantly instead."""
    slow_llm = SlowLLM('{"grounded": true, "confidence": 0.9}', delay=2.0)
    start = time.monotonic()
    handle = check_grounding_async("answer", [_evidence("text")], llm=slow_llm)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5  # returned long before the 2s LLM call could finish
    assert handle.status(timeout=0) == GroundingStatus.PENDING


def test_check_grounding_async_eventually_reports_passed():
    llm = ScriptedLLM('{"grounded": true, "confidence": 0.9}')
    handle = check_grounding_async("answer", [_evidence("text")], llm=llm)
    assert handle.status(timeout=5) == GroundingStatus.PASSED


def test_check_grounding_async_eventually_reports_flagged_not_failed():
    # Mode 1 is a detector, not a gate - an ungrounded answer is FLAGGED,
    # never FAILED (FAILED is Mode 2's gating vocabulary only).
    llm = ScriptedLLM('{"grounded": false, "confidence": 0.9}')
    handle = check_grounding_async("answer", [_evidence("text")], llm=llm)
    assert handle.status(timeout=5) == GroundingStatus.FLAGGED


def test_check_grounding_async_result_blocks_until_done_for_logging():
    llm = ScriptedLLM('{"grounded": true, "confidence": 0.8}')
    handle = check_grounding_async("answer", [_evidence("text")], llm=llm)
    check = handle.result()
    assert check.grounded is True


# ---------------------------------------------------------------------------
# Mode 2 - sync, blocking, gated release (FR20)
# ---------------------------------------------------------------------------


def test_generate_with_grounding_gate_releases_answer_on_pass():
    gen_client = FakeGenerationClient(["The", " real", " answer"])
    llm = ScriptedLLM('{"grounded": true, "confidence": 0.9}')
    result = generate_with_grounding_gate([{"role": "user", "content": "q"}], [_evidence("text")], client=gen_client, llm=llm)
    assert result == "The real answer"


def test_generate_with_grounding_gate_never_leaks_raw_answer_on_fail():
    """FR20 acceptance, matched exactly: a test forcing FAIL confirms the
    fallback response is returned and the ungrounded buffered answer is
    never sent."""
    gen_client = FakeGenerationClient(["a hallucinated", " answer"])
    llm = ScriptedLLM('{"grounded": false, "confidence": 0.9}')
    result = generate_with_grounding_gate([{"role": "user", "content": "q"}], [_evidence("text")], client=gen_client, llm=llm)

    assert result == INSUFFICIENT_EVIDENCE_MESSAGE
    assert "hallucinated" not in result


def test_generate_with_grounding_gate_fails_below_confidence_threshold():
    # grounded=True but confidence below Mode 2's (higher) bar must still FAIL
    gen_client = FakeGenerationClient(["borderline answer"])
    llm = ScriptedLLM('{"grounded": true, "confidence": 0.6}')
    result = generate_with_grounding_gate(
        [{"role": "user", "content": "q"}], [_evidence("text")], client=gen_client, llm=llm, pass_threshold=DEFAULT_MODE2_PASS_THRESHOLD
    )
    assert result == INSUFFICIENT_EVIDENCE_MESSAGE


def test_mode2_threshold_is_stricter_than_mode1_threshold():
    # SS15: Mode 2 (gated release) should demand more confidence than
    # Mode 1 (detector only, no release decision rides on it).
    assert DEFAULT_MODE2_PASS_THRESHOLD > DEFAULT_MODE1_PASS_THRESHOLD


# ---------------------------------------------------------------------------
# FR21 - high-risk category detection / mode routing
# ---------------------------------------------------------------------------


def test_detect_risk_category_medical():
    assert detect_risk_category("What is the recommended dosage for ibuprofen?") == "medical"


def test_detect_risk_category_legal():
    assert detect_risk_category("Can my employer terminate employment without notice?") == "legal"


def test_detect_risk_category_financial():
    assert detect_risk_category("Should I take out a mortgage this year?") == "financial"


def test_detect_risk_category_none_for_ordinary_query():
    assert detect_risk_category("What is your refund policy?") is None


def test_select_mode_routes_high_risk_to_mode2():
    assert select_mode("What medication should I take for a headache?") == "mode2"


def test_select_mode_routes_ordinary_query_to_mode1():
    assert select_mode("What time does the store open?") == "mode1"


def test_select_mode_decided_from_query_text_alone_before_generation():
    # NFR7: mode selection must not require generation output - proven
    # structurally here since select_mode's signature only accepts a query
    # string, nothing generation-shaped.
    import inspect

    sig = inspect.signature(select_mode)
    assert list(sig.parameters) == ["query"]
