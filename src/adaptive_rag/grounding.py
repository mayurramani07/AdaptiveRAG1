"""Grounding checks (Phase 7, FR19-FR21): verifies a generated answer is
actually supported by the evidence it was built from - a hallucination
detector, run in one of two modes depending on the query's risk category.

Implemented as an LLM-as-judge check using the same small Groq model
ingestion/recovery already call (`settings.groq_extraction_model`, never
`settings.groq_model` - FR18/NFR4's separation holds here too), rather than
adding a new NLI-specific dependency or signing up for a new hosted API.
SS2.3 allows either "hosted free-tier NLI API or self-hosted small model"
for this component - reusing infra already wired in this codebase is the
same reasoning Phase 5's evidence grader used to reuse Phase 4's reranker
instead of adding a second model. Verified 2026-09-16 against the real Groq
API: correctly grounded an accurate paraphrase and correctly flagged a
hallucinated answer (wrong number + a fabricated claim) as ungrounded.

Mode 1 (FR19, default) runs the check on a background thread and returns a
handle immediately - it is a **detector, not a gate**: nothing in this
module can retract tokens `generation.stream_sse_response` already sent.
Mode 2 (FR20, high-risk) is synchronous and blocking - `generate_with_grounding_gate`
never returns the raw generated answer on FAIL, only the fallback message.
FR21's category detection is a rule-based baseline (no labeled training
data exists yet to do better - same placeholder-classifier situation
Phase 3's `plan_query` is in), and runs on the query text alone, so mode
selection is always decided before generation begins (NFR7).
"""
from __future__ import annotations

import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from adaptive_rag.generation import GenerationLike, generate_buffered
from adaptive_rag.ingestion import LLMLike, get_llm_client
from adaptive_rag.recovery import INSUFFICIENT_EVIDENCE_MESSAGE

# ponytail: provisional, SS15 lists both as unset tunables. Mode 2 (gated
# release, high-risk) gets a higher bar than Mode 1 (detector only, no
# release decision rides on it) - revisit once labeled data exists.
DEFAULT_MODE1_PASS_THRESHOLD = 0.5
DEFAULT_MODE2_PASS_THRESHOLD = 0.7

_GROUNDING_SYSTEM_PROMPT = (
    "You are a fact-checker. Given a piece of evidence and a generated answer, determine whether "
    "every factual claim in the answer is actually supported by the evidence. Respond with strict "
    'JSON only: {"grounded": true|false, "confidence": 0.0-1.0}. If the answer makes any claim not '
    "present in or contradicted by the evidence, respond grounded: false. No prose, JSON only."
)

# ponytail: rule-based baseline, not a trained classifier - no labeled
# category data exists yet (same situation Phase 3's plan_query is in).
# Revisit once real query logs with category labels exist (Phase 9).
_MEDICAL_KEYWORDS = ("symptom", "diagnosis", "medication", "dosage", "treatment", "disease", "prescription", "side effect", "medical condition")
_LEGAL_KEYWORDS = ("lawsuit", "contract", "liability", "sue ", "legal advice", "attorney", "lawyer", "terminate employment", "compliance violation")
_FINANCIAL_KEYWORDS = ("invest", "tax ", "loan", "mortgage", "retirement", "401k", "stock ", "financial advice", "interest rate")
_HIGH_RISK_CATEGORIES = (("medical", _MEDICAL_KEYWORDS), ("legal", _LEGAL_KEYWORDS), ("financial", _FINANCIAL_KEYWORDS))


class GroundingStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FLAGGED = "flagged"  # Mode 1 only - detector, not a gate
    FAILED = "failed"  # Mode 2 only - gates release


@dataclass
class GroundingCheck:
    grounded: bool
    confidence: float


def _safe_json(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def check_grounding(answer: str, evidence: list[dict], llm: LLMLike | None = None) -> GroundingCheck:
    """Core check: is `answer` actually supported by `evidence`? Grades
    against the evidence's resolved source text (`text` field) - never a
    graph edge label, same SS3 invariant enforced at retrieval time."""
    client = llm or get_llm_client()
    context = "\n\n".join(e["text"] for e in evidence)
    data = _safe_json(client.complete_json(_GROUNDING_SYSTEM_PROMPT, f"Evidence:\n{context}\n\nGenerated answer:\n{answer}"))
    if not isinstance(data, dict) or "grounded" not in data:
        # ponytail: fail toward "not grounded" on unparseable output - same
        # fail-safe direction ingestion.py's disambiguation check uses. A
        # missed pass costs a fallback response; a wrongly-passed check
        # risks shipping a hallucinated answer, which is worse.
        return GroundingCheck(grounded=False, confidence=0.0)
    confidence = data.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.0
    return GroundingCheck(grounded=data.get("grounded") is True, confidence=max(0.0, min(1.0, confidence)))


def is_passed(check: GroundingCheck, threshold: float) -> bool:
    """Never trusts the model's own boolean blindly (same defense-in-depth
    principle as NFR6/grading.py's thresholds) - both the model's verdict
    AND a minimum confidence must hold."""
    return check.grounded and check.confidence >= threshold


@lru_cache
def _get_grounding_pool() -> ThreadPoolExecutor:
    # Same shared-pool-per-process pattern as planning._get_retrieval_pool
    # (not reused directly - that's a private function in another module;
    # this is its own small pool, not a reach across a privacy boundary).
    return ThreadPoolExecutor(max_workers=4)


@dataclass
class AsyncGroundingHandle:
    """FR19: returned immediately - the caller's already-streamed response
    is never delayed by this. `status()` is for logging/eval-feed purposes
    only; nothing in Mode 1 may use it to gate or retract a response."""

    future: Future

    def status(self, timeout: float = 0) -> GroundingStatus:
        try:
            check = self.future.result(timeout=timeout)
        except TimeoutError:
            return GroundingStatus.PENDING
        return GroundingStatus.PASSED if is_passed(check, DEFAULT_MODE1_PASS_THRESHOLD) else GroundingStatus.FLAGGED

    def result(self) -> GroundingCheck:
        """Blocks until the background check finishes - for logging/eval
        feed after the response has already been sent, never for gating."""
        return self.future.result()


def check_grounding_async(answer: str, evidence: list[dict], llm: LLMLike | None = None) -> AsyncGroundingHandle:
    """FR19 (Mode 1, default): kicks off the check on a background thread
    and returns immediately - a detector, not a gate. Load-test-shaped
    acceptance ("response latency unaffected by grounding check completion
    time") holds by construction: this function never blocks on the
    future it creates."""
    future = _get_grounding_pool().submit(check_grounding, answer, evidence, llm)
    return AsyncGroundingHandle(future=future)


def generate_with_grounding_gate(
    messages: list[dict],
    evidence: list[dict],
    client: GenerationLike | None = None,
    llm: LLMLike | None = None,
    pass_threshold: float = DEFAULT_MODE2_PASS_THRESHOLD,
) -> str:
    """FR20 (Mode 2, high-risk): the full Mode 2 flow - buffer the
    generation, run the blocking grounding check, release the buffered
    answer only on PASS. On FAIL, returns `INSUFFICIENT_EVIDENCE_MESSAGE`
    (reused from Phase 5's recovery fallback, same "we don't have enough
    confidence to answer" situation, just triggered by a different check)
    - the ungrounded buffered answer is never returned to the caller,
    verified by a test that grounds a FAIL and asserts the raw answer text
    never appears in the result (FR20 acceptance, matched exactly)."""
    answer = generate_buffered(messages, client=client)
    check = check_grounding(answer, evidence, llm=llm)
    return answer if is_passed(check, pass_threshold) else INSUFFICIENT_EVIDENCE_MESSAGE


def detect_risk_category(query: str) -> str | None:
    """FR21: which high-risk category (if any) this query falls into.
    Runs on the query text alone, before generation begins (NFR7)."""
    normalized = query.lower()
    for category, keywords in _HIGH_RISK_CATEGORIES:
        if any(re.search(re.escape(kw.strip()), normalized) for kw in keywords):
            return category
    return None


def select_mode(query: str) -> str:
    """FR21: mode selection happens purely from the query text, so it is
    always decided before generation begins (NFR7) - never mid-stream."""
    return "mode2" if detect_risk_category(query) else "mode1"
