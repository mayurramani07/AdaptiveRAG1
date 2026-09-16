"""Evidence grading (Phase 5, FR11/FR12): scores retrieved evidence as
Correct/Ambiguous/Incorrect before it's allowed to reach generation.

Reuses `retrieval.get_reranker()` (FastEmbed's local ONNX cross-encoder,
already loaded in memory for Phase 4's reranking step) as the grading
signal, rather than adding a second model. This is a deliberate scope
choice, not an oversight: SS2.3's component table calls for a "self-hosted
fine-tuned model (domain-specific)" for the evidence grader, but fine-tuning
needs labeled data that doesn't exist yet in this project (the labeled
evaluation dataset is Phase 8/9 work). FR11's own acceptance criterion is
explicitly marked "baseline, revisit in Phase 9" - same placeholder-
threshold convention this codebase already follows for the Query Planner
(Phase 3, also a rule-based baseline pending training data) and for
`DEFAULT_MIN_RELATIONSHIP_CONFIDENCE` in ingestion.py. Swap `get_reranker()`
for a fine-tuned grader model later without touching `grade_evidence`'s
shape - it's injected via the same `RerankerLike` Protocol either way.

Per SS3's "the graph is an index, not evidence" rule, this grades whatever
text is in a candidate's "text" field - which is always resolved source
text (Chunk.text), never a synthesized graph edge label, for every
retriever built in Phase 4 (Dense/BM25/Graph all return real chunk text).
"""
from __future__ import annotations

from enum import Enum

from adaptive_rag.retrieval import RerankerLike, get_reranker

# ponytail: provisional thresholds, not tuned against real labeled data -
# SS15 explicitly lists "Evidence grader Correct/Ambiguous/Incorrect score
# boundaries" as an unset tunable. Set from one real measurement of the
# actual cross-encoder's score range (ms-marco-MiniLM-L-6-v2 via FastEmbed,
# unbounded logits, not 0-1): a clearly-relevant passage scored ~8.5, a
# clearly-irrelevant one ~-11, a loosely-related one ~-1.1. Revisit once a
# labeled relevance dataset exists (Phase 9).
DEFAULT_CORRECT_THRESHOLD = 2.0
DEFAULT_AMBIGUOUS_THRESHOLD = -3.0


class Grade(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


def _grade_for_score(score: float, correct_threshold: float, ambiguous_threshold: float) -> Grade:
    if score >= correct_threshold:
        return Grade.CORRECT
    if score >= ambiguous_threshold:
        return Grade.AMBIGUOUS
    return Grade.INCORRECT


def grade_evidence(
    query: str,
    candidates: list[dict],
    grader: RerankerLike | None = None,
    correct_threshold: float = DEFAULT_CORRECT_THRESHOLD,
    ambiguous_threshold: float = DEFAULT_AMBIGUOUS_THRESHOLD,
) -> list[dict]:
    """FR11: grades each candidate's resolved source text, batched into one
    model call (same pattern as `retrieval.rerank_candidates`). Does not
    re-sort - candidates are expected to already be rank-ordered by
    whatever produced them (Phase 4's `rerank_candidates`), and
    `needs_recovery` below reads the first item as "the best evidence"."""
    if not candidates:
        return []
    model = grader or get_reranker()
    scores = model.rerank(query, [c["text"] for c in candidates])
    return [
        {**candidate, "grader_score": score, "grade": _grade_for_score(score, correct_threshold, ambiguous_threshold)}
        for candidate, score in zip(candidates, scores)
    ]


def needs_recovery(graded_evidence: list[dict]) -> bool:
    """FR12/FR13: the Recovery Planner triggers only when evidence is
    graded Ambiguous or Incorrect - Correct evidence passes straight
    through to the Context Builder (FR12, Phase 6) with no recovery
    attempt. Uses the top-ranked (first) graded candidate as the overall
    evidence-set grade, per `grade_evidence`'s ordering contract above.
    Empty evidence always needs recovery - there's nothing to hand
    forward."""
    if not graded_evidence:
        return True
    return graded_evidence[0]["grade"] != Grade.CORRECT
