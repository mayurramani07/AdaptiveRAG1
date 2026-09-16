"""Offline evaluation suite (Phase 8, FR22-26): per-flag planner accuracy,
grader accuracy, and reranker accuracy against a hand-labeled golden set -
the CI deploy gate. Acceptance criterion (SS7.8): "a deliberately regressed
component (e.g. reranker returning random order) fails the offline eval CI
gate and blocks deployment" - `tests/test_eval.py` proves exactly this
scenario, not just that the suite runs.

`eval_data/golden_set.json` is an explicitly bootstrap dataset (~8 planner
examples, 5 grader examples, 2 reranker examples), hand-labeled from this
project's own documented planning/grading rules and verified 2026-09-16
against the real running code - NOT the production-scale dataset SS14 calls
for ("grows from flagged/reviewed production queries", Phase 9's job).
Thresholds below are bootstrap-appropriate (a tiny golden set means one
wrong answer swings accuracy a lot) and explicitly provisional, same
placeholder-threshold convention as every other tunable in this codebase
(SS15).

Planner and grader evaluation run against the REAL components by default
(rule-based planner, real FastEmbed cross-encoder) - both are free/local/
fast, so there's no reason to fake them for a meaningful regression check;
`grader`/`reranker` params exist for injecting a fake specifically to prove
the CI gate catches a regression (see the tests).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from adaptive_rag.grading import grade_evidence
from adaptive_rag.planning import plan_query, understand_query
from adaptive_rag.retrieval import RerankerLike, rerank_candidates

DEFAULT_GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent.parent / "eval_data" / "golden_set.json"

PLANNER_FLAGS = ("dense", "bm25", "graph", "freshness", "apply_filters")

# ponytail: bootstrap thresholds against a ~2-8 example golden set - one
# wrong answer swings accuracy by 12-50%. Revisit once a real production-
# scale labeled dataset exists (Phase 9, SS14).
DEFAULT_PLANNER_THRESHOLD = 0.7
DEFAULT_GRADER_THRESHOLD = 0.7
DEFAULT_RERANKER_THRESHOLD = 0.7


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_planner(examples: list[dict]) -> dict[str, float]:
    """SS6: per-flag accuracy, not one blended number - a regression in a
    single flag (e.g. `graph`) must be visible even if the other 4 flags
    are still perfect."""
    if not examples:
        return dict.fromkeys(PLANNER_FLAGS, 0.0)
    correct = dict.fromkeys(PLANNER_FLAGS, 0)
    for example in examples:
        understanding = understand_query(example["query"])
        plan = plan_query(understanding, example["query"])
        for flag in PLANNER_FLAGS:
            if getattr(plan, flag) == example["expected"][flag]:
                correct[flag] += 1
    total = len(examples)
    return {flag: correct[flag] / total for flag in PLANNER_FLAGS}


def evaluate_grader(examples: list[dict], grader: RerankerLike | None = None) -> dict:
    if not examples:
        return {"accuracy": 0.0, "total": 0}
    correct = 0
    for example in examples:
        candidate = {"id": "x", "doc_id": "d", "chunk_id": "x", "text": example["evidence_text"]}
        graded = grade_evidence(example["query"], [candidate], grader=grader)
        if graded[0]["grade"].value == example["expected_grade"]:
            correct += 1
    total = len(examples)
    return {"accuracy": correct / total, "total": total}


def evaluate_reranker(examples: list[dict], reranker: RerankerLike | None = None) -> dict:
    """The PRD's own literal acceptance example: "reranker returning
    random order" should fail this - checks whether the reranker actually
    puts the known-best document first."""
    if not examples:
        return {"accuracy": 0.0, "total": 0}
    correct = 0
    for example in examples:
        candidates = [{"id": str(i), "doc_id": "d", "chunk_id": str(i), "text": t} for i, t in enumerate(example["documents"])]
        reranked = rerank_candidates(example["query"], candidates, reranker=reranker, top_n_cap=len(candidates))
        if reranked[0]["chunk_id"] == str(example["expected_best_index"]):
            correct += 1
    total = len(examples)
    return {"accuracy": correct / total, "total": total}


@dataclass
class EvalReport:
    passed: bool
    planner_accuracy: dict[str, float] = field(default_factory=dict)
    grader_accuracy: dict = field(default_factory=dict)
    reranker_accuracy: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def run_offline_eval(
    golden_set: dict,
    grader: RerankerLike | None = None,
    reranker: RerankerLike | None = None,
    planner_threshold: float = DEFAULT_PLANNER_THRESHOLD,
    grader_threshold: float = DEFAULT_GRADER_THRESHOLD,
    reranker_threshold: float = DEFAULT_RERANKER_THRESHOLD,
) -> EvalReport:
    """The CI deploy gate (SS7.8 acceptance). `grader`/`reranker` let a
    caller (e.g. a canary test, or a future "evaluate this candidate model
    before promoting it" workflow) swap in an alternate component without
    touching the golden set or this function."""
    planner_acc = evaluate_planner(golden_set.get("planner_examples", []))
    grader_acc = evaluate_grader(golden_set.get("grader_examples", []), grader=grader)
    reranker_acc = evaluate_reranker(golden_set.get("reranker_examples", []), reranker=reranker)

    failures = []
    for flag, acc in planner_acc.items():
        if acc < planner_threshold:
            failures.append(f"planner[{flag}] accuracy {acc:.2f} < {planner_threshold}")
    if grader_acc["accuracy"] < grader_threshold:
        failures.append(f"grader accuracy {grader_acc['accuracy']:.2f} < {grader_threshold}")
    if reranker_acc["accuracy"] < reranker_threshold:
        failures.append(f"reranker accuracy {reranker_acc['accuracy']:.2f} < {reranker_threshold}")

    return EvalReport(passed=not failures, planner_accuracy=planner_acc, grader_accuracy=grader_acc, reranker_accuracy=reranker_acc, failures=failures)


def main() -> int:
    golden_set = load_golden_set()
    report = run_offline_eval(golden_set)
    print(
        json.dumps(
            {
                "passed": report.passed,
                "planner_accuracy": report.planner_accuracy,
                "grader_accuracy": report.grader_accuracy,
                "reranker_accuracy": report.reranker_accuracy,
                "failures": report.failures,
            },
            indent=2,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
