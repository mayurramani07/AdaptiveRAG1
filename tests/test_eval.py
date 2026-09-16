from adaptive_rag.eval import (
    DEFAULT_GOLDEN_SET_PATH,
    evaluate_grader,
    evaluate_planner,
    evaluate_reranker,
    load_golden_set,
    run_offline_eval,
)


class ScriptedGrader:
    def __init__(self, score_by_text: dict[str, float], default: float = 0.0):
        self.score_by_text = score_by_text
        self.default = default

    def rerank(self, query, documents):
        return [self.score_by_text.get(doc, self.default) for doc in documents]


class ReversedOrderReranker:
    """Simulates the PRD's own literal regression example: a reranker
    returning documents in the WRONG order - the true-best document
    (always index 0 in this golden set) gets the LOWEST score, so it ends
    up ranked last instead of first."""

    def rerank(self, query, documents):
        return list(range(len(documents)))  # doc 0 -> lowest score, last doc -> highest


def test_golden_set_loads_and_has_all_three_sections():
    golden_set = load_golden_set(DEFAULT_GOLDEN_SET_PATH)
    assert golden_set["planner_examples"]
    assert golden_set["grader_examples"]
    assert golden_set["reranker_examples"]


# ---------------------------------------------------------------------------
# evaluate_planner (per-flag, not blended - SS6)
# ---------------------------------------------------------------------------


def test_evaluate_planner_returns_accuracy_per_flag_not_one_blended_number():
    examples = [
        {"query": "hello", "expected": {"dense": False, "bm25": False, "graph": False, "freshness": False, "apply_filters": False}},
    ]
    result = evaluate_planner(examples)
    assert set(result.keys()) == {"dense", "bm25", "graph", "freshness", "apply_filters"}
    assert all(v == 1.0 for v in result.values())


def test_evaluate_planner_detects_a_regression_in_a_single_flag():
    # A query the real planner correctly routes - if a future change to
    # the graph-triggering heuristic breaks JUST that flag, this must show
    # up as a graph-specific accuracy drop, not hidden behind a healthy
    # blended score.
    examples = [
        {
            "query": "How is Steve Jobs connected to Apple and Beats?",
            "expected": {"dense": True, "bm25": True, "graph": False, "freshness": False, "apply_filters": False},  # deliberately wrong "graph" expectation
        }
    ]
    result = evaluate_planner(examples)
    assert result["graph"] == 0.0  # caught
    assert result["dense"] == 1.0  # unaffected flags still correct


def test_evaluate_planner_empty_examples_returns_zero_not_crash():
    result = evaluate_planner([])
    assert all(v == 0.0 for v in result.values())


# ---------------------------------------------------------------------------
# evaluate_grader
# ---------------------------------------------------------------------------


def test_evaluate_grader_scores_against_scripted_grader():
    examples = [
        {"query": "q", "evidence_text": "relevant", "expected_grade": "correct"},
        {"query": "q", "evidence_text": "irrelevant", "expected_grade": "incorrect"},
    ]
    grader = ScriptedGrader({"relevant": 10.0, "irrelevant": -10.0})
    result = evaluate_grader(examples, grader=grader)
    assert result == {"accuracy": 1.0, "total": 2}


def test_evaluate_grader_real_bootstrap_golden_set_passes_at_100_percent():
    # Confirms the bundled golden set is accurately labeled against the
    # real FastEmbed cross-encoder - not just internally consistent.
    golden_set = load_golden_set()
    result = evaluate_grader(golden_set["grader_examples"])
    assert result["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# evaluate_reranker
# ---------------------------------------------------------------------------


def test_evaluate_reranker_real_bootstrap_golden_set_passes():
    golden_set = load_golden_set()
    result = evaluate_reranker(golden_set["reranker_examples"])
    assert result["accuracy"] == 1.0


def test_evaluate_reranker_catches_reversed_order_regression():
    golden_set = load_golden_set()
    result = evaluate_reranker(golden_set["reranker_examples"], reranker=ReversedOrderReranker())
    assert result["accuracy"] == 0.0  # every example's true-best doc is now ranked last


# ---------------------------------------------------------------------------
# run_offline_eval - the CI gate (PRD's literal acceptance criterion)
# ---------------------------------------------------------------------------


def test_run_offline_eval_passes_on_the_real_bundled_golden_set():
    golden_set = load_golden_set()
    report = run_offline_eval(golden_set)
    assert report.passed is True
    assert report.failures == []


def test_run_offline_eval_fails_when_reranker_regresses_to_random_order():
    """PRD SS7.8's literal acceptance criterion: 'A deliberately regressed
    component (e.g. reranker returning random order) fails the offline
    eval CI gate and blocks deployment.'"""
    golden_set = load_golden_set()
    report = run_offline_eval(golden_set, reranker=ReversedOrderReranker())

    assert report.passed is False
    assert any("reranker" in f for f in report.failures)


def test_run_offline_eval_fails_when_grader_regresses():
    golden_set = load_golden_set()
    always_wrong_grader = ScriptedGrader({}, default=-100.0)  # everything grades Incorrect
    report = run_offline_eval(golden_set, grader=always_wrong_grader)

    assert report.passed is False
    assert any("grader" in f for f in report.failures)


def test_run_offline_eval_reports_which_specific_flag_failed():
    golden_set = {
        "planner_examples": [
            {"query": "hello", "expected": {"dense": True, "bm25": True, "graph": False, "freshness": False, "apply_filters": False}}  # wrong on purpose
        ],
        "grader_examples": [],
        "reranker_examples": [],
    }
    report = run_offline_eval(golden_set, grader_threshold=0.0, reranker_threshold=0.0)
    assert report.passed is False
    assert any("planner[dense]" in f for f in report.failures)
