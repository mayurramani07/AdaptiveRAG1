from adaptive_rag.grading import Grade, grade_evidence, needs_recovery


class ScriptedGrader:
    def __init__(self, score_by_text: dict[str, float]):
        self.score_by_text = score_by_text
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return [self.score_by_text[doc] for doc in documents]


def _hit(chunk_id, text):
    return {"id": chunk_id, "doc_id": "doc1", "chunk_id": chunk_id, "text": text}


# ---------------------------------------------------------------------------
# grade_evidence (FR11)
# ---------------------------------------------------------------------------


def test_grade_evidence_assigns_correct_ambiguous_incorrect():
    candidates = [_hit("a", "relevant"), _hit("b", "loosely related"), _hit("c", "irrelevant")]
    grader = ScriptedGrader({"relevant": 8.5, "loosely related": -1.1, "irrelevant": -11.0})
    graded = grade_evidence("query", candidates, grader=grader)

    assert graded[0]["grade"] == Grade.CORRECT
    assert graded[1]["grade"] == Grade.AMBIGUOUS
    assert graded[2]["grade"] == Grade.INCORRECT
    assert graded[0]["grader_score"] == 8.5


def test_grade_evidence_grades_resolved_source_text_not_a_graph_label():
    # Per SS3: whatever's in "text" is graded - this test exists to pin
    # that grade_evidence never looks at anything else (no separate
    # "label"/"edge" field influences the grade).
    candidate = {"id": "a", "doc_id": "d", "chunk_id": "a", "text": "real source passage", "label": "SOME_EDGE_LABEL"}
    grader = ScriptedGrader({"real source passage": 5.0})
    graded = grade_evidence("query", [candidate], grader=grader)
    assert grader.calls == [("query", ["real source passage"])]
    assert graded[0]["grade"] == Grade.CORRECT


def test_grade_evidence_batches_into_one_call():
    candidates = [_hit(str(i), f"text {i}") for i in range(5)]
    grader = ScriptedGrader({f"text {i}": 0.0 for i in range(5)})
    grade_evidence("query", candidates, grader=grader)
    assert len(grader.calls) == 1
    assert len(grader.calls[0][1]) == 5


def test_grade_evidence_empty_input_returns_empty():
    assert grade_evidence("query", [], grader=ScriptedGrader({})) == []


def test_grade_evidence_custom_thresholds():
    candidates = [_hit("a", "text")]
    grader = ScriptedGrader({"text": 1.0})
    graded = grade_evidence("query", candidates, grader=grader, correct_threshold=0.5, ambiguous_threshold=-1.0)
    assert graded[0]["grade"] == Grade.CORRECT


# ---------------------------------------------------------------------------
# needs_recovery (FR12/FR13)
# ---------------------------------------------------------------------------


def test_needs_recovery_false_when_top_evidence_correct():
    graded = [{**_hit("a", "x"), "grade": Grade.CORRECT}, {**_hit("b", "y"), "grade": Grade.INCORRECT}]
    assert needs_recovery(graded) is False


def test_needs_recovery_true_when_top_evidence_ambiguous():
    graded = [{**_hit("a", "x"), "grade": Grade.AMBIGUOUS}]
    assert needs_recovery(graded) is True


def test_needs_recovery_true_when_top_evidence_incorrect():
    graded = [{**_hit("a", "x"), "grade": Grade.INCORRECT}]
    assert needs_recovery(graded) is True


def test_needs_recovery_true_on_empty_evidence():
    assert needs_recovery([]) is True


def test_needs_recovery_only_looks_at_top_ranked_item():
    # A lower-ranked Correct item shouldn't rescue an overall-bad result -
    # the top (best-ranked) item is what determines the overall grade.
    graded = [{**_hit("a", "x"), "grade": Grade.INCORRECT}, {**_hit("b", "y"), "grade": Grade.CORRECT}]
    assert needs_recovery(graded) is True
