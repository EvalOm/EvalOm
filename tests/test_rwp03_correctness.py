import json
import math
from pathlib import Path

import pytest

from evalom.dataset import EvaluationDataset
from evalom.evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    MRREvaluator,
    NDCGEvaluator,
    NumericToleranceEvaluator,
    RegexEvaluator,
    RetrievalRecallEvaluator,
    build_evaluator,
)
from evalom.runner import EvaluationRunner


def test_jsonl_handles_bom_and_unicode_line_separator(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    rows = [
        {"id": "one", "input": "question", "expected_answer": "line\u2028separator"},
        {"id": "two", "input": "second", "expected_answer": ""},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8-sig")

    dataset = EvaluationDataset.from_jsonl(path)

    assert len(dataset) == 2
    assert dataset[0].expected_answer == "line\u2028separator"


def test_dataset_rejects_invalid_metadata_and_duplicate_ids(tmp_path: Path):
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        '{"id":"one","input":"q","metadata":null}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata.*object"):
        EvaluationDataset.from_jsonl(invalid)

    duplicate = EvaluationDataset([
        {"id": "same", "input": "one"},
        {"id": "same", "input": "two"},
    ])
    with pytest.raises(ValueError, match="Duplicate sample identifier"):
        duplicate.validate()


def test_csv_rejects_missing_and_extra_fields(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    missing.write_text("input,expected_answer\nq,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        EvaluationDataset.from_csv(missing)

    extra = tmp_path / "extra.csv"
    extra.write_text("id,input\n1,q,unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more fields"):
        EvaluationDataset.from_csv(extra)

    short = tmp_path / "short.csv"
    short.write_text("id,input,expected_answer\n1,q\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fewer fields"):
        EvaluationDataset.from_csv(short)


@pytest.mark.parametrize(
    ("evaluator", "sample", "output"),
    [
        (ExactMatchEvaluator(), {"expected_answer": 0}, {"answer": 0}),
        (ExactMatchEvaluator(), {"expected_answer": False}, {"answer": False}),
        (ContainsEvaluator(["0"]), {}, {"answer": 0}),
    ],
)
def test_evaluators_preserve_falsey_values(evaluator, sample, output):
    assert evaluator.evaluate(sample, output)["passed"] is True


def test_missing_answer_and_ground_truth_are_not_successes():
    exact = ExactMatchEvaluator()
    assert exact.evaluate({"expected_answer": ""}, {})["status"] == "insufficient_evidence"
    assert exact.evaluate({}, {"answer": ""})["passed"] is False
    assert RetrievalRecallEvaluator().evaluate({}, {"contexts": ["x"]})["passed"] is False


def test_evaluator_rejects_scalar_contexts_and_invalid_regex():
    with pytest.raises(ValueError, match="sequence"):
        RetrievalRecallEvaluator().evaluate(
            {"expected_context": "doc1"}, {"contexts": ["doc1"]}
        )
    with pytest.raises(ValueError, match="non-empty"):
        RegexEvaluator("")
    with pytest.raises(ValueError, match="Invalid regex"):
        RegexEvaluator("[")


def test_numeric_tolerance_has_binary_failure_and_finite_validation():
    evaluator = NumericToleranceEvaluator(0.1)
    result = evaluator.evaluate({"expected_answer": 1}, {"answer": 1.5})
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert evaluator.evaluate({"expected_answer": 1}, {"answer": float("nan")})["status"] == "invalid"
    with pytest.raises(ValueError, match="finite"):
        NumericToleranceEvaluator(float("inf"))


def test_retrieval_metrics_deduplicate_and_stay_bounded():
    sample = {"expected_context": ["a", "b", "c"]}
    recall = RetrievalRecallEvaluator(k=3).evaluate(
        sample, {"contexts": ["a", "a", "x", "b"]}
    )
    assert recall["score"] == pytest.approx(2 / 3)
    assert 0 <= recall["score"] <= 1

    mrr = MRREvaluator().evaluate(
        {"expected_context": ["a"]}, {"contexts": ["x", "a", "a"]}
    )
    assert mrr["score"] == pytest.approx(0.5)

    ndcg = NDCGEvaluator(k=3).evaluate(
        sample, {"contexts": ["x", "a", "b"]}
    )
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (
        1 + 1 / math.log2(3) + 1 / math.log2(4)
    )
    assert ndcg["score"] == pytest.approx(expected)
    assert 0 <= ndcg["score"] <= 1


def test_runner_preserves_details_and_alias_identity():
    dataset = EvaluationDataset([
        {"id": "1", "input": "q", "expected_context": ["doc"]},
    ])
    result = EvaluationRunner(
        dataset,
        [build_evaluator({"name": "recall", "type": "recall", "k": 1})],
    ).run(lambda _: {"answer": "ok", "contexts": ["doc"]})

    assert result.metrics[0]["name"] == "recall"
    assert result.metrics[0]["hits"] == 1
    assert result.sample_results[0]["result"][0]["retrieved"] == ["doc"]
