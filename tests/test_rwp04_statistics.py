import pytest

from evalom.comparison import compare_binary, compare_continuous
from evalom.runner import EvaluationRunResult, EvaluationRunner


@pytest.mark.parametrize(
    ("baseline_successes", "candidate_successes"),
    [(0, 0), (20, 20), (20, 15), (15, 20)],
)
def test_binary_difference_interval_is_non_degenerate_and_ordered(
    baseline_successes: int, candidate_successes: int
):
    result = compare_binary(baseline_successes, 20, candidate_successes, 20)
    lower = result.confidence_interval["lower"]
    upper = result.confidence_interval["upper"]
    assert result.method == "newcombe_wilson_difference"
    assert lower < upper
    assert lower <= result.difference <= upper


def test_independent_continuous_comparison_uses_reproducible_bootstrap():
    first = compare_continuous([1, 2, 3], [2, 4, 6], resamples=500, seed=11)
    second = compare_continuous([1, 2, 3], [2, 4, 6], resamples=500, seed=11)
    assert first == second
    assert first.method == "independent_percentile_bootstrap"
    assert first.metadata["resamples"] == 500


def test_singleton_and_nonfinite_comparisons_are_not_overstated():
    singleton = compare_continuous([1], [2], resamples=20)
    assert singleton.status == "insufficient_evidence"
    assert singleton.interpretation == "uncertain"
    with pytest.raises(ValueError, match="finite"):
        compare_continuous([1, float("nan")], [2, 3])


def test_run_comparison_does_not_replace_missing_metrics_with_zero():
    baseline = EvaluationRunResult(
        "dataset", "v1", 1, metrics=[{"name": "accuracy", "score": 0.0}]
    )
    candidate = EvaluationRunResult(
        "dataset", "v1", 1, metrics=[{"name": "other", "score": 1.0}]
    )
    result = EvaluationRunner.compare_runs(baseline, candidate)
    assert result["accuracy"]["candidate"] is None
    assert result["accuracy"]["delta"] is None
    assert result["accuracy"]["comparison"]["status"] == "unavailable"


def test_run_comparison_rejects_different_dataset_identity():
    baseline = EvaluationRunResult(
        "baseline", "v1", 1, metrics=[{"name": "accuracy", "score": 0.5}]
    )
    candidate = EvaluationRunResult(
        "candidate", "v1", 1, metrics=[{"name": "accuracy", "score": 0.9}]
    )
    result = EvaluationRunner.compare_runs(baseline, candidate)
    assert result["accuracy"]["comparison"]["status"] == "insufficient_evidence"
    assert result["accuracy"]["comparison"]["evidence_status"] == "incompatible"


def test_run_comparison_uses_matching_sample_ids_as_pairs():
    baseline = EvaluationRunResult(
        "dataset", "v1", 2,
        metrics=[
            {"name": "accuracy", "sample_id": "a", "score": 0.5},
            {"name": "accuracy", "sample_id": "b", "score": 0.5},
        ],
    )
    candidate = EvaluationRunResult(
        "dataset", "v1", 2,
        metrics=[
            {"name": "accuracy", "sample_id": "b", "score": 0.6},
            {"name": "accuracy", "sample_id": "a", "score": 0.6},
        ],
    )
    result = EvaluationRunner.compare_runs(baseline, candidate)
    assert result["accuracy"]["comparison"]["design"] == "paired"
