import pytest

from evalom.comparison import compare_binary, compare_continuous, validate_comparability
from evalom.policy import ReleasePolicy


def test_independent_continuous_comparison_uses_candidate_minus_baseline():
    result = compare_continuous([1] * 50, [2] * 50, resamples=100, seed=7)

    assert result.status == "verified"
    assert result.difference == pytest.approx(1.0)
    assert result.difference_convention == "candidate - baseline"
    assert result.interpretation == "improvement"
    assert result.confidence_interval["lower"] <= 1.0 <= result.confidence_interval["upper"]


def test_metric_direction_changes_interpretation_not_effect_sign():
    result = compare_continuous([10] * 50, [8] * 50, direction="lower_is_better", resamples=100)

    assert result.difference == pytest.approx(-2.0)
    assert result.interpretation == "improvement"


def test_paired_comparison_requires_and_preserves_identifier_pairs():
    result = compare_continuous(
        [1, 2, 3],
        [2, 3, 4],
        paired=True,
        baseline_ids=["a", "b", "c"],
        candidate_ids=["c", "a", "b"],
        resamples=100,
        seed=2,
    )

    assert result.status == "verified"
    assert result.design == "paired"
    assert result.difference == pytest.approx(1.0)
    assert result.metadata["pair_count"] == 3

    unsupported = compare_continuous([1, 2], [2, 3], paired=True, resamples=20)
    assert unsupported.status == "insufficient_evidence"
    assert unsupported.evidence_status == "unsupported"


def test_equal_lengths_do_not_establish_pairing_and_unmatched_pairs_are_not_dropped():
    result = compare_continuous(
        [1, 2], [2, 3], paired=True,
        baseline_ids=["a", "b"], candidate_ids=["a", "c"], resamples=20,
    )
    assert result.status == "insufficient_evidence"
    assert result.confidence_interval == {"lower": None, "upper": None}


def test_binary_comparison_reuses_proportion_statistics():
    result = compare_binary(50, 100, 60, 100)

    assert result.status == "verified"
    assert result.baseline_value == pytest.approx(0.5)
    assert result.candidate_value == pytest.approx(0.6)
    assert result.difference == pytest.approx(0.1)
    assert result.metadata["baseline_statistics"]["method"] == "wilson_score"


def test_empty_and_insufficient_inputs_never_become_successful_comparisons():
    result = compare_binary(0, 0, 0, 10)
    assert result.status == "insufficient_evidence"
    assert result.interpretation == "uncertain"

    continuous = compare_continuous([], [1, 2], resamples=20)
    assert continuous.status == "insufficient_evidence"


def test_comparability_requires_matching_definitions_and_rejects_missing_metadata():
    compatible, errors, _ = validate_comparability(
        {"metric_definition": "accuracy", "dataset_version": "v1"},
        {"metric_definition": "accuracy", "dataset_version": "v1"},
    )
    assert compatible is True
    assert errors == []

    incompatible, errors, _ = validate_comparability(
        {"metric_definition": "accuracy"},
        {"metric_definition": "f1"},
    )
    assert incompatible is False
    assert "Incompatible metric_definition." in errors

    result = compare_continuous(
        [1, 2], [2, 3],
        baseline_metadata={"metric_definition": "accuracy"},
        candidate_metadata={},
        resamples=20,
    )
    assert result.status == "insufficient_evidence"
    assert result.evidence_status == "unsupported"


def test_comparison_serialization_preserves_reproducibility_metadata():
    result = compare_continuous([1, 2, 3], [2, 3, 4], paired=True,
                                baseline_ids=["a", "b", "c"],
                                candidate_ids=["a", "b", "c"],
                                resamples=50, seed=11)
    payload = result.to_dict()
    assert payload["metadata"]["resamples"] == 50
    assert payload["metadata"]["seed"] == 11
    assert payload["difference"] == pytest.approx(1.0)


def test_insufficient_comparison_does_not_authorize_release():
    comparison = compare_continuous([], [1], resamples=20)
    decision = ReleasePolicy(min_statistical_confidence=0).evaluate({
        "status": comparison.status,
        "risk_level": "LOW",
        "confidence_score": 100,
        "dataset_quality": 1,
        "evaluator_reliability": 1,
        "comparison": comparison.to_dict(),
    })

    assert comparison.status == "insufficient_evidence"
    assert decision["decision"] == "BLOCK"
