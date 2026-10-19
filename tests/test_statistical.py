import math

import pytest

from evalom.policy import ReleasePolicy
from evalom.statistical import (
    BinaryStatisticalResult,
    calculate_binary_statistics,
    calculate_wilson_interval,
    wilson_score_interval,
)


def test_wilson_interval_matches_independent_reference():
    result = wilson_score_interval(45, 60)
    assert result.status == "valid"
    assert result.observed_success_rate == 0.75
    assert result.failures == 15
    assert result.lower_bound == pytest.approx(0.627679, abs=1e-5)
    assert result.upper_bound == pytest.approx(0.842235, abs=1e-5)


def test_non_default_confidence_level_and_determinism():
    first = calculate_binary_statistics(8, 10, confidence_level=0.9)
    second = calculate_binary_statistics(8, 10, confidence_level=90)
    assert first == second
    assert calculate_wilson_interval(8, 10) == wilson_score_interval(8, 10)
    assert first.confidence_level == 0.9
    assert first.to_dict()["method"] == "wilson_score"


def test_zero_observations_are_explicitly_insufficient():
    result = wilson_score_interval(0, 0)
    assert result == BinaryStatisticalResult(0, 0, 0, None, 0.95, None, None, "insufficient_data")


@pytest.mark.parametrize("successes,total", [(0, 1), (1, 1), (0, 10), (10, 10), (1, 2)])
def test_boundary_and_small_samples_are_valid(successes, total):
    result = wilson_score_interval(successes, total)
    assert result.status == "valid"
    assert 0.0 <= result.lower_bound <= result.observed_success_rate <= result.upper_bound <= 1.0


def test_larger_sample_narrows_comparable_interval():
    small = wilson_score_interval(5, 10)
    large = wilson_score_interval(50, 100)
    assert large.upper_bound - large.lower_bound < small.upper_bound - small.lower_bound


def test_wider_confidence_level_is_not_narrower():
    narrow = wilson_score_interval(50, 100, confidence_level=0.90)
    wide = wilson_score_interval(50, 100, confidence_level=0.99)
    assert wide.lower_bound <= narrow.lower_bound
    assert wide.upper_bound >= narrow.upper_bound


@pytest.mark.parametrize(
    "successes,total,confidence",
    [(True, 1, 0.95), (1.0, 1, 0.95), (-1, 1, 0.95), (2, 1, 0.95), (0, -1, 0.95),
     (0, 1, 0), (0, 1, 1), (0, 1, math.nan), (0, 1, math.inf)],
)
def test_invalid_inputs_are_rejected(successes, total, confidence):
    with pytest.raises(ValueError):
        wilson_score_interval(successes, total, confidence_level=confidence)


def test_policy_blocks_insufficient_binary_statistics():
    decision = ReleasePolicy(min_statistical_confidence=0).evaluate({
        "status": "verified",
        "risk_level": "LOW",
        "confidence_score": 100,
        "dataset_quality": 1,
        "evaluator_reliability": 1,
        "binary_statistics": wilson_score_interval(0, 0).to_dict(),
    })
    assert decision["decision"] == "BLOCK"
    assert any("Binary statistical result is insufficient" in failure for failure in decision["failures"])
