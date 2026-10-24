import math

import pytest

from evalom.policy import ReleasePolicy
from evalom.verification import EvaluationVerificationResult, VerificationEngine


def _valid_summary():
    return {
        "status": "verified",
        "risk_level": "LOW",
        "confidence_score": 90.0,
        "dataset_quality": 0.9,
        "evaluator_reliability": 0.9,
        "statistical_confidence": 0.9,
        "lower_confidence_bound": 0.9,
        "dataset_health": {"duplicate_ratio": 0.0, "distribution_shift": 0.0},
        "issues": [],
    }


@pytest.mark.parametrize(
    "field",
    [
        "confidence_score",
        "dataset_quality",
        "evaluator_reliability",
        "statistical_confidence",
        "lower_confidence_bound",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "invalid", None])
def test_invalid_policy_metrics_never_allow(field, value):
    summary = _valid_summary()
    summary[field] = value
    decision = ReleasePolicy().evaluate(summary)
    assert decision["decision"] == "BLOCK"
    assert any(field in failure for failure in decision["failures"])


def test_missing_distribution_shift_is_not_zero():
    summary = _valid_summary()
    del summary["dataset_health"]["distribution_shift"]
    decision = ReleasePolicy().evaluate(summary)
    assert decision["decision"] == "BLOCK"
    assert any("Distribution-shift evidence" in failure for failure in decision["failures"])


def test_measured_zero_remains_valid():
    decision = ReleasePolicy().evaluate(_valid_summary())
    assert decision["decision"] == "ALLOW"


def test_invalid_comparison_direction_blocks():
    summary = _valid_summary()
    summary["comparisons"] = {
        "accuracy": {
            "status": "verified",
            "difference": -0.5,
            "direction": "higher-is-better",
        }
    }
    policy = ReleasePolicy(required_comparisons=[{"metric": "accuracy", "max_regression": 0.1}])
    decision = policy.evaluate(summary)
    assert decision["decision"] == "BLOCK"
    assert any("invalid direction" in failure for failure in decision["failures"])


def test_issue_allowance_requires_exact_match():
    summary = _valid_summary()
    summary["issues"] = ["Dataset quality is too weak to support a high-confidence release decision."]
    exact = ReleasePolicy(allowed_issues=list(summary["issues"]), block_on_issues=True).evaluate(summary)
    partial = ReleasePolicy(allowed_issues=["Dataset quality"], block_on_issues=True).evaluate(summary)
    assert exact["decision"] == "ALLOW"
    assert partial["decision"] == "BLOCK"


def test_reliability_is_unavailable_without_independent_evidence():
    result = EvaluationVerificationResult(
        confidence_score=50,
        risk_level="HIGH",
        recommendation="insufficient",
        evaluator_reliability=None,
        dataset_quality=0.9,
        statistical_confidence=0.9,
    )
    summary = result.to_dict()
    summary["dataset_health"] = {"duplicate_ratio": 0.0, "distribution_shift": 0.0}
    decision = ReleasePolicy(min_confidence_score=0, min_dataset_quality=0, min_statistical_confidence=0).evaluate(summary)
    assert decision["decision"] == "BLOCK"
    assert any("evaluator_reliability" in failure for failure in decision["failures"])


def test_verification_without_dataset_does_not_fabricate_health():
    from evalom.runner import EvaluationRunResult

    result = EvaluationRunResult(
        dataset_name="report",
        dataset_version="v1",
        total_samples=1,
        metrics=[{"name": "accuracy", "score": 1.0, "passed": True}],
    )
    summary = VerificationEngine(minimum_sample_size=1).verify(result).to_dict()
    assert summary["dataset_quality"] is None
    assert summary["dataset_health"]["duplicate_ratio"] is None
    assert summary["dataset_health"]["distribution_shift"] is None
