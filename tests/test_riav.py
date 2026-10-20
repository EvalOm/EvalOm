import pytest

from evalom.riav import (
    ConfusionMatrixMetrics,
    compare_assessment_to_reference,
    compute_confusion_matrix_metrics,
    evaluate_reference_outcomes,
)


def test_ground_truth_comparison_matches_when_in_sync():
    result = compare_assessment_to_reference("correct", "accepted")
    assert result.status == "match"
    assert result.match is True
    assert result.reference_label == "correct"


def test_ground_truth_comparison_flags_contradictory_evidence():
    result = compare_assessment_to_reference("defective", "accepted")
    assert result.status == "mismatch"
    assert result.match is False


def test_ground_truth_comparison_requires_reference_label():
    result = compare_assessment_to_reference(None, "accepted")
    assert result.status == "unverified"
    assert result.reference_label is None


def test_confusion_matrix_metrics_match_hand_calculated_fixture():
    metrics = compute_confusion_matrix_metrics(
        [
            *["defective"] * 10,
            *["correct"] * 20,
        ],
        [
            *["defective"] * 8,
            *["correct"] * 2,
            *["correct"] * 18,
            *["defective"] * 2,
        ],
    )

    assert metrics.tp == 8
    assert metrics.fp == 2
    assert metrics.tn == 18
    assert metrics.fn == 2
    assert metrics.total_eligible_cases == 30
    assert metrics.excluded_cases == 0

    payload = metrics.to_dict()
    assert payload["precision"] == pytest.approx(0.8)
    assert payload["recall"] == pytest.approx(0.8)
    assert payload["false_positive_rate"] == pytest.approx(0.1)
    assert payload["false_negative_rate"] == pytest.approx(0.2)
    assert payload["accuracy"] == pytest.approx(26 / 30)
    assert payload["error_rate"] == pytest.approx(4 / 30)
    assert payload["metric_denominators"]["precision"] == 10


def test_excluded_unverified_cases_do_not_affect_binary_metrics():
    metrics = compute_confusion_matrix_metrics(
        ["defective", "correct", "unverified", "correct"],
        ["defective", "correct", "unknown", "correct"],
    )

    assert metrics.tp == 1
    assert metrics.tn == 2
    assert metrics.excluded_cases == 1
    assert metrics.total_eligible_cases == 3


def test_empty_input_returns_empty_metrics():
    metrics = compute_confusion_matrix_metrics([], [])
    assert isinstance(metrics, ConfusionMatrixMetrics)
    assert metrics.total_eligible_cases == 0
    assert metrics.excluded_cases == 0
    assert metrics.to_dict()["precision"] is None
    assert metrics.to_dict()["accuracy"] is None


def test_invalid_decision_combinations_are_rejected():
    with pytest.raises(ValueError):
        compute_confusion_matrix_metrics(["correct"], ["maybe"])  # invalid state should be rejected


def test_integrated_riav_result_reuses_binary_statistics_and_reconciles_counts():
    result = evaluate_reference_outcomes(
        ["defective", "defective", "correct", "correct"],
        ["defective", "correct", "correct", "defective"],
        identifiers=["a", "b", "c", "d"],
    )

    assert result.status == "failed"
    assert result.evaluated_cases == 4
    assert result.insufficient_cases == 0
    assert result.confusion_matrix.to_dict()["tp"] == 1
    assert result.confusion_matrix.to_dict()["tn"] == 1
    assert result.confusion_matrix.to_dict()["specificity"] == 0.5
    assert result.confusion_matrix.to_dict()["f1"] == 0.5
    assert result.binary_statistics.successes == 2
    assert result.binary_statistics.failures == 2
    assert result.binary_statistics.status == "valid"
    assert result.to_dict()["binary_statistics"]["method"] == "wilson_score"


def test_integrated_riav_marks_missing_references_insufficient_and_not_pass():
    result = evaluate_reference_outcomes(
        ["correct", None],
        ["correct", "correct"],
        identifiers=["a", "b"],
    )

    assert result.status == "insufficient_evidence"
    assert result.insufficient_cases == 1
    assert result.binary_statistics.total_observations == 1
    assert result.binary_statistics.successes == 1


def test_integrated_riav_rejects_mismatched_or_duplicate_inputs():
    with pytest.raises(ValueError):
        evaluate_reference_outcomes(["correct"], ["correct", "defective"])
    with pytest.raises(ValueError):
        evaluate_reference_outcomes(["correct", "defective"], ["correct", "defective"], identifiers=["x", "x"])
