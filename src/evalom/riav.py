from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


_BINARY_REFERENCE_ALIASES = {
    "correct": "correct",
    "accept": "correct",
    "accepted": "correct",
    "pass": "correct",
    "passed": "correct",
    "true": "correct",
    "positive": "correct",
    "safe": "correct",
    "allow": "correct",
    "allowed": "correct",
    "good": "correct",
    "defective": "defective",
    "reject": "defective",
    "rejected": "defective",
    "fail": "defective",
    "failed": "defective",
    "false": "defective",
    "negative": "defective",
    "unsafe": "defective",
    "block": "defective",
    "blocked": "defective",
    "bad": "defective",
}

_UNVERIFIED_VALUES = {
    "",
    "none",
    "null",
    "missing",
    "unknown",
    "unverified",
    "not_applicable",
    "n/a",
    "na",
    "pending",
    "insufficient",
    "insufficient_evidence",
}


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return None


def _extract_candidate(value: Any, *keys: str) -> Any:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
    if hasattr(value, "__dict__"):
        for key in keys:
            if hasattr(value, key):
                return getattr(value, key)
    return None


def _normalize_binary_label(value: Any, *, target: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in _UNVERIFIED_VALUES:
            return None
        if target == "reference":
            if raw in _BINARY_REFERENCE_ALIASES:
                return _BINARY_REFERENCE_ALIASES[raw]
        else:
            decision_map = {
                "accepted": "correct",
                "accept": "correct",
                "accepted_correct": "correct",
                "correct": "correct",
                "allow": "correct",
                "pass": "correct",
                "passed": "correct",
                "flagged_defective": "defective",
                "defective": "defective",
                "flagged": "defective",
                "reject": "defective",
                "rejected": "defective",
                "block": "defective",
                "blocked": "defective",
                "fail": "defective",
                "failed": "defective",
            }
            if raw in decision_map:
                return decision_map[raw]
        return None
    if isinstance(value, bool):
        return "correct" if value else "defective"
    mapping = _coerce_mapping(value)
    if mapping is not None:
        for key in ("label", "status", "decision", "outcome", "value"):
            candidate = _extract_candidate(mapping, key)
            if candidate is not None:
                normalized = _normalize_binary_label(candidate, target=target)
                if normalized is not None:
                    return normalized
    return None


@dataclass(frozen=True)
class ReferenceOutcome:
    label: str | None
    provenance: str | None = None
    valid: bool = True
    reason: str | None = None

    @classmethod
    def from_value(cls, value: Any, *, provenance: str | None = None) -> "ReferenceOutcome":
        if value is None:
            return cls(label=None, provenance=provenance, valid=False, reason="Reference label is missing.")
        mapping = _coerce_mapping(value)
        if mapping is not None:
            provided = _extract_candidate(mapping, "label", "status", "decision", "outcome", "value")
            if provided is not None:
                normalized = _normalize_binary_label(provided, target="reference")
                if normalized is not None:
                    return cls(label=normalized, provenance=provenance or _extract_candidate(mapping, "provenance", "source"), valid=True)
                return cls(
                    label=None,
                    provenance=provenance or _extract_candidate(mapping, "provenance", "source"),
                    valid=False,
                    reason="Reference outcome is not a valid binary label.",
                )
        normalized = _normalize_binary_label(value, target="reference")
        if normalized is not None:
            return cls(label=normalized, provenance=provenance, valid=True)
        return cls(label=None, provenance=provenance, valid=False, reason="Reference label is unverified or invalid.")

    def as_binary_decision(self) -> str | None:
        if self.label is None:
            return None
        return self.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "provenance": self.provenance,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvalOmAssessment:
    label: str | None
    basis: str | None = None
    valid: bool = True
    reason: str | None = None

    @classmethod
    def from_value(cls, value: Any, *, basis: str | None = None) -> "EvalOmAssessment":
        if value is None:
            return cls(label=None, basis=basis, valid=False, reason="Assessment is missing.")
        mapping = _coerce_mapping(value)
        if mapping is not None:
            provided = _extract_candidate(mapping, "label", "status", "decision", "outcome", "value")
            if provided is not None:
                normalized = _normalize_binary_label(provided, target="assessment")
                if normalized is not None:
                    return cls(label=normalized, basis=basis or _extract_candidate(mapping, "basis", "source"), valid=True)
                return cls(
                    label=None,
                    basis=basis or _extract_candidate(mapping, "basis", "source"),
                    valid=False,
                    reason="Assessment is not a valid binary decision.",
                )
        normalized = _normalize_binary_label(value, target="assessment")
        if normalized is not None:
            return cls(label=normalized, basis=basis, valid=True)
        return cls(label=None, basis=basis, valid=False, reason="Assessment is unverified or invalid.")

    def as_binary_decision(self) -> str | None:
        if self.label is None:
            return None
        return self.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "basis": self.basis,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GroundTruthComparisonResult:
    status: str
    match: bool
    reference_label: str | None
    assessment_label: str | None
    reference_provenance: str | None = None
    assessment_basis: str | None = None
    reason: str | None = None
    evidence_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "match": self.match,
            "reference_label": self.reference_label,
            "assessment_label": self.assessment_label,
            "reference_provenance": self.reference_provenance,
            "assessment_basis": self.assessment_basis,
            "reason": self.reason,
            "evidence_status": self.evidence_status,
        }


def compare_assessment_to_reference(
    reference: Any,
    assessment: Any,
    *,
    agent_claim: Any | None = None,
    evidence: Any | None = None,
    provenance: str | None = None,
) -> GroundTruthComparisonResult:
    reference_outcome = ReferenceOutcome.from_value(reference, provenance=provenance)
    assessment_outcome = EvalOmAssessment.from_value(assessment, basis=evidence)

    if not reference_outcome.valid or reference_outcome.label is None:
        return GroundTruthComparisonResult(
            status="unverified",
            match=False,
            reference_label=reference_outcome.label,
            assessment_label=assessment_outcome.label,
            reference_provenance=reference_outcome.provenance,
            assessment_basis=assessment_outcome.basis,
            reason=reference_outcome.reason or "Reference outcome is insufficient for comparison.",
            evidence_status="reference_missing" if reference_outcome.label is None else "reference_invalid",
        )

    if not assessment_outcome.valid or assessment_outcome.label is None:
        return GroundTruthComparisonResult(
            status="unverified",
            match=False,
            reference_label=reference_outcome.label,
            assessment_label=assessment_outcome.label,
            reference_provenance=reference_outcome.provenance,
            assessment_basis=assessment_outcome.basis,
            reason=assessment_outcome.reason or "Assessment is insufficient for comparison.",
            evidence_status="assessment_missing" if assessment_outcome.label is None else "assessment_invalid",
        )

    if reference_outcome.label == assessment_outcome.label:
        return GroundTruthComparisonResult(
            status="match",
            match=True,
            reference_label=reference_outcome.label,
            assessment_label=assessment_outcome.label,
            reference_provenance=reference_outcome.provenance,
            assessment_basis=assessment_outcome.basis,
            reason="Reference outcome and EvalOm assessment agree.",
            evidence_status="matched",
        )

    return GroundTruthComparisonResult(
        status="mismatch",
        match=False,
        reference_label=reference_outcome.label,
        assessment_label=assessment_outcome.label,
        reference_provenance=reference_outcome.provenance,
        assessment_basis=assessment_outcome.basis,
        reason="Reference outcome disagrees with the EvalOm assessment.",
        evidence_status="disagreed",
    )


compare_ground_truth = compare_assessment_to_reference
assess_ground_truth = compare_assessment_to_reference


@dataclass
class ConfusionMatrixMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    excluded_cases: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def total_eligible_cases(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict[str, Any]:
        precision_denom = self.tp + self.fp
        recall_denom = self.tp + self.fn
        false_positive_denom = self.fp + self.tn
        false_negative_denom = self.fn + self.tp
        accuracy_denom = self.total_eligible_cases

        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total_eligible_cases": self.total_eligible_cases,
            "excluded_cases": self.excluded_cases,
            "exclusion_reasons": dict(self.exclusion_reasons),
            "precision": _safe_ratio(self.tp, precision_denom),
            "recall": _safe_ratio(self.tp, recall_denom),
            "false_positive_rate": _safe_ratio(self.fp, false_positive_denom),
            "false_negative_rate": _safe_ratio(self.fn, false_negative_denom),
            "accuracy": _safe_ratio(self.tp + self.tn, accuracy_denom),
            "error_rate": _safe_ratio(self.fp + self.fn, accuracy_denom),
            "metric_denominators": {
                "precision": precision_denom,
                "recall": recall_denom,
                "false_positive_rate": false_positive_denom,
                "false_negative_rate": false_negative_denom,
                "accuracy": accuracy_denom,
                "error_rate": accuracy_denom,
            },
        }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _record_exclusion(exclusions: dict[str, int], reason: str) -> None:
    exclusions[reason] = exclusions.get(reason, 0) + 1


def _is_unverified_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _UNVERIFIED_VALUES
    mapping = _coerce_mapping(value)
    if mapping is not None:
        for key in ("label", "status", "decision", "outcome", "value"):
            candidate = _extract_candidate(mapping, key)
            if candidate is not None and _is_unverified_value(candidate):
                return True
    return False


def compute_confusion_matrix_metrics(
    reference_labels: Iterable[Any],
    decisions: Iterable[Any],
) -> ConfusionMatrixMetrics:
    reference_values = list(reference_labels)
    decision_values = list(decisions)
    if len(reference_values) != len(decision_values):
        raise ValueError("reference_labels and decisions must be the same length.")

    metrics = ConfusionMatrixMetrics()
    for reference_value, decision_value in zip(reference_values, decision_values):
        reference_label = _normalize_binary_label(reference_value, target="reference")
        decision_label = _normalize_binary_label(decision_value, target="assessment")

        if reference_label is None:
            if _is_unverified_value(reference_value):
                metrics.excluded_cases += 1
                _record_exclusion(metrics.exclusion_reasons, "unverified_or_missing")
                continue
            raise ValueError(f"Invalid reference label {reference_value!r}.")

        if decision_label is None:
            if _is_unverified_value(decision_value):
                metrics.excluded_cases += 1
                _record_exclusion(metrics.exclusion_reasons, "unverified_or_missing")
                continue
            raise ValueError(f"Invalid decision label {decision_value!r}.")

        if reference_label not in {"correct", "defective"}:
            raise ValueError(f"Invalid reference label {reference_value!r}.")
        if decision_label not in {"correct", "defective"}:
            raise ValueError(f"Invalid decision label {decision_value!r}.")

        if reference_label == "defective" and decision_label == "defective":
            metrics.tp += 1
        elif reference_label == "correct" and decision_label == "defective":
            metrics.fp += 1
        elif reference_label == "correct" and decision_label == "correct":
            metrics.tn += 1
        elif reference_label == "defective" and decision_label == "correct":
            metrics.fn += 1

    return metrics


def calculate_confusion_matrix_metrics(reference_labels: Iterable[Any], decisions: Iterable[Any]) -> ConfusionMatrixMetrics:
    return compute_confusion_matrix_metrics(reference_labels, decisions)


class AssessmentStatus(str):
    """Compatibility wrapper for explicit assessment states."""

    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"
    INVALID = "invalid"


def normalize_reference_label(value: Any) -> str | None:
    return _normalize_binary_label(value, target="reference")


def normalize_assessment_decision(value: Any) -> str | None:
    return _normalize_binary_label(value, target="assessment")


__all__ = [
    "AssessmentStatus",
    "ReferenceOutcome",
    "EvalOmAssessment",
    "GroundTruthComparisonResult",
    "ConfusionMatrixMetrics",
    "compare_assessment_to_reference",
    "compare_ground_truth",
    "assess_ground_truth",
    "compute_confusion_matrix_metrics",
    "calculate_confusion_matrix_metrics",
    "normalize_reference_label",
    "normalize_assessment_decision",
]
