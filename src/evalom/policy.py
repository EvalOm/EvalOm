from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_POLICY_FIELDS = {
    "policy_id",
    "policy_version",
    "max_risk_level",
    "min_confidence_score",
    "min_dataset_quality",
    "min_evaluator_reliability",
    "min_statistical_confidence",
    "max_duplicate_ratio",
    "max_distribution_shift",
    "confidence_level",
    "allowed_issues",
    "block_on_issues",
    "required_checks",
    "optional_checks",
    "required_evidence",
    "required_comparisons",
    "required_task_profile",
}
_CHECK_STATUSES = {"passed", "failed", "not_run", "indeterminate"}
_POLICY_VERSION = "1.0"
_DIRECTIONS = {"higher_is_better", "lower_is_better", "none"}


def _normalize_confidence_level(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence_level must be a finite number between 0 and 1, or a percentage between 1 and 100.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("confidence_level must be finite.")
    if 0.0 < numeric < 1.0:
        return numeric
    if 1.0 < numeric <= 100.0:
        return numeric / 100.0
    raise ValueError("confidence_level must be in the open interval (0, 1) or a percentage between 1 and 100.")


@dataclass
class ReleasePolicy:
    """Policy used to decide whether verification output is safe enough to ship."""

    policy_id: str = "default"
    policy_version: str = _POLICY_VERSION
    max_risk_level: str = "MEDIUM"
    min_confidence_score: float = 80.0
    min_dataset_quality: float = 0.75
    min_evaluator_reliability: float = 0.80
    min_statistical_confidence: float = 0.75
    max_duplicate_ratio: float = 0.05
    max_distribution_shift: float = 0.35
    confidence_level: float = 0.95
    allowed_issues: list[str] = field(default_factory=list)
    block_on_issues: bool = False
    required_checks: list[str] = field(default_factory=list)
    optional_checks: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    required_comparisons: list[dict[str, Any]] = field(default_factory=list)
    required_task_profile: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty string.")
        if self.policy_version != _POLICY_VERSION:
            raise ValueError(f"Unsupported policy_version: {self.policy_version!r}.")
        for name in ("required_checks", "optional_checks", "required_evidence"):
            values = getattr(self, name)
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must be a list of non-empty strings.")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates.")
        if set(self.required_checks) & set(self.optional_checks):
            raise ValueError("A check cannot be both required and optional.")
        if not isinstance(self.required_comparisons, list) or any(not isinstance(item, dict) for item in self.required_comparisons):
            raise ValueError("required_comparisons must be a list of dictionaries.")
        comparison_ids: list[str] = []
        for item in self.required_comparisons:
            metric = item.get("metric")
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("Each required comparison needs a metric.")
            comparison_ids.append(metric)
            for field_name in ("max_regression", "min_improvement"):
                if field_name in item and (
                    isinstance(item[field_name], bool)
                    or not isinstance(item[field_name], (int, float))
                    or not math.isfinite(float(item[field_name]))
                    or float(item[field_name]) < 0
                ):
                    raise ValueError(f"{field_name} must be a finite non-negative number.")
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("required_comparisons must not contain duplicate metrics.")
        if self.required_task_profile is not None and not isinstance(self.required_task_profile, dict):
            raise ValueError("required_task_profile must be a dictionary.")
        if not isinstance(self.max_risk_level, str):
            raise ValueError("max_risk_level must be LOW, MEDIUM, or HIGH.")
        self.max_risk_level = self.max_risk_level.upper()
        if self.max_risk_level not in _RISK_ORDER:
            raise ValueError("max_risk_level must be LOW, MEDIUM, or HIGH.")

        for name in (
            "min_confidence_score",
            "min_dataset_quality",
            "min_evaluator_reliability",
            "max_duplicate_ratio",
            "max_distribution_shift",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number in its allowed range.")
            numeric_value = float(value)
            upper_bound = 100.0 if name == "min_confidence_score" else 1.0
            if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= upper_bound:
                raise ValueError(
                    f"{name} must be between 0 and {upper_bound:g}."
                )
            setattr(self, name, numeric_value)

        for name in ("min_statistical_confidence",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(f"{name} must be a finite, non-negative number.")
            setattr(self, name, numeric_value)

        self.confidence_level = _normalize_confidence_level(self.confidence_level)

        if not isinstance(self.allowed_issues, list) or any(
            not isinstance(issue, str) or not issue.strip()
            for issue in self.allowed_issues
        ):
            raise ValueError("allowed_issues must be a list of non-empty strings.")
        if not isinstance(self.block_on_issues, bool):
            raise ValueError("block_on_issues must true or false.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ReleasePolicy":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("Release policy must be a dictionary.")
        unknown_fields = set(payload) - _POLICY_FIELDS
        if unknown_fields:
            names = ", ".join(sorted(str(name) for name in unknown_fields))
            raise ValueError(f"Unknown release policy field(s): {names}.")

        allowed_issues = payload.get("allowed_issues", [])
        if not isinstance(allowed_issues, list):
            raise ValueError("allowed_issues must be a list of non-empty strings.")

        return cls(
            policy_id=payload.get("policy_id", "default"),
            policy_version=payload.get("policy_version", _POLICY_VERSION),
            max_risk_level=payload.get("max_risk_level", "MEDIUM"),
            min_confidence_score=payload.get("min_confidence_score", 80.0),
            min_dataset_quality=payload.get("min_dataset_quality", 0.75),
            min_evaluator_reliability=payload.get("min_evaluator_reliability", 0.80),
            min_statistical_confidence=payload.get("min_statistical_confidence", 0.75),
            max_duplicate_ratio=payload.get("max_duplicate_ratio", 0.05),
            max_distribution_shift=payload.get("max_distribution_shift", 0.35),
            confidence_level=payload.get("confidence_level", 0.95),
            allowed_issues=allowed_issues,
            block_on_issues=payload.get("block_on_issues", False),
            required_checks=payload.get("required_checks", []),
            optional_checks=payload.get("optional_checks", []),
            required_evidence=payload.get("required_evidence", []),
            required_comparisons=payload.get("required_comparisons", []),
            required_task_profile=payload.get("required_task_profile"),
        )

    def evaluate(self, verification: dict[str, Any] | Any) -> dict[str, Any]:
        if hasattr(verification, "to_dict"):
            summary = verification.to_dict()
        elif isinstance(verification, dict):
            summary = verification
        else:
            raise TypeError("Verification must be a dict-like result or an object with to_dict().")

        failures: list[str] = []

        def numeric(name: str, value: Any, *, required: bool = True) -> float | None:
            if value is None:
                if required:
                    failures.append(f"Required metric {name!r} is missing.")
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                failures.append(f"Metric {name!r} is invalid; expected a finite number.")
                return None
            number = float(value)
            if not math.isfinite(number):
                failures.append(f"Metric {name!r} is invalid; expected a finite number.")
                return None
            return number

        risk_level = str(summary.get("risk_level") or "HIGH").upper()
        confidence_score = numeric("confidence_score", summary.get("confidence_score"))
        dataset_health = summary.get("dataset_health")
        if dataset_health is not None and not isinstance(dataset_health, dict):
            failures.append("dataset_health is invalid; expected a dictionary.")
            dataset_health = {}
        dataset_health = dataset_health or {}
        dataset_quality = numeric(
            "dataset_quality",
            summary.get("dataset_quality") if "dataset_quality" in summary else dataset_health.get("quality_score"),
        )
        evaluator_reliability = numeric("evaluator_reliability", summary.get("evaluator_reliability"))
        statistical_confidence = numeric("statistical_confidence", summary.get("statistical_confidence"))
        interval = summary.get("confidence_interval")
        if interval is not None and not isinstance(interval, dict):
            failures.append("confidence_interval is invalid; expected a dictionary.")
            interval = {}
        lower_bound = numeric(
            "lower_confidence_bound",
            summary["lower_confidence_bound"] if "lower_confidence_bound" in summary else (interval or {}).get("lower"),
            required="lower_confidence_bound" in summary or bool(interval and "lower" in interval),
        )
        if lower_bound is None:
            lower_bound = statistical_confidence
        duplicate_ratio = numeric(
            "duplicate_ratio",
            dataset_health.get("duplicate_ratio"),
            required="duplicate_ratio" in dataset_health,
        )
        distribution_shift = numeric(
            "distribution_shift",
            dataset_health.get("distribution_shift"),
            required="distribution_shift" in dataset_health,
        )
        if duplicate_ratio is None:
            failures.append("Duplicate ratio evidence is unavailable.")
        if distribution_shift is None:
            failures.append("Distribution-shift evidence is unavailable.")
        issues = [str(issue) for issue in summary.get("issues", [])]
        binary_statistics = summary.get("binary_statistics")

        passed_requirements: list[str] = []
        failed_requirements: list[str] = []
        not_run_requirements: list[str] = []
        indeterminate_requirements: list[str] = []
        checks = summary.get("checks", {})
        if isinstance(checks, list):
            check_map: dict[str, dict[str, Any]] = {}
            for item in checks:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise ValueError("checks list entries must contain a string name.")
                name = item["name"]
                if name in check_map:
                    raise ValueError(f"Duplicate check result: {name}.")
                check_map[name] = item
            checks = check_map
        if not isinstance(checks, dict):
            raise ValueError("checks must be a dictionary or list.")
        for check_name in self.required_checks:
            result = checks.get(check_name)
            status = result.get("status") if isinstance(result, dict) else "not_run"
            if status not in _CHECK_STATUSES:
                raise ValueError(f"Unsupported status for required check {check_name!r}: {status!r}.")
            if status == "passed":
                passed_requirements.append(check_name)
            elif status == "failed":
                failed_requirements.append(check_name)
                failures.append(f"Required check {check_name!r} failed.")
            elif status == "not_run":
                not_run_requirements.append(check_name)
                failures.append(f"Required check {check_name!r} was not run.")
            else:
                indeterminate_requirements.append(check_name)
                failures.append(f"Required check {check_name!r} is indeterminate.")
        evidence = summary.get("evidence", {})
        if isinstance(evidence, list):
            normalized_evidence: dict[str, dict[str, Any]] = {}
            for item in evidence:
                if not isinstance(item, dict) or not isinstance(item.get("evidence_type"), str):
                    raise ValueError("evidence list entries must contain an evidence_type.")
                evidence_type = item["evidence_type"]
                if evidence_type in normalized_evidence:
                    raise ValueError(f"Duplicate evidence result: {evidence_type}.")
                normalized_evidence[evidence_type] = item
            evidence = normalized_evidence
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be a dictionary or list.")
        for evidence_type in self.required_evidence:
            item = evidence.get(evidence_type)
            usable = isinstance(item, dict) and item.get("available") is True and item.get("sufficient") is True
            if not usable:
                failures.append(f"Required evidence {evidence_type!r} is missing or insufficient.")
        comparisons = summary.get("comparisons", {})
        if isinstance(comparisons, list):
            normalized_comparisons: dict[str, dict[str, Any]] = {}
            for item in comparisons:
                if not isinstance(item, dict) or not isinstance(item.get("metric_name", item.get("metric")), str):
                    raise ValueError("comparison list entries must contain a metric name.")
                metric_name = item.get("metric_name", item.get("metric"))
                if metric_name in normalized_comparisons:
                    raise ValueError(f"Duplicate comparison result: {metric_name}.")
                normalized_comparisons[metric_name] = item
            comparisons = normalized_comparisons
        if not isinstance(comparisons, dict):
            raise ValueError("comparisons must be a dictionary or list.")
        for requirement in self.required_comparisons:
            metric = requirement["metric"]
            comparison = comparisons.get(metric)
            if not isinstance(comparison, dict):
                failures.append(f"Required comparison {metric!r} is missing.")
                continue
            if comparison.get("status") in {"insufficient_evidence", "unsupported", "invalid"}:
                failures.append(f"Required comparison {metric!r} is unresolved.")
                continue
            difference = comparison.get("difference")
            if isinstance(difference, bool) or not isinstance(difference, (int, float)) or not math.isfinite(float(difference)):
                failures.append(f"Required comparison {metric!r} has no valid difference.")
                continue
            direction = comparison.get("direction", "higher_is_better")
            if direction not in _DIRECTIONS:
                failures.append(f"Required comparison {metric!r} has an invalid direction.")
                continue
            regression = -float(difference) if direction == "higher_is_better" else float(difference)
            if requirement.get("max_regression") is not None and regression > float(requirement["max_regression"]):
                failures.append(f"Comparison {metric!r} exceeds the permitted regression threshold.")
            if requirement.get("min_improvement") is not None:
                improvement = float(difference) if direction == "higher_is_better" else -float(difference)
                if improvement < float(requirement["min_improvement"]):
                    failures.append(f"Comparison {metric!r} does not meet the required improvement threshold.")
        profile_result = summary.get("task_profile_result")
        if self.required_task_profile is not None:
            if not isinstance(profile_result, dict):
                failures.append("Required task profile result is missing.")
            elif profile_result.get("status") != "verified":
                failures.append("Required task profile requirements are not verified.")
        if isinstance(binary_statistics, dict) and binary_statistics.get("status") != "valid":
            failures.append("Binary statistical result is insufficient; release is blocked.")
        if isinstance(binary_statistics, dict) and binary_statistics.get("status") == "valid":
            try:
                lower_bound = float(binary_statistics["lower_bound"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("binary_statistics must contain a numeric lower_bound.") from exc
        assurance_status = summary.get("status")
        if assurance_status is not None and assurance_status != "verified":
            failures.append(
                f"Assurance status {assurance_status!r} is not verified; release is blocked."
            )
        if risk_level not in _RISK_ORDER:
            failures.append(f"Unknown verification risk level {risk_level!r}; release is blocked.")
        if _RISK_ORDER.get(risk_level, 2) > _RISK_ORDER.get(self.max_risk_level.upper(), 2):
            failures.append(f"Risk level {risk_level} exceeds the configured max risk {self.max_risk_level.upper()}.")
        if confidence_score is not None and confidence_score < self.min_confidence_score:
            failures.append(
                f"Confidence score {confidence_score:.2f} is below the required minimum of {self.min_confidence_score:.2f}."
            )
        if dataset_quality is not None and dataset_quality < self.min_dataset_quality:
            failures.append(
                f"Dataset quality {dataset_quality:.4f} is below the required minimum of {self.min_dataset_quality:.4f}."
            )
        if evaluator_reliability is not None and evaluator_reliability < self.min_evaluator_reliability:
            failures.append(
                f"Evaluator reliability {evaluator_reliability:.4f} is below the required minimum of {self.min_evaluator_reliability:.4f}."
            )
        if lower_bound is not None and lower_bound < self.min_statistical_confidence:
            failures.append(
                f"Lower confidence bound {lower_bound:.4f} is below the required minimum of {self.min_statistical_confidence:.4f}."
            )
        if duplicate_ratio is not None and duplicate_ratio > self.max_duplicate_ratio:
            failures.append(
                f"Duplicate ratio {duplicate_ratio:.4f} exceeds the maximum allowed {self.max_duplicate_ratio:.4f}."
            )
        if distribution_shift is not None and distribution_shift > self.max_distribution_shift:
            failures.append(
                f"Distribution shift {distribution_shift:.4f} exceeds the maximum allowed {self.max_distribution_shift:.4f}."
            )

        allowed = {str(item).strip().lower() for item in self.allowed_issues}
        warnings: list[str] = []
        for issue in issues:
            normalized = issue.strip().lower()
            if normalized in allowed:
                continue
            warnings.append(issue)
        if self.block_on_issues:
            failures.extend(warnings)

        return {
            "passed": not failures,
            "decision": "ALLOW" if not failures else "BLOCK",
            "risk_level": risk_level,
            "max_risk_level": self.max_risk_level.upper(),
            "confidence_score": confidence_score,
            "dataset_quality": dataset_quality,
            "evaluator_reliability": evaluator_reliability,
            "statistical_confidence": statistical_confidence,
            "lower_confidence_bound": lower_bound,
            "confidence_level": self.confidence_level,
            "minimum_statistical_confidence": self.min_statistical_confidence,
            "duplicate_ratio": duplicate_ratio,
            "distribution_shift": distribution_shift,
            "failures": failures,
            "warnings": [] if self.block_on_issues else warnings,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "passed_requirements": passed_requirements,
            "failed_requirements": failed_requirements,
            "not_run_requirements": not_run_requirements,
            "indeterminate_requirements": indeterminate_requirements,
            "policy": self.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "max_risk_level": self.max_risk_level.upper(),
            "min_confidence_score": float(self.min_confidence_score),
            "min_dataset_quality": float(self.min_dataset_quality),
            "min_evaluator_reliability": float(self.min_evaluator_reliability),
            "min_statistical_confidence": float(self.min_statistical_confidence),
            "max_duplicate_ratio": float(self.max_duplicate_ratio),
            "max_distribution_shift": float(self.max_distribution_shift),
            "confidence_level": float(self.confidence_level),
            "allowed_issues": list(self.allowed_issues),
            "block_on_issues": bool(self.block_on_issues),
            "required_checks": list(self.required_checks),
            "optional_checks": list(self.optional_checks),
            "required_evidence": list(self.required_evidence),
            "required_comparisons": [dict(item) for item in self.required_comparisons],
            "required_task_profile": dict(self.required_task_profile) if self.required_task_profile is not None else None,
        }


def evaluate_verification_policy(policy: ReleasePolicy | dict[str, Any], verification: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(policy, dict):
        policy = ReleasePolicy.from_dict(policy)
    return policy.evaluate(verification)


__all__ = ["ReleasePolicy", "evaluate_verification_policy"]
