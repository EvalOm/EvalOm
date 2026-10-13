from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_POLICY_FIELDS = {
    "max_risk_level",
    "min_confidence_score",
    "min_dataset_quality",
    "min_evaluator_reliability",
    "min_statistical_confidence",
    "max_duplicate_ratio",
    "max_distribution_shift",
    "confidence_level",
    "allowed_issues",
}


def _normalize_confidence_level(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence_level must be a finite number between 0 and 1, or a percentage between 1 and 100.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("confidence_level must be finite.")
    if 0.0 < numeric < 1.0:
        return numeric
    if 1.0 <= numeric <= 100.0:
        return numeric / 100.0
    raise ValueError("confidence_level must be in the open interval (0, 1) or a percentage between 1 and 100.")


@dataclass
class ReleasePolicy:
    """Policy used to decide whether verification output is safe enough to ship."""

    max_risk_level: str = "MEDIUM"
    min_confidence_score: float = 80.0
    min_dataset_quality: float = 0.75
    min_evaluator_reliability: float = 0.80
    min_statistical_confidence: float = 0.75
    max_duplicate_ratio: float = 0.05
    max_distribution_shift: float = 0.35
    confidence_level: float = 0.95
    allowed_issues: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
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
            max_risk_level=payload.get("max_risk_level", "MEDIUM"),
            min_confidence_score=payload.get("min_confidence_score", 80.0),
            min_dataset_quality=payload.get("min_dataset_quality", 0.75),
            min_evaluator_reliability=payload.get("min_evaluator_reliability", 0.80),
            min_statistical_confidence=payload.get("min_statistical_confidence", 0.75),
            max_duplicate_ratio=payload.get("max_duplicate_ratio", 0.05),
            max_distribution_shift=payload.get("max_distribution_shift", 0.35),
            confidence_level=payload.get("confidence_level", 0.95),
            allowed_issues=allowed_issues,
        )

    def evaluate(self, verification: dict[str, Any] | Any) -> dict[str, Any]:
        if hasattr(verification, "to_dict"):
            summary = verification.to_dict()
        elif isinstance(verification, dict):
            summary = verification
        else:
            raise TypeError("Verification must be a dict-like result or an object with to_dict().")

        risk_level = str(summary.get("risk_level") or "HIGH").upper()
        confidence_score = float(summary.get("confidence_score", 0.0))
        dataset_quality = float(summary.get("dataset_quality", summary.get("dataset_health", {}).get("quality_score", 0.0)))
        evaluator_reliability = float(summary.get("evaluator_reliability", 0.0))
        statistical_confidence = float(summary.get("statistical_confidence", 0.0))
        lower_bound = float(summary.get("lower_confidence_bound", summary.get("confidence_interval", {}).get("lower", statistical_confidence)))
        dataset_health = summary.get("dataset_health") if isinstance(summary.get("dataset_health"), dict) else {}
        duplicate_ratio = float(dataset_health.get("duplicate_ratio", 0.0))
        distribution_shift = float(dataset_health.get("distribution_shift", 0.0))
        issues = [str(issue) for issue in summary.get("issues", [])]

        failures: list[str] = []
        if risk_level not in _RISK_ORDER:
            failures.append(f"Unknown verification risk level {risk_level!r}; release is blocked.")
        if _RISK_ORDER.get(risk_level, 2) > _RISK_ORDER.get(self.max_risk_level.upper(), 2):
            failures.append(f"Risk level {risk_level} exceeds the configured max risk {self.max_risk_level.upper()}.")
        if confidence_score < self.min_confidence_score:
            failures.append(
                f"Confidence score {confidence_score:.2f} is below the required minimum of {self.min_confidence_score:.2f}."
            )
        if dataset_quality < self.min_dataset_quality:
            failures.append(
                f"Dataset quality {dataset_quality:.4f} is below the required minimum of {self.min_dataset_quality:.4f}."
            )
        if evaluator_reliability < self.min_evaluator_reliability:
            failures.append(
                f"Evaluator reliability {evaluator_reliability:.4f} is below the required minimum of {self.min_evaluator_reliability:.4f}."
            )
        if lower_bound < self.min_statistical_confidence:
            failures.append(
                f"Lower confidence bound {lower_bound:.4f} is below the required minimum of {self.min_statistical_confidence:.4f}."
            )
        if duplicate_ratio > self.max_duplicate_ratio:
            failures.append(
                f"Duplicate ratio {duplicate_ratio:.4f} exceeds the maximum allowed {self.max_duplicate_ratio:.4f}."
            )
        if distribution_shift > self.max_distribution_shift:
            failures.append(
                f"Distribution shift {distribution_shift:.4f} exceeds the maximum allowed {self.max_distribution_shift:.4f}."
            )

        allowed = {str(item).strip.lower() for item in self.allowed_issues}
        warnings: list[str] = []
        for issue in issues:
            normalized = issue.strip.lower()
            if any(normalized.startswith(prefix) for prefix in allowed):
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
            "policy": self.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_risk_level": self.max_risk_level.upper(),
            "min_confidence_score": float(self.min_confidence_score),
            "min_dataset_quality": float(self.min_dataset_quality),
            "min_evaluator_reliability": float(self.min_evaluator_reliability),
            "min_statistical_confidence": float(self.min_statistical_confidence),
            "max_duplicate_ratio": float(self.max_duplicate_ratio),
            "max_distribution_shift": float(self.max_distribution_shift),
            "confidence_level": float(self.confidence_level),
            "allowed_issues": list(self.allowed_issues),
        }


def evaluate_verification_policy(policy: ReleasePolicy | dict[str, Any], verification: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(policy, dict):
        policy = ReleasePolicy.from_dict(policy)
    return policy.evaluate(verification)


__all__ = ["ReleasePolicy", "evaluate_verification_policy"]
