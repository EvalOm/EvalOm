from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

from .continuous import ContinuousStatisticalResult, bootstrap_mean_interval
from .statistical import BinaryStatisticalResult, wilson_score_interval


_DIRECTIONS = {"higher_is_better", "lower_is_better", "none"}
_COMPARABILITY_FIELDS = (
    "metric_definition",
    "metric_version",
    "dataset_name",
    "dataset_version",
    "task_profile",
    "label_mapping",
    "sampling_design",
)


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence_level must be a number.")
    numeric = float(value)
    if 1 < numeric <= 100:
        numeric /= 100
    if not 0 < numeric < 1 or not math.isfinite(numeric):
        raise ValueError("confidence_level must be between 0 and 1, exclusive.")
    return numeric


def _bootstrap_difference_interval(
    baseline: list[float],
    candidate: list[float],
    *,
    confidence_level: float,
    resamples: int,
    seed: int | None,
) -> tuple[float, float]:
    import random

    rng = random.Random(seed)
    differences = sorted(
        statistics.fmean(
            candidate[rng.randrange(len(candidate))] for _ in candidate
        ) - statistics.fmean(
            baseline[rng.randrange(len(baseline))] for _ in baseline
        )
        for _ in range(resamples)
    )
    alpha = (1 - confidence_level) / 2
    def percentile(probability: float) -> float:
        position = probability * (len(differences) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return differences[lower]
        fraction = position - lower
        return differences[lower] + (differences[upper] - differences[lower]) * fraction
    return percentile(alpha), percentile(1 - alpha)


def _validate_values(values: Sequence[Any], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of finite numbers.")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain only finite numbers.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers.")
        normalized.append(number)
    return normalized


def validate_comparability(
    baseline_metadata: dict[str, Any] | None,
    candidate_metadata: dict[str, Any] | None,
    *,
    required_fields: Sequence[str] = ("metric_definition",),
) -> tuple[bool, list[str], list[str]]:
    """Validate metadata that affects whether a statistical comparison is meaningful."""
    baseline = baseline_metadata or {}
    candidate = candidate_metadata or {}
    errors: list[str] = []
    warnings: list[str] = []
    for field_name in required_fields:
        if field_name not in baseline or field_name not in candidate:
            errors.append(f"Missing comparability metadata: {field_name}.")
        elif baseline[field_name] != candidate[field_name]:
            errors.append(f"Incompatible {field_name}.")
    for field_name in _COMPARABILITY_FIELDS:
        if field_name in required_fields:
            continue
        if field_name in baseline and field_name in candidate and baseline[field_name] != candidate[field_name]:
            if field_name in {"dataset_name", "dataset_version", "label_mapping", "sampling_design"}:
                errors.append(f"Incompatible {field_name}.")
            else:
                warnings.append(f"{field_name} differs between baseline and candidate.")
    return not errors, errors, warnings


@dataclass(frozen=True)
class ComparisonResult:
    metric_name: str
    baseline_value: float | None
    candidate_value: float | None
    difference: float | None
    difference_convention: str
    direction: str
    interpretation: str
    status: str
    evidence_status: str
    design: str
    method: str | None
    confidence_level: float | None
    confidence_interval: dict[str, float | None]
    statistically_supported: bool | None
    practical_threshold: float | None = None
    practically_significant: bool | None = None
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "difference": self.difference,
            "difference_convention": self.difference_convention,
            "direction": self.direction,
            "interpretation": self.interpretation,
            "status": self.status,
            "evidence_status": self.evidence_status,
            "design": self.design,
            "method": self.method,
            "confidence_level": self.confidence_level,
            "confidence_interval": dict(self.confidence_interval),
            "statistically_supported": self.statistically_supported,
            "practical_threshold": self.practical_threshold,
            "practically_significant": self.practically_significant,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }


def _interpret(difference: float | None, direction: str, supported: bool | None) -> str:
    if difference is None or direction == "none":
        return "not_applicable" if direction == "none" else "uncertain"
    if supported is not True:
        return "uncertain"
    if difference == 0:
        return "no_change"
    improvement = difference > 0 if direction == "higher_is_better" else difference < 0
    return "improvement" if improvement else "regression"


def _result(
    *,
    metric_name: str,
    baseline: float | None,
    candidate: float | None,
    difference: float | None,
    direction: str,
    design: str,
    method: str | None,
    confidence_level: float | None,
    interval: dict[str, float | None],
    supported: bool | None,
    evidence_status: str,
    status: str,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
    practical_threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> ComparisonResult:
    practical = None
    if practical_threshold is not None and difference is not None:
        magnitude = difference if direction == "higher_is_better" else -difference
        practical = magnitude >= practical_threshold
    return ComparisonResult(
        metric_name=metric_name,
        baseline_value=baseline,
        candidate_value=candidate,
        difference=difference,
        difference_convention="candidate - baseline",
        direction=direction,
        interpretation=_interpret(difference, direction, supported),
        status=status,
        evidence_status=evidence_status,
        design=design,
        method=method,
        confidence_level=confidence_level,
        confidence_interval=interval,
        statistically_supported=supported,
        practical_threshold=practical_threshold,
        practically_significant=practical,
        warnings=tuple(warnings or ()),
        limitations=tuple(limitations or ()),
        metadata=dict(metadata or {}),
    )


def compare_continuous(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    *,
    metric_name: str = "metric",
    direction: str = "higher_is_better",
    paired: bool = False,
    baseline_ids: Sequence[Any] | None = None,
    candidate_ids: Sequence[Any] | None = None,
    confidence_level: float = 0.95,
    resamples: int = 1000,
    seed: int | None = 0,
    minimum_sample_size: int | None = None,
    practical_threshold: float | None = None,
    baseline_metadata: dict[str, Any] | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> ComparisonResult:
    if direction not in _DIRECTIONS:
        raise ValueError("direction must be higher_is_better, lower_is_better, or none.")
    level = _normalize_confidence(confidence_level)
    if practical_threshold is not None and (
        isinstance(practical_threshold, bool)
        or not isinstance(practical_threshold, (int, float))
        or not math.isfinite(float(practical_threshold))
        or float(practical_threshold) < 0
    ):
        raise ValueError("practical_threshold must be a finite non-negative number.")
    compatible, errors, warnings = validate_comparability(
        baseline_metadata,
        candidate_metadata,
        required_fields=("metric_definition",) if baseline_metadata or candidate_metadata else (),
    )
    if not compatible:
        return _result(
            metric_name=metric_name, baseline=None, candidate=None, difference=None,
            direction=direction, design="paired" if paired else "independent",
            method=None, confidence_level=level, interval={"lower": None, "upper": None},
            supported=None, evidence_status="unsupported", status="insufficient_evidence",
            warnings=errors + warnings,
        )
    baseline_values = _validate_values(baseline, "baseline")
    candidate_values = _validate_values(candidate, "candidate")
    if not baseline_values or not candidate_values:
        return _result(
            metric_name=metric_name, baseline=None, candidate=None, difference=None,
            direction=direction, design="paired" if paired else "independent",
            method=None, confidence_level=level, interval={"lower": None, "upper": None},
            supported=None, evidence_status="insufficient", status="insufficient_evidence",
            warnings=warnings, limitations=["Both runs require at least one observation."],
        )
    if minimum_sample_size is not None and (
        isinstance(minimum_sample_size, bool) or not isinstance(minimum_sample_size, int) or minimum_sample_size < 0
    ):
        raise ValueError("minimum_sample_size must be a non-negative integer or None.")
    if paired:
        if baseline_ids is None or candidate_ids is None:
            return _result(
                metric_name=metric_name, baseline=None, candidate=None, difference=None,
                direction=direction, design="paired", method=None, confidence_level=level,
                interval={"lower": None, "upper": None}, supported=None,
                evidence_status="unsupported", status="insufficient_evidence",
                warnings=warnings, limitations=["Pairing requires stable identifiers on both runs."],
            )
        if len(baseline_values) != len(baseline_ids) or len(candidate_values) != len(candidate_ids):
            raise ValueError("Observation identifiers must match their value sequences.")
        baseline_map = dict(zip(baseline_ids, baseline_values))
        candidate_map = dict(zip(candidate_ids, candidate_values))
        if len(baseline_map) != len(baseline_ids) or len(candidate_map) != len(candidate_ids):
            raise ValueError("Observation identifiers must be unique.")
        if set(baseline_map) != set(candidate_map):
            return _result(
                metric_name=metric_name, baseline=None, candidate=None, difference=None,
                direction=direction, design="paired", method=None, confidence_level=level,
                interval={"lower": None, "upper": None}, supported=None,
                evidence_status="unsupported", status="insufficient_evidence",
                warnings=warnings, limitations=["Paired runs must cover the same identifiers."],
            )
        differences = [candidate_map[key] - baseline_map[key] for key in baseline_ids]
        baseline_mean = statistics.fmean(baseline_values)
        candidate_mean = statistics.fmean(candidate_values)
        uncertainty: ContinuousStatisticalResult = bootstrap_mean_interval(
            differences, confidence_level=level, resamples=resamples, seed=seed,
            minimum_sample_size=minimum_sample_size,
        )
        supported = None if uncertainty.lower_bound is None else not (
            uncertainty.lower_bound <= 0 <= uncertainty.upper_bound
        )
        return _result(
            metric_name=metric_name, baseline=baseline_mean, candidate=candidate_mean,
            difference=candidate_mean - baseline_mean, direction=direction, design="paired",
            method="paired_percentile_bootstrap", confidence_level=level,
            interval={"lower": uncertainty.lower_bound, "upper": uncertainty.upper_bound},
            supported=supported,
            evidence_status="adequate" if uncertainty.status == "valid" else "insufficient",
            status="verified" if uncertainty.status == "valid" else "insufficient_evidence",
            warnings=warnings, limitations=list(uncertainty.limitations),
            practical_threshold=practical_threshold,
            metadata={"resamples": resamples, "seed": seed, "pair_count": len(differences)},
        )
    baseline_stats = bootstrap_mean_interval(
        baseline_values, confidence_level=level, resamples=resamples, seed=seed,
        minimum_sample_size=minimum_sample_size,
    )
    candidate_stats = bootstrap_mean_interval(
        candidate_values, confidence_level=level, resamples=resamples, seed=seed,
        minimum_sample_size=minimum_sample_size,
    )
    baseline_mean = baseline_stats.mean
    candidate_mean = candidate_stats.mean
    difference = candidate_mean - baseline_mean
    if len(baseline_values) < 2 or len(candidate_values) < 2:
        lower = upper = None
        supported = None
    else:
        lower, upper = _bootstrap_difference_interval(
            baseline_values, candidate_values,
            confidence_level=level, resamples=resamples, seed=seed,
        )
        supported = not (lower <= 0 <= upper)
    adequate = baseline_stats.status == "valid" and candidate_stats.status == "valid" and lower is not None
    return _result(
        metric_name=metric_name, baseline=baseline_mean, candidate=candidate_mean,
        difference=difference, direction=direction, design="independent",
        method="independent_percentile_bootstrap", confidence_level=level,
        interval={"lower": lower, "upper": upper}, supported=supported,
        evidence_status="adequate" if adequate else "insufficient",
        status="verified" if adequate else "insufficient_evidence",
        warnings=warnings,
        limitations=[
            "Independent bootstrap resamples assume independent, representative observations.",
            "The interval does not correct sampling bias, correlation, or distribution shift.",
        ],
        practical_threshold=practical_threshold,
        metadata={
            "baseline_count": len(baseline_values),
            "candidate_count": len(candidate_values),
            "resamples": resamples,
            "seed": seed,
        },
    )


def compare_binary(
    baseline_successes: int,
    baseline_total: int,
    candidate_successes: int,
    candidate_total: int,
    *,
    metric_name: str = "binary_metric",
    direction: str = "higher_is_better",
    confidence_level: float = 0.95,
    minimum_sample_size: int | None = None,
    practical_threshold: float | None = None,
) -> ComparisonResult:
    if direction not in _DIRECTIONS:
        raise ValueError("direction must be higher_is_better, lower_is_better, or none.")
    level = _normalize_confidence(confidence_level)
    baseline_stats: BinaryStatisticalResult = wilson_score_interval(
        baseline_successes, baseline_total, confidence_level=level
    )
    candidate_stats: BinaryStatisticalResult = wilson_score_interval(
        candidate_successes, candidate_total, confidence_level=level
    )
    if minimum_sample_size is not None and (
        isinstance(minimum_sample_size, bool) or not isinstance(minimum_sample_size, int) or minimum_sample_size < 0
    ):
        raise ValueError("minimum_sample_size must be a non-negative integer or None.")
    if baseline_stats.status != "valid" or candidate_stats.status != "valid":
        return _result(
            metric_name=metric_name, baseline=baseline_stats.observed_success_rate,
            candidate=candidate_stats.observed_success_rate, difference=None,
            direction=direction, design="independent", method="two_proportion_normal_approximation",
            confidence_level=level, interval={"lower": None, "upper": None},
            supported=None, evidence_status="insufficient", status="insufficient_evidence",
            limitations=["Both runs require at least one binary observation."],
        )
    if minimum_sample_size is not None and min(baseline_total, candidate_total) < minimum_sample_size:
        return _result(
            metric_name=metric_name, baseline=baseline_stats.observed_success_rate,
            candidate=candidate_stats.observed_success_rate,
            difference=candidate_stats.observed_success_rate - baseline_stats.observed_success_rate,
            direction=direction, design="independent", method="two_proportion_normal_approximation",
            confidence_level=level, interval={"lower": None, "upper": None},
            supported=None, evidence_status="insufficient", status="insufficient_evidence",
            limitations=[f"Both groups require at least {minimum_sample_size} observations."],
        )
    difference = candidate_stats.observed_success_rate - baseline_stats.observed_success_rate
    lower = candidate_stats.lower_bound - baseline_stats.upper_bound
    upper = candidate_stats.upper_bound - baseline_stats.lower_bound
    return _result(
        metric_name=metric_name, baseline=baseline_stats.observed_success_rate,
        candidate=candidate_stats.observed_success_rate, difference=difference,
        direction=direction, design="independent", method="newcombe_wilson_difference",
        confidence_level=level, interval={"lower": lower, "upper": upper},
        supported=not (lower <= 0 <= upper), evidence_status="adequate", status="verified",
        practical_threshold=practical_threshold,
        metadata={"baseline_statistics": baseline_stats.to_dict(), "candidate_statistics": candidate_stats.to_dict()},
        limitations=[
            "Newcombe's interval combines independent Wilson score intervals.",
            "Paired binary comparisons are unsupported.",
        ],
    )


__all__ = [
    "ComparisonResult",
    "compare_binary",
    "compare_continuous",
    "validate_comparability",
]
