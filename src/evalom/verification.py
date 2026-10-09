from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .dataset import EvaluationDataset
from .runner import EvaluationRunResult


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _average(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def normalize_evaluation_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert heterogeneous evaluation payloads into a consistent local schema."""
    if not isinstance(record, dict):
        raise TypeError("Evaluation records must be dictionaries.")

    score = record.get("score")
    if score is None:
        score = record.get("average_score")
    if score is None:
        score = record.get("metric_score")
    if score is None and isinstance(record.get("metrics"), dict):
        score = record["metrics"].get("score")

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if isinstance(record.get("result"), dict):
        metadata = {**metadata, **record["result"]}

    output = record.get("output")
    if isinstance(output, dict):
        answer = output.get("answer") or output.get("value")
    else:
        answer = output

    context = record.get("context") or record.get("contexts") or record.get("retrieved_context") or []
    if isinstance(context, str):
        context = [context]

    return {
        "sample_id": str(record.get("sample_id") or record.get("id") or record.get("sampleId") or record.get("trace_id") or "unknown"),
        "input": record.get("input") or record.get("question") or record.get("prompt") or "",
        "output": answer if answer is not None else (record.get("answer") or record.get("output_text") or ""),
        "ground_truth": record.get("ground_truth") or record.get("expected_answer") or record.get("expected") or record.get("target") or "",
        "context": context,
        "evaluator": record.get("evaluator") or record.get("metric") or record.get("name") or "unknown",
        "score": float(score) if score is not None else 0.0,
        "metadata": metadata,
    }


@dataclass
class DatasetHealth:
    sample_size: int
    duplicate_ratio: float
    coverage_ratio: float
    drift_score: float
    quality_score: float
    distribution_shift: float = 0.0
    edge_case_coverage: float = 0.0
    category_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "duplicate_ratio": round(self.duplicate_ratio, 4),
            "coverage_ratio": round(self.coverage_ratio, 4),
            "drift_score": round(self.drift_score, 4),
            "quality_score": round(self.quality_score, 4),
            "distribution_shift": round(self.distribution_shift, 4),
            "edge_case_coverage": round(self.edge_case_coverage, 4),
            "category_distribution": dict(self.category_distribution),
        }


@dataclass
class EvaluationVerificationResult:
    confidence_score: float
    risk_level: str
    recommendation: str
    evaluator_reliability: float
    dataset_quality: float
    statistical_confidence: float
    issues: list[str] = field(default_factory=list)
    confidence_interval: dict[str, float] = field(default_factory=lambda: {"lower": 0.0, "upper": 0.0})
    failure_breakdown: dict[str, int] = field(default_factory=dict)
    evaluator_agreement: float | None = None
    human_agreement: float | None = None
    dataset_health: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_score": round(self.confidence_score, 2),
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "evaluator_reliability": round(self.evaluator_reliability, 4),
            "dataset_quality": round(self.dataset_quality, 4),
            "statistical_confidence": round(self.statistical_confidence, 4),
            "evaluator_agreement": self.evaluator_agreement,
            "human_agreement": self.human_agreement,
            "confidence_interval": {
                "lower": round(float(self.confidence_interval.get("lower", 0.0)), 4),
                "upper": round(float(self.confidence_interval.get("upper", 0.0)), 4),
            },
            "failure_breakdown": dict(self.failure_breakdown),
            "dataset_health": dict(self.dataset_health),
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }


class VerificationEngine:
    """Compute whether an evaluation result is trustworthy enough for release decisions."""

    def __init__(
        self,
        *,
        minimum_sample_size: int = 50,
        minimum_dataset_quality: float = 0.75,
        minimum_evaluator_reliability: float = 0.8,
        minimum_statistical_confidence: float = 0.75,
    ) -> None:
        self.minimum_sample_size = minimum_sample_size
        self.minimum_dataset_quality = minimum_dataset_quality
        self.minimum_evaluator_reliability = minimum_evaluator_reliability
        self.minimum_statistical_confidence = minimum_statistical_confidence

    def analyze_dataset(self, dataset: EvaluationDataset | None, fallback_sample_size: int = 0) -> DatasetHealth:
        if dataset is None:
            sample_size = fallback_sample_size
            return DatasetHealth(
                sample_size=sample_size,
                duplicate_ratio=0.0,
                coverage_ratio=0.5 if sample_size > 0 else 0.0,
                drift_score=1.0,
                quality_score=0.5 if sample_size > 0 else 0.0,
                distribution_shift=0.0,
                edge_case_coverage=0.5 if sample_size > 0 else 0.0,
            )

        sample_size = len(dataset.samples)
        seen_inputs: set[str] = set()
        duplicates = 0
        hard_edge_cases = 0
        category_distribution: dict[str, int] = {}
        for sample in dataset.samples:
            key = str(sample.input).strip().lower()
            if not key:
                continue
            if key in seen_inputs:
                duplicates += 1
            else:
                seen_inputs.add(key)

            metadata = getattr(sample, "metadata", {}) or {}
            difficulty = str(metadata.get("difficulty", "")).lower()
            if sample.expected_context and (
                "edge" in difficulty or "hard" in difficulty or "adversarial" in difficulty or len(sample.expected_context) > 1
            ):
                hard_edge_cases += 1

            category = metadata.get("category") or metadata.get("topic") or metadata.get("source") or metadata.get("dataset") or "uncategorized"
            category_name = str(category).lower()
            category_distribution[category_name] = category_distribution.get(category_name, 0) + 1

        duplicate_ratio = duplicates / sample_size if sample_size else 0.0
        covered_samples = sum(1 for sample in dataset.samples if sample.expected_context)
        coverage_ratio = covered_samples / sample_size if sample_size else 0.0
        edge_case_coverage = hard_edge_cases / sample_size if sample_size else 0.0
        max_group_size = max(category_distribution.values()) if category_distribution else 0
        distribution_shift = _clamp(1.0 - (max_group_size / sample_size) if sample_size else 0.0, 0.0, 1.0)
        drift_score = _clamp(1.0 - (0.4 * (1.0 - coverage_ratio)) - (0.6 * duplicate_ratio), 0.0, 1.0)
        quality_score = _clamp(1.0 - duplicate_ratio * 0.75 - max(0.0, 0.5 - coverage_ratio) * 0.5, 0.0, 1.0)

        return DatasetHealth(
            sample_size=sample_size,
            duplicate_ratio=duplicate_ratio,
            coverage_ratio=coverage_ratio,
            drift_score=drift_score,
            quality_score=quality_score,
            distribution_shift=distribution_shift,
            edge_case_coverage=edge_case_coverage,
            category_distribution=category_distribution,
        )

    def verify(
        self,
        result: EvaluationRunResult | None,
        *,
        dataset: EvaluationDataset | None = None,
        evaluator_agreement: float | None = None,
        human_agreement: float | None = None,
        sample_size: int | None = None,
    ) -> EvaluationVerificationResult:
        if result is None:
            raise ValueError("An EvaluationRunResult is required for verification.")

        dataset_health = self.analyze_dataset(dataset, fallback_sample_size=result.total_samples)
        if sample_size is not None:
            dataset_health.sample_size = sample_size
            if sample_size > 0 and dataset is None:
                dataset_health.coverage_ratio = 0.5
                dataset_health.quality_score = 0.5

        metric_scores = [float(item["score"]) for item in result.metrics]
        mean_score = _average(metric_scores)
        metric_variance = 0.0 if not metric_scores else _average([(score - mean_score) ** 2 for score in metric_scores])
        stddev = (metric_variance ** 0.5) if metric_scores else 0.0
        evaluator_reliability = _clamp(1.0 - metric_variance * 2.0, 0.0, 1.0)

        agreement_score = 1.0
        if evaluator_agreement is not None:
            agreement_score = _clamp(float(evaluator_agreement), 0.0, 1.0)
            evaluator_reliability = (evaluator_reliability + agreement_score) / 2.0
        if human_agreement is not None:
            human_score = _clamp(float(human_agreement), 0.0, 1.0)
            agreement_score = (agreement_score + human_score) / 2.0 if evaluator_agreement is not None else human_score
            evaluator_reliability = (evaluator_reliability + human_score) / 2.0

        effective_sample_size = dataset_health.sample_size or len(result.metrics) or result.total_samples
        statistical_confidence = _clamp((effective_sample_size / max(self.minimum_sample_size, 1)) * 0.85, 0.0, 1.0)
        statistical_confidence = _clamp(statistical_confidence * (1.0 - min(0.5, metric_variance * 1.5)), 0.0, 1.0)

        dataset_quality = max(dataset_health.quality_score, 0.0)
        if dataset_health.sample_size < self.minimum_sample_size:
            dataset_quality = min(dataset_quality, 0.9)

        confidence_score = (
            evaluator_reliability * 0.4
            + dataset_quality * 0.35
            + statistical_confidence * 0.25
        ) * 100.0

        margin_of_error = 0.0 if not metric_scores else (1.96 * stddev / (len(metric_scores) ** 0.5))
        confidence_interval = {
            "lower": max(0.0, mean_score - margin_of_error),
            "upper": min(1.0, mean_score + margin_of_error),
        }

        failure_breakdown: dict[str, int] = {}
        for sample_result in result.sample_results:
            for metric in sample_result.get("result", []):
                name = str(metric.get("name", "unknown"))
                if not metric.get("passed", False):
                    failure_breakdown[name] = failure_breakdown.get(name, 0) + 1

        issues: list[str] = []
        if evaluator_agreement is not None and evaluator_agreement < 0.85:
            issues.append("Evaluator agreement is below the recommended threshold.")
        if human_agreement is not None and human_agreement < 0.8:
            issues.append("Human agreement is below the recommended threshold.")
        if evaluator_reliability < self.minimum_evaluator_reliability:
            issues.append("Evaluator reliability is below the recommended threshold.")
        if dataset_quality < self.minimum_dataset_quality:
            issues.append("Dataset quality is too weak to support a high-confidence release decision.")
        if statistical_confidence < self.minimum_statistical_confidence:
            issues.append("Sample size or variance makes the evaluation statistically weak.")
        if dataset_health.duplicate_ratio > 0.05:
            issues.append(f"Detected {dataset_health.duplicate_ratio:.2%} duplicate or repeated inputs in the dataset.")
        if dataset_health.coverage_ratio < 0.5:
            issues.append("The dataset coverage is low relative to the expected production mix.")
        if dataset_health.distribution_shift > 0.35:
            issues.append("Dataset distribution shift suggests the evaluation may not represent production traffic.")
        if result.average_score() < 0.8 and dataset_quality >= self.minimum_dataset_quality:
            issues.append("The observed average score is lower than the typical release threshold.")
        if failure_breakdown:
            issue_details = ", ".join(f"{name}: {count}" for name, count in sorted(failure_breakdown.items()))
            issues.append(f"Failure analysis indicates weak spots in evaluators: {issue_details}.")

        if confidence_score >= 80 and not issues:
            risk_level = "LOW"
            recommendation = "Proceed with deployment confidence based on the current evaluation quality."
        elif confidence_score >= 60:
            risk_level = "MEDIUM"
            recommendation = "Review evaluator quality and dataset coverage before relying on this score as a release gate."
        else:
            risk_level = "HIGH"
            recommendation = "Do not treat this evaluation as a release gate until the dataset and evaluator quality are improved."

        dataset_summary = dataset_health.to_dict()
        dataset_hash = dataset.fingerprint() if dataset is not None else None
        return EvaluationVerificationResult(
            confidence_score=confidence_score,
            risk_level=risk_level,
            recommendation=recommendation,
            evaluator_reliability=evaluator_reliability,
            dataset_quality=dataset_quality,
            statistical_confidence=statistical_confidence,
            issues=issues,
            confidence_interval=confidence_interval,
            failure_breakdown=failure_breakdown,
            evaluator_agreement=evaluator_agreement,
            human_agreement=human_agreement,
            dataset_health=dataset_summary,
            metadata={
                "average_score": round(mean_score, 4),
                "confidence_interval": confidence_interval,
                "agreement_score": round(agreement_score, 4),
                "metrics": result.metrics,
                "dataset_hash": dataset_hash,
            },
        )


def verify_evaluation_result(
    result: EvaluationRunResult,
    *,
    dataset: EvaluationDataset | None = None,
    evaluator_agreement: float | None = None,
    human_agreement: float | None = None,
    sample_size: int | None = None,
) -> EvaluationVerificationResult:
    engine = VerificationEngine()
    return engine.verify(
        result,
        dataset=dataset,
        evaluator_agreement=evaluator_agreement,
        human_agreement=human_agreement,
        sample_size=sample_size,
    )


__all__ = [
    "DatasetHealth",
    "EvaluationVerificationResult",
    "VerificationEngine",
    "normalize_evaluation_record",
    "verify_evaluation_result",
]
