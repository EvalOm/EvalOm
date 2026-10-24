from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .comparison import compare_continuous
from .dataset import EvaluationDataset, Sample


@dataclass
class EvaluationRunResult:
    dataset_name: str
    dataset_version: str
    total_samples: int
    metrics: list[dict[str, Any]] = field(default_factory=list)
    sample_results: list[dict[str, Any]] = field(default_factory=list)
    quality_gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def average_score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(float(item["score"]) for item in self.metrics) / len(self.metrics)

    def check_quality_gates(self) -> dict[str, Any]:
        insufficient_metrics = [
            item for item in self.metrics
            if item.get("status") == "insufficient_evidence"
        ]
        failures: list[str] = [
            f"{item['name']} has insufficient reference evidence."
            for item in insufficient_metrics
        ]
        if not self.quality_gates:
            return {"passed": not failures, "failures": failures}
        for metric_name, gate in self.quality_gates.items():
            metric_values = [item["score"] for item in self.metrics if item["name"] == metric_name]
            if not metric_values:
                failures.append(f"{metric_name} missing: no scores were produced for the configured quality gate.")
                continue
            average_metric = sum(float(value) for value in metric_values) / len(metric_values)
            if gate.get("minimum") is not None and average_metric < float(gate["minimum"]):
                failures.append(f"{metric_name} below minimum: {average_metric:.4f} < {float(gate['minimum']):.4f}")
            if gate.get("maximum") is not None and average_metric > float(gate["maximum"]):
                failures.append(f"{metric_name} above maximum: {average_metric:.4f} > {float(gate['maximum']):.4f}")

        return {"passed": not failures, "failures": failures}

    def summary(self) -> str:
        avg = self.average_score()
        quality = self.check_quality_gates()
        status = "PASS" if quality["passed"] and avg >= 0.8 else "FAIL"
        lines = [
            "EvalOm evaluation summary",
            f"Dataset: {self.dataset_name} ({self.dataset_version})",
            f"Samples evaluated: {self.total_samples}",
            f"Average score: {avg:.2f}",
            f"Status: {status}",
        ]
        if quality["failures"]:
            lines.append("Quality gate failures:")
            lines.extend(f"- {failure}" for failure in quality["failures"])
        policy = self.metadata.get("policy_decision")
        if isinstance(policy, dict):
            lines.append(f"Release policy: {policy.get('decision', 'UNKNOWN')}")
            lines.extend(f"- {failure}" for failure in policy.get("failures", []))
        verification = self.metadata.get("verification")
        if isinstance(verification, dict):
            lines.append(f"Assurance status: {verification.get('status', 'unknown')}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "total_samples": self.total_samples,
            "metrics": self.metrics,
            "sample_results": self.sample_results,
            "quality_gates": self.quality_gates,
            "metadata": self.metadata,
        }


class EvaluationRunner:
    def __init__(self, dataset: EvaluationDataset, evaluators: Iterable[Any], quality_gates: dict[str, dict[str, Any]] | None = None):
        self.dataset = dataset
        self.evaluators = list(evaluators)
        self.quality_gates = quality_gates or {}

    def run(self, app: Callable[[str], dict[str, Any]], metadata: dict[str, Any] | None = None) -> EvaluationRunResult:
        self.dataset.validate()
        sample_metrics = []
        sample_results = []
        evaluator_counts: dict[str, int] = {}
        evaluator_names: list[str] = []
        for evaluator in self.evaluators:
            base_name = str(evaluator.name)
            evaluator_counts[base_name] = evaluator_counts.get(base_name, 0) + 1
            evaluator_names.append(
                base_name
                if evaluator_counts[base_name] == 1
                else f"{base_name}#{evaluator_counts[base_name]}"
            )

        for sample in self.dataset.samples:
            row = {"id": sample.id, "input": sample.input, "expected_answer": sample.expected_answer, "expected_context": sample.expected_context}
            output = app(sample.input)
            if not isinstance(output, dict):
                raise TypeError("Application output must be a dictionary with at least an 'answer' field.")

            if "answer" not in output:
                raise ValueError("Application output must include an 'answer' field.")

            row["output"] = output
            evaluator_scores = []
            for evaluator, result_name in zip(self.evaluators, evaluator_names):
                result = evaluator.evaluate(row, output)
                result = dict(result)
                result["name"] = result_name
                evaluator_scores.append({
                    **result,
                    "name": result_name,
                    "passed": result.get("passed", False),
                    "status": result.get("status", "valid"),
                })

            sample_results.append({
                "sample_id": sample.id,
                "input": sample.input,
                "expected_answer": sample.expected_answer,
                "output": output,
                "result": evaluator_scores,
            })
            sample_metrics.extend({
                **metric,
                "sample_id": sample.id,
                "name": metric["name"],
                "score": metric["score"],
                "passed": metric["passed"],
                "status": metric.get("status", "valid"),
            } for metric in evaluator_scores)

        result_metadata = dict(metadata or {})
        result_metadata.setdefault("dataset_hash", self.dataset.fingerprint())
        return EvaluationRunResult(
            dataset_name=self.dataset.name,
            dataset_version=self.dataset.version,
            total_samples=len(self.dataset.samples),
            metrics=sample_metrics,
            sample_results=sample_results,
            quality_gates=self.quality_gates,
            metadata=result_metadata,
        )

    @classmethod
    def compare_runs(cls, first: EvaluationRunResult, second: EvaluationRunResult) -> dict[str, Any]:
        results = {}
        metrics = sorted({metric["name"] for metric in first.metrics} | {metric["name"] for metric in second.metrics})
        for metric_name in metrics:
            first_values = [metric["score"] for metric in first.metrics if metric["name"] == metric_name]
            second_values = [metric["score"] for metric in second.metrics if metric["name"] == metric_name]
            first_avg = sum(first_values) / len(first_values) if first_values else 0.0
            second_avg = sum(second_values) / len(second_values) if second_values else 0.0
            delta = second_avg - first_avg
            comparison = compare_continuous(
                first_values,
                second_values,
                metric_name=metric_name,
                direction="higher_is_better",
            )
            results[metric_name] = {
                "baseline": first_avg,
                "candidate": second_avg,
                "delta": delta,
                "delta_pct": ((delta / first_avg) * 100.0) if first_avg else 0.0,
                "comparison": comparison.to_dict(),
            }
        return results
