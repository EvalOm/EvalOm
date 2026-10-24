from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .policy import ReleasePolicy

_CONFIG_KEYS = {"project", "dataset", "evaluators", "quality_gates", "release_policy", "policy"}
_PROJECT_KEYS = {"name"}
_DATASET_KEYS = {"path"}
_EVALUATOR_KEYS = {"name", "type", "terms", "required_terms", "pattern", "tolerance", "k"}
_QUALITY_GATE_KEYS = {"minimum", "maximum"}
_SUPPORTED_EVALUATORS = {
    "exact_match", "contains", "regex", "numeric_tolerance",
    "retrieval_recall", "recall", "mrr", "ndcg",
}


def _require_mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping.")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} key(s): {', '.join(str(item) for item in unknown)}.")


def _finite_number(value: Any, field_name: str, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number.")
    number = float(value)
    if minimum is not None and number < minimum or maximum is not None and number > maximum:
        bounds = []
        if minimum is not None:
            bounds.append(f">= {minimum:g}")
        if maximum is not None:
            bounds.append(f"<= {maximum:g}")
        raise ValueError(f"{field_name} must be {' and '.join(bounds)}.")
    return number


@dataclass
class QualityGate:
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QualityGate":
        _reject_unknown(payload, _QUALITY_GATE_KEYS, "quality gate")
        if not payload:
            raise ValueError("Quality gate definition must include minimum or maximum.")
        minimum = payload.get("minimum")
        maximum = payload.get("maximum")
        if minimum is None and maximum is None:
            raise ValueError("Quality gate definition must include minimum or maximum.")
        if minimum is not None:
            minimum = _finite_number(minimum, "quality gate minimum", minimum=0, maximum=1)
        if maximum is not None:
            maximum = _finite_number(maximum, "quality gate maximum", minimum=0, maximum=1)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("Quality gate minimum cannot exceed maximum.")
        return cls(
            minimum=minimum,
            maximum=maximum,
        )


@dataclass
class EvaluationConfig:
    project_name: str = "demo"
    dataset_path: str = "datasets/demo.jsonl"
    evaluators: list[dict[str, Any]] = field(default_factory=list)
    quality_gates: dict[str, QualityGate] = field(default_factory=dict)
    release_policy: ReleasePolicy | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationConfig":
        _require_mapping(payload, "Configuration")
        _reject_unknown(payload, _CONFIG_KEYS, "configuration")

        project = _require_mapping(payload.get("project", {}), "project")
        dataset = _require_mapping(payload.get("dataset", {}), "dataset")
        _reject_unknown(project, _PROJECT_KEYS, "project")
        _reject_unknown(dataset, _DATASET_KEYS, "dataset")
        project_name = project.get("name", "demo")
        if "path" not in dataset:
            raise ValueError("dataset.path is required; refusing to substitute a demo dataset.")
        dataset_path = dataset["path"]
        if not isinstance(project_name, str) or not project_name.strip():
            raise ValueError("project.name must be a non-empty string.")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError("dataset.path must be a non-empty string.")

        if "evaluators" not in payload:
            evaluators: list[dict[str, Any]] = [{"name": "exact_match"}]
        else:
            evaluators_value = payload["evaluators"]
            if not isinstance(evaluators_value, list):
                raise ValueError("evaluators must be a list.")
            evaluators = []
            for index, item in enumerate(evaluators_value):
                evaluator = _require_mapping(item, f"evaluators[{index}]")
                _reject_unknown(evaluator, _EVALUATOR_KEYS, f"evaluators[{index}]")
                name = evaluator.get("name") or evaluator.get("type")
                if not isinstance(name, str) or name not in _SUPPORTED_EVALUATORS:
                    raise ValueError(f"Unsupported evaluator in evaluators[{index}]: {name!r}.")
                evaluator_type = evaluator.get("type")
                if evaluator_type is not None and (
                    not isinstance(evaluator_type, str)
                    or evaluator_type not in _SUPPORTED_EVALUATORS | {"deterministic"}
                ):
                    raise ValueError(f"Unsupported evaluator type in evaluators[{index}]: {evaluator_type!r}.")
                if "terms" in evaluator or "required_terms" in evaluator:
                    terms = evaluator.get("terms", evaluator.get("required_terms"))
                    if not isinstance(terms, list) or any(not isinstance(term, str) or not term.strip() for term in terms):
                        raise ValueError(f"evaluators[{index}].terms must be a list of non-empty strings.")
                if "pattern" in evaluator and (
                    not isinstance(evaluator["pattern"], str) or not evaluator["pattern"]
                ):
                    raise ValueError(f"evaluators[{index}].pattern must be a non-empty string.")
                if "tolerance" in evaluator:
                    _finite_number(evaluator["tolerance"], f"evaluators[{index}].tolerance", minimum=0)
                if "k" in evaluator:
                    if isinstance(evaluator["k"], bool) or not isinstance(evaluator["k"], int) or evaluator["k"] < 1:
                        raise ValueError(f"evaluators[{index}].k must be a positive integer.")
                evaluators.append(dict(evaluator))

        quality = payload.get("quality_gates", {})
        _require_mapping(quality, "quality_gates")
        quality_gates: dict[str, QualityGate] = {}
        for name, value in quality.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Quality gate names must be non-empty strings.")
            if name not in _SUPPORTED_EVALUATORS:
                raise ValueError(f"Unsupported quality gate evaluator: {name!r}.")
            quality_gates[name] = QualityGate.from_dict(_require_mapping(value, f"quality_gates.{name}"))
        release_policy = payload.get("release_policy", payload.get("policy"))

        return cls(
            project_name=project_name,
            dataset_path=dataset_path,
            evaluators=evaluators,
            quality_gates=quality_gates,
            release_policy=(
                ReleasePolicy.from_dict(release_policy)
                if release_policy is not None
                else None
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationConfig":
        config_path = Path(path)
        raw_text = config_path.read_text(encoding="utf-8")

        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load evaluation configuration.") from exc
        try:
            data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            detail = f" at line {exc.problem_mark.line + 1}, column {exc.problem_mark.column + 1}" if getattr(exc, "problem_mark", None) else ""
            raise ValueError(f"Invalid YAML in {config_path}{detail}: {exc.problem}") from exc
        if data is None:
            raise ValueError(f"Configuration file is empty: {config_path}.")
        return cls.from_dict(data)
