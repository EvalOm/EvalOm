from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


SUPPORTED_TASK_TYPES = frozenset(
    {"feature_implementation", "bug_fix", "test_generation", "code_review"}
)
SUPPORTED_PROFILE_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_EVALUATORS = frozenset(
    {"exact_match", "contains", "regex", "numeric_tolerance", "retrieval_recall", "mrr", "ndcg"}
)
SUPPORTED_METRICS = SUPPORTED_EVALUATORS
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,99}$")
_REQUIRED_KEYS = {"profile_id", "profile_version", "task_type", "requirements"}
_ALLOWED_KEYS = _REQUIRED_KEYS | {"schema_version", "name", "description", "metadata"}
_REQUIREMENT_KEYS = {"requirement_id", "kind", "required", "evaluator", "metric", "evidence_type", "metadata"}
_EVIDENCE_TYPES = frozenset(
    {"artifact", "test_result", "execution_result", "evaluator_result", "dataset_result", "human_review", "other"}
)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must match {_IDENTIFIER.pattern!r}.")
    return value


@dataclass(frozen=True)
class ProfileRequirement:
    requirement_id: str
    kind: str
    required: bool = True
    evaluator: str | None = None
    metric: str | None = None
    evidence_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.requirement_id, "requirement_id")
        if self.kind not in {"evaluator", "metric", "evidence"}:
            raise ValueError("requirement kind must be evaluator, metric, or evidence.")
        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean.")
        if self.kind == "evaluator":
            if self.evaluator not in SUPPORTED_EVALUATORS:
                raise ValueError(f"Unsupported evaluator reference: {self.evaluator!r}.")
        elif self.kind == "metric":
            if self.metric not in SUPPORTED_METRICS:
                raise ValueError(f"Unsupported metric reference: {self.metric!r}.")
        elif self.kind == "evidence":
            if self.evidence_type not in _EVIDENCE_TYPES:
                raise ValueError(f"Unsupported evidence type: {self.evidence_type!r}.")
        if not isinstance(self.metadata, dict):
            raise ValueError("requirement metadata must be a dictionary.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProfileRequirement":
        if not isinstance(payload, Mapping):
            raise ValueError("requirements must contain dictionaries.")
        unknown = set(payload) - _REQUIREMENT_KEYS
        if unknown:
            raise ValueError(f"Unknown requirement field(s): {', '.join(sorted(map(str, unknown)))}.")
        return cls(
            requirement_id=payload.get("requirement_id", ""),
            kind=payload.get("kind", ""),
            required=payload.get("required", True),
            evaluator=payload.get("evaluator"),
            metric=payload.get("metric"),
            evidence_type=payload.get("evidence_type"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "kind": self.kind,
            "required": self.required,
            "evaluator": self.evaluator,
            "metric": self.metric,
            "evidence_type": self.evidence_type,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TaskProfile:
    profile_id: str
    profile_version: str
    task_type: str
    requirements: tuple[ProfileRequirement, ...]
    schema_version: int = SUPPORTED_SCHEMA_VERSION
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.profile_id, "profile_id")
        if self.profile_version != SUPPORTED_PROFILE_VERSION:
            raise ValueError(f"Unsupported profile_version: {self.profile_version!r}.")
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {self.task_type!r}.")
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version!r}.")
        if not self.requirements:
            raise ValueError("requirements must contain at least one requirement.")
        identifiers = [item.requirement_id for item in self.requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("requirement_id values must be unique.")
        if not isinstance(self.metadata, dict):
            raise ValueError("profile metadata must be a dictionary.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskProfile":
        if not isinstance(payload, Mapping):
            raise ValueError("profile must be a dictionary.")
        missing = _REQUIRED_KEYS - set(payload)
        if missing:
            raise ValueError(f"Missing required profile field(s): {', '.join(sorted(missing))}.")
        unknown = set(payload) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(f"Unknown profile field(s): {', '.join(sorted(map(str, unknown)))}.")
        requirements = payload["requirements"]
        if not isinstance(requirements, list):
            raise ValueError("requirements must be a list.")
        return cls(
            profile_id=payload["profile_id"],
            profile_version=payload["profile_version"],
            task_type=payload["task_type"],
            requirements=tuple(ProfileRequirement.from_dict(item) for item in requirements),
            schema_version=payload.get("schema_version", SUPPORTED_SCHEMA_VERSION),
            name=payload.get("name"),
            description=payload.get("description"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "name": self.name,
            "description": self.description,
            "requirements": [item.to_dict() for item in self.requirements],
            "metadata": dict(self.metadata),
        }

    @property
    def definition_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProfileResolution:
    profile_id: str
    profile_version: str
    definition_hash: str
    required: tuple[ProfileRequirement, ...]
    optional: tuple[ProfileRequirement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "definition_hash": self.definition_hash,
            "required": [item.to_dict() for item in self.required],
            "optional": [item.to_dict() for item in self.optional],
        }


def resolve_profile(profile: TaskProfile | Mapping[str, Any]) -> ProfileResolution:
    selected = profile if isinstance(profile, TaskProfile) else TaskProfile.from_dict(profile)
    required = tuple(item for item in selected.requirements if item.required)
    optional = tuple(item for item in selected.requirements if not item.required)
    return ProfileResolution(
        profile_id=selected.profile_id,
        profile_version=selected.profile_version,
        definition_hash=selected.definition_hash,
        required=required,
        optional=optional,
    )


def _check_requirement(
    requirement: ProfileRequirement,
    *,
    checks: Mapping[str, Any],
    evidence: Iterable[Any],
) -> dict[str, Any]:
    if requirement.kind in {"evaluator", "metric"}:
        key = requirement.evaluator or requirement.metric
        result = checks.get(key)
        if result is None:
            return {"requirement_id": requirement.requirement_id, "status": "not_run", "passed": False}
        if not isinstance(result, Mapping):
            raise ValueError(f"Check result for {key!r} must be a dictionary.")
        status = result.get("status")
        passed = result.get("passed") is True and status not in {"failed", "indeterminate", "not_run"}
        if status == "insufficient_evidence":
            return {"requirement_id": requirement.requirement_id, "status": "indeterminate", "passed": False}
        return {"requirement_id": requirement.requirement_id, "status": "passed" if passed else "failed", "passed": passed}
    matches = []
    for item in evidence:
        item_type = item.get("evidence_type") if isinstance(item, Mapping) else getattr(item, "evidence_type", None)
        if item_type == requirement.evidence_type:
            matches.append(item)
    if not matches:
        return {"requirement_id": requirement.requirement_id, "status": "not_run", "passed": False}
    usable = any(
        (item.get("lifecycle_status") if isinstance(item, Mapping) else getattr(item, "lifecycle_status", None)) == "active"
        and (item.get("digest_status") if isinstance(item, Mapping) else getattr(item, "digest_status", None))
        in {"verified", "not_applicable"}
        for item in matches
    )
    return {"requirement_id": requirement.requirement_id, "status": "passed" if usable else "indeterminate", "passed": usable}


def evaluate_profile(
    profile: TaskProfile | Mapping[str, Any],
    *,
    checks: Mapping[str, Any] | None = None,
    evidence: Iterable[Any] = (),
) -> dict[str, Any]:
    resolution = resolve_profile(profile)
    check_values = checks or {}
    evidence_values = tuple(evidence)
    results = [
        _check_requirement(requirement, checks=check_values, evidence=evidence_values)
        for requirement in (*resolution.required, *resolution.optional)
    ]
    required_results = results[: len(resolution.required)]
    if any(item["status"] == "failed" for item in required_results):
        status = "failed"
    elif any(item["status"] != "passed" for item in required_results):
        status = "insufficient_evidence"
    else:
        status = "verified"
    return {
        "status": status,
        "profile_id": resolution.profile_id,
        "profile_version": resolution.profile_version,
        "definition_hash": resolution.definition_hash,
        "checks": results,
        "required_checks": [item.requirement_id for item in resolution.required],
        "optional_checks": [item.requirement_id for item in resolution.optional],
    }


def builtin_profile(task_type: str) -> TaskProfile:
    definitions = {
        "feature_implementation": [
            {"requirement_id": "acceptance-checks", "kind": "evaluator", "evaluator": "exact_match"},
            {"requirement_id": "test-results", "kind": "evidence", "evidence_type": "test_result"},
        ],
        "bug_fix": [
            {"requirement_id": "regression-check", "kind": "evaluator", "evaluator": "exact_match"},
            {"requirement_id": "regression-evidence", "kind": "evidence", "evidence_type": "test_result"},
        ],
        "test_generation": [
            {"requirement_id": "generated-tests", "kind": "evidence", "evidence_type": "artifact"},
            {"requirement_id": "test-execution", "kind": "evidence", "evidence_type": "test_result"},
        ],
        "code_review": [
            {"requirement_id": "review-result", "kind": "evidence", "evidence_type": "human_review"},
            {"requirement_id": "review-checks", "kind": "evaluator", "evaluator": "contains"},
        ],
    }
    if task_type not in definitions:
        raise ValueError(f"Unsupported task_type: {task_type!r}.")
    return TaskProfile(
        profile_id=f"builtin.{task_type}",
        profile_version=SUPPORTED_PROFILE_VERSION,
        task_type=task_type,
        requirements=tuple(ProfileRequirement.from_dict(item) for item in definitions[task_type]),
    )


__all__ = [
    "ProfileRequirement",
    "ProfileResolution",
    "TaskProfile",
    "SUPPORTED_TASK_TYPES",
    "builtin_profile",
    "evaluate_profile",
    "resolve_profile",
]
