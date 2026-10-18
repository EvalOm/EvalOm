from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .riav import AssessmentStatus


EVENT_TYPES = frozenset({"tool_invocation", "tool_result", "retry", "failure", "completion"})
MAX_TASK_ID_LENGTH = 200
MAX_OUTCOME_LENGTH = 2_000
MAX_EVENT_ID_LENGTH = 200
MAX_TOOL_NAME_LENGTH = 200
MAX_METADATA_KEYS = 50
MAX_METADATA_BYTES = 16_384
MAX_EVENTS = 10_000
MAX_ARTIFACTS = 100
CHECK_STATUSES = frozenset({"passed", "failed", "not_run", "indeterminate"})


def _bounded_string(value: Any, name: str, maximum: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    if required and not value.strip():
        raise ValueError(f"{name} cannot be empty.")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds the maximum length of {maximum}.")
    return value


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")
    if len(metadata) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata cannot contain more than {MAX_METADATA_KEYS} keys.")
    try:
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain JSON-serializable values.") from exc
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds the maximum size of {MAX_METADATA_BYTES} bytes.")
    return dict(metadata)


@dataclass(frozen=True)
class ArtifactExpectation:
    path: str
    sha256: str | None = None
    min_size: int | None = None
    max_size: int | None = None

    def __post_init__(self) -> None:
        path = _bounded_string(self.path, "artifact path", 500)
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("artifact path must be relative and cannot contain '..'.")
        if self.sha256 is not None:
            digest = _bounded_string(self.sha256, "artifact sha256", 64)
            if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
                raise ValueError("artifact sha256 must be a 64-character hexadecimal digest.")
        for name, value in (("min_size", self.min_size), ("max_size", self.max_size)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.min_size is not None and self.max_size is not None and self.min_size > self.max_size:
            raise ValueError("min_size cannot exceed max_size.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactExpectation":
        if not isinstance(payload, dict):
            raise ValueError("artifact expectations must be dictionaries.")
        return cls(
            path=payload.get("path", ""),
            sha256=payload.get("sha256"),
            min_size=payload.get("min_size"),
            max_size=payload.get("max_size"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "min_size": self.min_size,
            "max_size": self.max_size,
        }


@dataclass(frozen=True)
class AgentTaskSpec:
    task_id: str
    expected_outcome: str
    expected_artifacts: tuple[ArtifactExpectation, ...] = ()
    allowed_tools: frozenset[str] | None = None
    max_retries: int = 0

    def __post_init__(self) -> None:
        _bounded_string(self.task_id, "task_id", MAX_TASK_ID_LENGTH)
        _bounded_string(self.expected_outcome, "expected_outcome", MAX_OUTCOME_LENGTH)
        if len(self.expected_artifacts) > MAX_ARTIFACTS:
            raise ValueError(f"expected_artifacts cannot contain more than {MAX_ARTIFACTS} items.")
        if self.allowed_tools is not None:
            for tool in self.allowed_tools:
                _bounded_string(tool, "allowed tool", MAX_TOOL_NAME_LENGTH)
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentTaskSpec":
        if not isinstance(payload, dict):
            raise ValueError("task specification must be a dictionary.")
        artifacts = payload.get("expected_artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError("expected_artifacts must be a list.")
        tools = payload.get("allowed_tools")
        if tools is not None and (not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools)):
            raise ValueError("allowed_tools must be a list of strings.")
        return cls(
            task_id=payload.get("task_id", ""),
            expected_outcome=payload.get("expected_outcome", ""),
            expected_artifacts=tuple(ArtifactExpectation.from_dict(item) for item in artifacts),
            allowed_tools=frozenset(tools) if tools is not None else None,
            max_retries=payload.get("max_retries", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "expected_outcome": self.expected_outcome,
            "expected_artifacts": [artifact.to_dict() for artifact in self.expected_artifacts],
            "allowed_tools": sorted(self.allowed_tools) if self.allowed_tools is not None else None,
            "max_retries": self.max_retries,
        }


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    event_type: str
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    duration_ms: int | None = None

    def __post_init__(self) -> None:
        _bounded_string(self.event_id, "event_id", MAX_EVENT_ID_LENGTH)
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported execution event type: {self.event_type!r}.")
        if self.tool_name is not None:
            _bounded_string(self.tool_name, "tool_name", MAX_TOOL_NAME_LENGTH)
        _validate_metadata(self.metadata)
        if self.timestamp is not None:
            _bounded_string(self.timestamp, "timestamp", 100)
        if self.duration_ms is not None and (
            isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int) or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be a non-negative integer.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionEvent":
        if not isinstance(payload, dict):
            raise ValueError("execution events must be dictionaries.")
        return cls(
            event_id=payload.get("event_id", ""),
            event_type=payload.get("event_type", ""),
            tool_name=payload.get("tool_name"),
            metadata=payload.get("metadata", {}),
            timestamp=payload.get("timestamp"),
            duration_ms=payload.get("duration_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class AgentExecution:
    task_id: str
    events: tuple[ExecutionEvent, ...] = ()
    evidence_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _bounded_string(self.task_id, "task_id", MAX_TASK_ID_LENGTH)
        if len(self.events) > MAX_EVENTS:
            raise ValueError(f"events cannot contain more than {MAX_EVENTS} items.")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("execution event identifiers must be unique.")
        for reference in self.evidence_references:
            _bounded_string(reference, "evidence reference", 500)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentExecution":
        if not isinstance(payload, dict):
            raise ValueError("agent execution must be a dictionary.")
        events = payload.get("events", [])
        if not isinstance(events, list):
            raise ValueError("events must be a list.")
        references = payload.get("evidence_references", [])
        if not isinstance(references, list):
            raise ValueError("evidence_references must be a list.")
        return cls(
            task_id=payload.get("task_id", ""),
            events=tuple(ExecutionEvent.from_dict(event) for event in events),
            evidence_references=tuple(references),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self.events],
            "evidence_references": list(self.evidence_references),
        }


@dataclass(frozen=True)
class AgentAssuranceResult:
    task_id: str
    status: str
    checks: tuple[dict[str, Any], ...]
    passed_checks: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    observed_failures: tuple[str, ...] = ()
    retry_count: int = 0
    evidence_references: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"verified", AssessmentStatus.FAILED, "insufficient_evidence"}:
            raise ValueError(f"Unsupported assurance status: {self.status!r}.")
        for check in self.checks:
            check_status = check.get("status")
            if check_status is not None and check_status not in CHECK_STATUSES:
                raise ValueError(f"Unsupported assurance check status: {check_status!r}.")

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "verified": self.verified,
            "checks": [dict(check) for check in self.checks],
            "passed_checks": list(self.passed_checks),
            "failed_checks": list(self.failed_checks),
            "reasons": list(self.reasons),
            "missing_evidence": list(self.missing_evidence),
            "observed_failures": list(self.observed_failures),
            "retry_count": self.retry_count,
            "evidence_references": list(self.evidence_references),
            "limitations": list(self.limitations),
        }


def _check_artifact(root: Path, expectation: ArtifactExpectation) -> tuple[bool, str]:
    candidate = (root / expectation.path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False, f"artifact escapes the configured root: {expectation.path}"
    if not candidate.is_file():
        return False, f"artifact is missing: {expectation.path}"
    size = candidate.stat().st_size
    if expectation.min_size is not None and size < expectation.min_size:
        return False, f"artifact is smaller than required: {expectation.path}"
    if expectation.max_size is not None and size > expectation.max_size:
        return False, f"artifact is larger than allowed: {expectation.path}"
    if expectation.sha256 is not None:
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expectation.sha256.lower():
            return False, f"artifact hash does not match: {expectation.path}"
    return True, f"artifact verified: {expectation.path}"


def verify_agent_execution(
    task: AgentTaskSpec | dict[str, Any],
    execution: AgentExecution | dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> AgentAssuranceResult:
    task_spec = AgentTaskSpec.from_dict(task) if isinstance(task, dict) else task
    trace = AgentExecution.from_dict(execution) if isinstance(execution, dict) else execution
    if trace.task_id != task_spec.task_id:
        raise ValueError("task_id must match between the task specification and execution.")

    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    failures: list[str] = []
    limitations = [
        "Agent success claims are not treated as independent verification.",
        "Only configured filesystem artifact checks are independently verified.",
    ]
    if not trace.events:
        missing.append("execution events")

    retries = sum(event.event_type == "retry" for event in trace.events)
    checks.append({
        "name": "execution_events",
        "passed": bool(trace.events),
        "status": "passed" if trace.events else "not_run",
        "reason": "Events were recorded." if trace.events else "No execution events were provided.",
    })
    checks.append({
        "name": "retry_limit",
        "passed": retries <= task_spec.max_retries,
        "status": "passed" if retries <= task_spec.max_retries else "failed",
        "observed": retries,
        "allowed": task_spec.max_retries,
    })
    if retries > task_spec.max_retries:
        failures.append(f"retry limit exceeded: observed {retries}, allowed {task_spec.max_retries}")

    observed_failures = tuple(
        str(event.metadata.get("reason") or event.event_id)
        for event in trace.events
        if event.event_type == "failure"
    )
    checks.append({
        "name": "execution_failures",
        "passed": not observed_failures,
        "status": "failed" if observed_failures else "passed",
        "observed": list(observed_failures),
    })
    failures.extend(f"execution failure observed: {reason}" for reason in observed_failures)

    tools = [event.tool_name for event in trace.events if event.event_type == "tool_invocation" and event.tool_name]
    forbidden = sorted({tool for tool in tools if task_spec.allowed_tools is not None and tool not in task_spec.allowed_tools})
    checks.append({
        "name": "tool_allowlist",
        "passed": not forbidden,
        "status": "failed" if forbidden else "passed",
        "forbidden": forbidden,
    })
    failures.extend(f"forbidden tool invocation: {tool}" for tool in forbidden)

    completions = [event for event in trace.events if event.event_type == "completion"]
    checks.append({
        "name": "completion_event",
        "passed": bool(completions),
        "status": "passed" if completions else "not_run",
        "observed": len(completions),
    })
    if not completions:
        missing.append("completion event")

    if task_spec.expected_artifacts:
        if artifact_root is None:
            missing.append("artifact root for expected artifact verification")
            checks.append({
                "name": "artifacts",
                "passed": False,
                "status": "not_run",
                "reason": "No artifact root was provided.",
            })
        else:
            root = Path(artifact_root).resolve()
            artifact_results = []
            for expectation in task_spec.expected_artifacts:
                passed, reason = _check_artifact(root, expectation)
                artifact_results.append({"path": expectation.path, "passed": passed, "reason": reason})
                if not passed:
                    failures.append(reason)
            artifacts_passed = not any(not item["passed"] for item in artifact_results)
            checks.append({
                "name": "artifacts",
                "passed": artifacts_passed,
                "status": "passed" if artifacts_passed else "failed",
                "items": artifact_results,
            })
    else:
        missing.append("independent outcome evidence")
        checks.append({
            "name": "artifacts",
            "passed": False,
            "status": "indeterminate",
            "reason": "No expected artifact checks were configured.",
        })

    passed_checks = tuple(check["name"] for check in checks if check.get("passed") is True)
    failed_checks = tuple(check["name"] for check in checks if check.get("passed") is False)
    if not trace.events:
        status = "insufficient_evidence"
    elif failures:
        status = AssessmentStatus.FAILED
    elif missing:
        status = "insufficient_evidence"
    else:
        status = "verified"
    return AgentAssuranceResult(
        task_id=task_spec.task_id,
        status=status,
        checks=tuple(checks),
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        reasons=tuple(dict.fromkeys([*failures, *missing])),
        missing_evidence=tuple(dict.fromkeys(missing)),
        observed_failures=observed_failures,
        retry_count=retries,
        evidence_references=trace.evidence_references,
        limitations=tuple(limitations),
    )


__all__ = [
    "AgentAssuranceResult",
    "AgentExecution",
    "AgentTaskSpec",
    "ArtifactExpectation",
    "ExecutionEvent",
    "EVENT_TYPES",
    "verify_agent_execution",
]
