import hashlib
from pathlib import Path

import pytest

from evalom.agent_assurance import (
    AgentExecution,
    AgentTaskSpec,
    ArtifactExpectation,
    ExecutionEvent,
    verify_agent_execution,
)


def _task(path: str = "result.txt", *, max_retries: int = 1) -> AgentTaskSpec:
    return AgentTaskSpec(
        task_id="task-1",
        expected_outcome="produce the requested result",
        expected_artifacts=(ArtifactExpectation(path=path),),
        allowed_tools=frozenset({"writer"}),
        max_retries=max_retries,
    )


def _execution(*events: ExecutionEvent) -> AgentExecution:
    return AgentExecution(task_id="task-1", events=tuple(events), evidence_references=("trace-1",))


def test_independently_verified_task(tmp_path: Path):
    (tmp_path / "result.txt").write_text("done", encoding="utf-8")
    result = verify_agent_execution(
        _task(),
        _execution(
            ExecutionEvent("1", "tool_invocation", tool_name="writer"),
            ExecutionEvent("2", "tool_result", tool_name="writer"),
            ExecutionEvent("3", "completion", metadata={"claim": "success"}),
        ),
        artifact_root=tmp_path,
    )
    assert result.status == "verified"
    assert result.verified is True
    assert result.evidence_references == ("trace-1",)


def test_success_claim_without_verification_is_insufficient():
    result = verify_agent_execution(
        AgentTaskSpec("task-1", "produce the result"),
        _execution(ExecutionEvent("1", "completion", metadata={"claim": "success"})),
    )
    assert result.status == "insufficient_evidence"
    assert "independent outcome evidence" in result.missing_evidence
    assert next(check for check in result.checks if check["name"] == "artifacts")["status"] == "indeterminate"


def test_missing_expected_artifact_is_failed(tmp_path: Path):
    result = verify_agent_execution(_task(), _execution(ExecutionEvent("1", "completion")), artifact_root=tmp_path)
    assert result.status == "failed"
    assert any("artifact is missing" in failure for failure in result.observed_failures) is False
    assert any("artifact is missing" in check["items"][0]["reason"] for check in result.checks if check["name"] == "artifacts")


def test_detected_failure_is_failed(tmp_path: Path):
    (tmp_path / "result.txt").write_text("done", encoding="utf-8")
    result = verify_agent_execution(
        _task(),
        _execution(ExecutionEvent("1", "failure", metadata={"reason": "write failed"})),
        artifact_root=tmp_path,
    )
    assert result.status == "failed"
    assert result.observed_failures == ("write failed",)


def test_retries_within_limit_are_recorded(tmp_path: Path):
    (tmp_path / "result.txt").write_text("done", encoding="utf-8")
    result = verify_agent_execution(
        _task(max_retries=1),
        _execution(ExecutionEvent("1", "retry"), ExecutionEvent("2", "completion")),
        artifact_root=tmp_path,
    )
    assert result.status == "verified"
    assert result.retry_count == 1


def test_retry_limit_exceeded_is_failed(tmp_path: Path):
    (tmp_path / "result.txt").write_text("done", encoding="utf-8")
    result = verify_agent_execution(
        _task(max_retries=1),
        _execution(ExecutionEvent("1", "retry"), ExecutionEvent("2", "retry"), ExecutionEvent("3", "completion")),
        artifact_root=tmp_path,
    )
    assert result.status == "failed"
    assert "retry_limit" in result.failed_checks
    assert result.retry_count == 2


def test_forbidden_tool_invocation_is_failed(tmp_path: Path):
    (tmp_path / "result.txt").write_text("done", encoding="utf-8")
    result = verify_agent_execution(
        _task(),
        _execution(ExecutionEvent("1", "tool_invocation", tool_name="shell"), ExecutionEvent("2", "completion")),
        artifact_root=tmp_path,
    )
    assert result.status == "failed"
    assert "tool_allowlist" in result.failed_checks


def test_empty_trace_is_insufficient(tmp_path: Path):
    result = verify_agent_execution(_task(), AgentExecution("task-1"), artifact_root=tmp_path)
    assert result.status == "insufficient_evidence"
    assert "execution events" in result.missing_evidence
    assert next(check for check in result.checks if check["name"] == "execution_events")["status"] == "not_run"


def test_failed_check_status_is_distinct_from_not_run(tmp_path: Path):
    result = verify_agent_execution(
        _task(),
        _execution(ExecutionEvent("1", "failure", metadata={"reason": "write failed"})),
        artifact_root=tmp_path,
    )
    statuses = {check["name"]: check["status"] for check in result.checks}
    assert statuses["execution_failures"] == "failed"
    assert statuses["completion_event"] == "not_run"


def test_malformed_input_and_invalid_event_types_are_rejected():
    with pytest.raises(ValueError):
        ExecutionEvent("1", "unknown")
    with pytest.raises(ValueError):
        AgentExecution.from_dict({"task_id": "task-1", "events": [{"event_id": "1", "event_type": "completion"}, {"event_id": "1", "event_type": "retry"}]})
    with pytest.raises(ValueError):
        AgentTaskSpec.from_dict({"task_id": "task-1", "expected_outcome": "x", "expected_artifacts": "not-a-list"})


def test_artifact_hash_and_path_safety(tmp_path: Path):
    content = b"verified"
    (tmp_path / "result.txt").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    task = AgentTaskSpec(
        "task-1",
        "produce result",
        expected_artifacts=(ArtifactExpectation("result.txt", sha256=digest, min_size=1),),
    )
    result = verify_agent_execution(task, _execution(ExecutionEvent("1", "completion")), artifact_root=tmp_path)
    assert result.status == "verified"
    with pytest.raises(ValueError):
        ArtifactExpectation("../outside.txt")
