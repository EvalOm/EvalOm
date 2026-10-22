import json
from pathlib import Path

from fastapi.testclient import TestClient

from evalom.agent_assurance import (
    AgentExecution,
    AgentTaskSpec,
    ArtifactExpectation,
    ExecutionEvent,
    verify_agent_execution,
)
from evalom.api import create_app
from evalom.dataset import EvaluationDataset
from evalom.evaluators import ExactMatchEvaluator, RetrievalRecallEvaluator
from evalom.policy import ReleasePolicy
from evalom.reports import generate_json_report, validate_report, write_json_report
from evalom.runner import EvaluationRunner
from evalom.verification import VerificationEngine


def test_rag_workflow_round_trips_report_and_verification(tmp_path: Path):
    dataset = EvaluationDataset(
        [
            {
                "id": "refund",
                "input": "What is the refund window?",
                "expected_answer": "30 days",
                "expected_context": ["refund policy"],
            },
            {
                "id": "shipping",
                "input": "How long does shipping take?",
                "expected_answer": "5 days",
                "expected_context": ["shipping policy"],
            },
        ],
        name="support",
        version="2026-10",
    )
    answers = {"What is the refund window?": "30 days", "How long does shipping take?": "5 days"}
    runner = EvaluationRunner(dataset, [ExactMatchEvaluator(), RetrievalRecallEvaluator()])
    result = runner.run(
        lambda question: {
            "answer": answers[question],
            "contexts": ["refund policy"] if "refund" in question else ["shipping policy"],
        },
        metadata={"run_id": "run-support-1"},
    )

    report_path = tmp_path / "support-report.json"
    write_json_report(result, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert validate_report(payload)["dataset"]["name"] == "support"
    assert generate_json_report(result) == payload

    verification = VerificationEngine(minimum_sample_size=1).verify(result, dataset=dataset)
    assert verification.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert verification.metadata["dataset_hash"] == dataset.fingerprint()


def test_missing_required_evidence_blocks_release():
    policy = ReleasePolicy(
        min_confidence_score=0,
        min_dataset_quality=0,
        min_evaluator_reliability=0,
        min_statistical_confidence=0,
        max_risk_level="HIGH",
        required_evidence=["verified_artifact"],
    )
    decision = policy.evaluate(
        {
            "risk_level": "LOW",
            "confidence_score": 100,
            "dataset_quality": 1,
            "evaluator_reliability": 1,
            "statistical_confidence": 1,
            "confidence_interval": {"lower": 1, "upper": 1},
            "evidence": {},
        }
    )
    assert decision["decision"] == "BLOCK"
    assert any("verified_artifact" in failure for failure in decision["failures"])


def test_agent_workflow_requires_independent_artifact_evidence(tmp_path: Path):
    task = AgentTaskSpec(
        task_id="generate-config",
        expected_outcome="generate a valid configuration",
        expected_artifacts=(ArtifactExpectation("config.json"),),
        allowed_tools=frozenset({"writer"}),
    )
    execution = AgentExecution(
        task_id="generate-config",
        events=(
            ExecutionEvent("invoke", "tool_invocation", tool_name="writer"),
            ExecutionEvent("complete", "completion", metadata={"claim": "success"}),
        ),
        evidence_references=("trace-config",),
    )
    result_without_artifact = verify_agent_execution(task, execution, artifact_root=tmp_path)
    assert result_without_artifact.status == "failed"

    (tmp_path / "config.json").write_text('{"enabled": true}', encoding="utf-8")
    result = verify_agent_execution(task, execution, artifact_root=tmp_path)
    assert result.status == "verified"
    assert result.evidence_references == ("trace-config",)


def test_http_report_persistence_and_workspace_isolation(tmp_path: Path):
    client = TestClient(
        create_app(
            database_path=tmp_path / "e2e.db",
            bootstrap_api_key="e2e-bootstrap-key-with-at-least-32-chars",
            bootstrap_workspace_id="workspace-a",
        )
    )
    admin = {"Authorization": "Bearer e2e-bootstrap-key-with-at-least-32-chars"}
    created = client.post(
        "/api/v1/verification",
        headers=admin,
        json={
            "dataset_name": "api-e2e",
            "records": [
                {
                    "sample_id": "1",
                    "input": "question",
                    "output": "answer",
                    "score": 1.0,
                }
            ],
        },
    )
    assert created.status_code == 200
    report_id = created.json()["report_id"]
    report = client.get(f"/api/v1/verification/{report_id}", headers=admin)
    assert report.status_code == 200
    assert report.json()["workspace_id"] == "workspace-a"

    other = client.post(
        "/api/v1/workspaces",
        headers=admin,
        json={"workspace_id": "workspace-b"},
    )
    other_admin = {"Authorization": f"Bearer {other.json()['admin_api_key']['api_key']}"}
    assert client.get(f"/api/v1/verification/{report_id}", headers=other_admin).status_code == 404
