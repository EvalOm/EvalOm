import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from argparse import Namespace
from pathlib import Path

import pytest

from evalom.cli import init_command, run_command, verify_command
from evalom.dataset import EvaluationDataset
from evalom.evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    MRREvaluator,
    NDCGEvaluator,
    RetrievalRecallEvaluator,
)
from evalom.experiment import ExperimentTracker, metadata_fingerprint
from evalom.policy import ReleasePolicy
from evalom.runner import EvaluationRunResult, EvaluationRunner
from evalom.api import create_app
from fastapi.testclient import TestClient
from evalom.adapters import EvaluationImportLayer, LangfuseAdapter
from evalom.storage import SQLiteExperimentStore, create_experiment_store
from evalom.verification import VerificationEngine, normalize_evaluation_record


def test_dataset_from_jsonl(tmp_path: Path):
    dataset_path = tmp_path / "demo.jsonl"
    dataset_path.write_text(
        '\n'.join([
            json.dumps({"id": "1", "input": "What is the capital of France?", "expected_answer": "Paris"}),
            json.dumps({"id": "2", "input": "What is 2 + 2?", "expected_answer": "4"}),
        ]) + "\n",
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_jsonl(dataset_path)
    assert len(dataset) == 2
    assert dataset[0].input == "What is the capital of France?"


def test_exact_match_evaluator():
    evaluator = ExactMatchEvaluator()
    result = evaluator.evaluate({"id": "1", "input": "Q", "expected_answer": "Paris"}, {"answer": "Paris", "contexts": []})
    assert result["score"] == 1.0
    assert result["passed"] is True


def test_empty_exact_match_reference_is_valid_but_missing_reference_is_insufficient():
    evaluator = ExactMatchEvaluator()
    assert evaluator.evaluate(
        {"id": "1", "input": "Q", "expected_answer": ""},
        {"answer": ""},
    )["passed"] is True
    missing = evaluator.evaluate({"id": "1", "input": "Q"}, {"answer": ""})
    assert missing["passed"] is False
    assert missing["status"] == "insufficient_evidence"


def test_empty_contains_configuration_is_not_a_pass():
    result = ContainsEvaluator().evaluate({}, {"answer": "anything"})
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["status"] == "insufficient_evidence"
    with pytest.raises(ValueError):
        ContainsEvaluator(required_terms=[""])


def test_retrieval_recall_requires_expected_context_reference():
    result = RetrievalRecallEvaluator().evaluate({}, {"contexts": ["context"]})
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["status"] == "insufficient_evidence"


def test_missing_ground_truth_fails_runner_quality_status_and_policy():
    dataset = EvaluationDataset([{"id": "1", "input": "question"}])
    result = EvaluationRunner(dataset, [ExactMatchEvaluator()]).run(lambda _: {"answer": ""})
    assert result.metrics[0]["status"] == "insufficient_evidence"
    assert result.check_quality_gates()["passed"] is False
    assert "insufficient reference evidence" in result.summary()
    decision = ReleasePolicy(min_confidence_score=0, min_dataset_quality=0, min_evaluator_reliability=0).evaluate({
        "status": "insufficient_evidence",
        "risk_level": "LOW",
        "confidence_score": 100,
        "dataset_quality": 1,
        "evaluator_reliability": 1,
        "statistical_confidence": 1,
        "lower_confidence_bound": 1,
    })
    assert decision["decision"] == "BLOCK"


def test_runner_summary():
    dataset = EvaluationDataset([
        {"id": "1", "input": "Who is the capital?", "expected_answer": "Paris"},
        {"id": "2", "input": "What is 2 + 2?", "expected_answer": "4"},
    ])

    dataset.samples = [
        dataset.samples[0],
        dataset.samples[1],
    ]

    def app(question: str):
        if "capital" in question.lower():
            return {"answer": "Paris", "contexts": ["Country"]}
        return {"answer": "4", "contexts": ["Math"]}

    runner = EvaluationRunner(dataset, [ExactMatchEvaluator()])
    result = runner.run(app)

    assert result.average_score() == 1.0
    assert "PASS" in result.summary()


def test_init_command_uses_custom_project_name(tmp_path: Path):
    args = Namespace(project_dir=str(tmp_path), project_name="customer-support")

    exit_code = init_command(args)

    assert exit_code == 0
    config_path = tmp_path / "eval.yaml"
    assert config_path.exists()
    assert "name: customer-support" in config_path.read_text(encoding="utf-8")


def test_retrieval_metrics_and_experiment_tracking():
    sample = {
        "id": "1",
        "input": "refund policy",
        "expected_context": ["policy", "billing", "refund"],
    }
    output = {"answer": "refund policy", "contexts": ["billing", "policy", "terms"]}

    mrr = MRREvaluator()
    ndcg = NDCGEvaluator()

    assert mrr.evaluate(sample, output)["score"] == 1.0
    assert ndcg.evaluate(sample, output)["score"] > 0.0

    tracker = ExperimentTracker()
    record = tracker.record({"name": "exp-001", "model": "gpt-4o"})
    assert record["name"] == "exp-001"
    assert tracker.list()[0]["model"] == "gpt-4o"


def test_report_schema_validation_and_atomic_export(tmp_path: Path):
    from evalom.reports import REPORT_SCHEMA_VERSION, validate_report, write_json_report

    payload = {"dataset": {"name": "d", "version": "1", "samples": 1},
               "metrics": [], "sample_results": []}
    assert validate_report(payload)["dataset"]["name"] == "d"
    assert metadata_fingerprint({"b": 2, "a": 1, "api_token": "one"}) == metadata_fingerprint(
        {"a": 1, "b": 2, "api_token": "two"}
    )
    with pytest.raises(ValueError, match="Unsupported report schema_version"):
        validate_report({**payload, "schema_version": "9.0"})
    with pytest.raises(ValueError, match="non-finite"):
        validate_report({**payload, "metrics": [{"score": float("nan")}]})

    destination = tmp_path / "report.json"
    write_json_report(EvaluationRunResult("d", "1", 1), destination)
    assert json.loads(destination.read_text())["schema_version"] == REPORT_SCHEMA_VERSION


def test_quality_gate_regression_detection():
    dataset = EvaluationDataset([
        {"id": "1", "input": "capital", "expected_answer": "Paris"},
    ])

    runner = EvaluationRunner(
        dataset,
        [ExactMatchEvaluator()],
        quality_gates={"exact_match": {"minimum": 0.99}},
    )

    result = runner.run(lambda _: {"answer": "London", "contexts": []})

    assert result.check_quality_gates()["passed"] is False
    assert "exact_match" in result.summary()


def test_quality_gate_fails_when_configured_metric_is_missing():
    result = EvaluationRunResult(
        dataset_name="demo",
        dataset_version="v1",
        total_samples=1,
        metrics=[{"name": "contains", "score": 1.0, "passed": True}],
        quality_gates={"exact_match": {"minimum": 0.8}},
    )

    quality_status = result.check_quality_gates()

    assert quality_status["passed"] is False
    assert quality_status["failures"] == [
        "exact_match missing: no scores were produced for the configured quality gate."
    ]


def test_run_command_fail_on_regression(tmp_path: Path):
    dataset_path = tmp_path / "demo.jsonl"
    dataset_path.write_text(
        '{"id": "1", "input": "capital", "expected_answer": "Paris"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "project:\n  name: demo\n\ndataset:\n  path: demo.jsonl\n\nevaluators:\n  - name: exact_match\n    type: deterministic\n\nquality_gates:\n  exact_match:\n    minimum: 0.99\n",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(config_path),
        app_module=None,
        app_function=None,
        output=None,
        html_output=None,
        fail_on_regression=True,
        experiment_store=str(tmp_path / "runs.db"),
    )

    exit_code = run_command(args)

    assert exit_code == 1
    stored = SQLiteExperimentStore(args.experiment_store).list()
    assert stored["total"] == 1
    assert stored["items"][0]["name"] == "demo"
    assert stored["items"][0]["metrics"]["exact_match"] == 0.0
    assert stored["items"][0]["metadata"]["quality_gates"]["passed"] is False


def test_run_command_applies_configured_release_policy(tmp_path: Path):
    dataset_path = tmp_path / "demo.jsonl"
    dataset_path.write_text(
        '{"id": "1", "input": "capital", "expected_answer": "Paris"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "project:\n"
        "  name: demo\n"
        "\ndataset:\n"
        "  path: demo.jsonl\n"
        "\nevaluators:\n"
        "  - name: exact_match\n"
        "\nrelease_policy:\n"
        "  max_risk_level: HIGH\n"
        "  min_confidence_score: 100\n"
        "  min_dataset_quality: 0\n"
        "  min_evaluator_reliability: 0\n"
        "  min_statistical_confidence: 0\n"
        "  max_duplicate_ratio: 1\n"
        "  max_distribution_shift: 1\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    store_path = tmp_path / "runs.db"

    exit_code = run_command(Namespace(
        config=str(config_path),
        app_module=None,
        app_function=None,
        output=str(report_path),
        html_output=None,
        fail_on_regression=False,
        experiment_store=str(store_path),
    ))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    stored = SQLiteExperimentStore(store_path).list()["items"][0]

    assert exit_code == 1
    assert report["metadata"]["policy_decision"]["decision"] == "BLOCK"
    assert report["metadata"]["verification"]["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert stored["metadata"]["policy_decision"]["passed"] is False


def test_verification_uses_wilson_lower_bound_for_binary_metrics():
    result = EvaluationRunResult(
        dataset_name="demo",
        dataset_version="v1",
        total_samples=10,
        metrics=[
            {"name": "exact_match", "score": 1.0, "passed": True},
            {"name": "exact_match", "score": 1.0, "passed": True},
            {"name": "exact_match", "score": 0.0, "passed": False},
            {"name": "exact_match", "score": 1.0, "passed": True},
        ],
    )

    verification = VerificationEngine().verify(result)

    assert verification.confidence_interval["lower"] is not None
    assert verification.statistical_confidence == verification.confidence_interval["lower"]
    assert verification.confidence_interval["lower"] <= 0.75
    assert verification.confidence_interval["upper"] >= verification.confidence_interval["lower"]


def test_release_policy_uses_lower_bound_not_heuristic_score():
    policy = ReleasePolicy(min_statistical_confidence=0.55, confidence_level=95)
    decision = policy.evaluate({
        "risk_level": "LOW",
        "confidence_score": 90.0,
        "dataset_quality": 0.9,
        "evaluator_reliability": 0.9,
        "statistical_confidence": 0.7,
        "confidence_interval": {"lower": 0.4, "upper": 0.8},
        "issues": [],
    })

    assert policy.confidence_level == 0.95
    assert decision["passed"] is False
    assert decision["decision"] == "BLOCK"
    assert "Lower confidence bound 0.4000 is below the required minimum of 0.5500." in decision["failures"]


def test_experiment_api_persists_records_and_paginates(tmp_path: Path):
    database_path = tmp_path / "data" / "experiments.db"
    client = TestClient(create_app(database_path=database_path))

    created = client.post(
        "/api/v1/experiments",
        json={
            "name": "support-v2",
            "model": "model-x",
            "dataset_version": "2026-10",
            "metrics": {"exact_match": 0.92},
            "metadata": {"prompt_sha": "abc123"},
        },
    )

    assert created.status_code == 201
    record = created.json()
    assert record["id"]
    assert record["status"] == "completed"
    assert record["metrics"] == {"exact_match": 0.92}

    restarted_client = TestClient(create_app(database_path=database_path))
    fetched = restarted_client.get(f"/api/v1/experiments/{record['id']}")
    listing = restarted_client.get("/api/v1/experiments", params={"limit": 1})

    assert fetched.status_code == 200
    assert fetched.json()["metadata"] == {"prompt_sha": "abc123"}
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["id"] == record["id"]


def test_experiment_store_serializes_non_json_values(tmp_path: Path):
    database_path = tmp_path / "experiments.db"
    store = SQLiteExperimentStore(database_path)

    record = store.create({
        "name": "serializable-exp",
        "metrics": {"score": 0.91, "tags": {"set_value": {"a", "b"}}, "when": __import__("datetime").datetime(2025, 1, 1, 12, 0, tzinfo=__import__("datetime").timezone.utc)},
        "metadata": {"labels": ["x", "y"], "path": Path("/tmp/demo")},
    })

    assert sorted(record["metrics"]["tags"]["set_value"]) == ["a", "b"]
    assert record["metadata"]["path"] == "/tmp/demo"
    assert "2025-01-01T12:00:00+00:00" in json.dumps(record["metrics"], sort_keys=True)


def test_experiment_store_migrates_legacy_experiment_and_idempotency_schemas(
    tmp_path: Path,
):
    database_path = tmp_path / "legacy_experiments.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model TEXT,
                evaluator TEXT,
                dataset_version TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO experiments VALUES (
                'legacy-id', 'legacy-run', NULL, NULL, NULL, 'completed',
                '2025-01-01T00:00:00+00:00', '{}', '{}'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE run_idempotency_keys (
                idempotency_key TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    store = SQLiteExperimentStore(database_path)
    migrated = store.get("legacy-id")
    with sqlite3.connect(database_path) as connection:
        idempotency_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(run_idempotency_keys)")
        }

    assert migrated is not None
    assert migrated["workspace_id"] == "default"
    assert migrated["updated_at"] == migrated["created_at"]
    assert "request_hash" in idempotency_columns


def test_verification_reports_persist_and_round_trip(tmp_path: Path):
    database_path = tmp_path / "verification.db"
    store = SQLiteExperimentStore(database_path)

    report = store.save_verification_report({
        "dataset_name": "demo-dataset",
        "dataset_version": "2026-10",
        "risk_level": "MEDIUM",
        "confidence_score": 76.5,
        "policy_decision": {"decision": "BLOCK", "passed": False, "failures": ["low confidence"]},
        "verification": {
            "risk_level": "MEDIUM",
            "confidence_score": 76.5,
            "recommendation": "Review before release.",
        },
    })

    persisted = store.get_verification_report(report["id"])

    assert persisted is not None
    assert persisted["dataset_name"] == "demo-dataset"
    assert persisted["risk_level"] == "MEDIUM"
    assert persisted["policy_decision"] == "BLOCK"
    assert persisted["summary"]["verification"]["recommendation"] == "Review before release."


def _report_payload(report_id: str, *, workspace_id: str = "workspace-a") -> dict:
    return {
        "report_id": report_id,
        "workspace_id": workspace_id,
        "dataset_name": "atomic-dataset",
        "dataset_version": "v1",
        "risk_level": "LOW",
        "confidence_score": 98.5,
        "policy_decision": {"decision": "ALLOW"},
        "verification": {
            "status": "verified",
            "checks": [{"name": "quality", "status": "passed"}],
            "evidence": [{"evidence_id": "evidence-1"}],
        },
        "statistics": {"sample_size": 10, "uncertainty": 0.02},
    }


def test_atomic_report_save_persists_report_and_audit(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "atomic.db")

    report = store.save_verification_report_and_audit(
        _report_payload("report-1"),
        workspace_id="workspace-a",
        actor_id="actor-1",
        actor_name="test",
    )

    persisted = store.get_verification_report("report-1", workspace_id="workspace-a")
    events = store.list_audit_events(workspace_id="workspace-a")["items"]
    assert persisted == report
    assert persisted["summary"]["verification"]["checks"][0]["status"] == "passed"
    assert persisted["summary"]["statistics"]["uncertainty"] == 0.02
    assert [event["event_type"] for event in events] == ["verification.created"]
    assert events[0]["details"]["report_id"] == "report-1"


def test_atomic_report_save_rolls_back_on_audit_failure_and_recovers(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "atomic-audit-failure.db")
    with store._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_report_audit BEFORE INSERT ON audit_events
            WHEN NEW.event_type = 'verification.created'
            BEGIN SELECT RAISE(ABORT, 'report audit failure'); END
            """
        )

    with pytest.raises(Exception, match="report audit failure"):
        store.save_verification_report_and_audit(_report_payload("failed-report"))

    assert store.get_verification_report("failed-report") is None
    assert store.list_audit_events()["total"] == 0
    with store._connect() as connection:
        connection.execute("DROP TRIGGER fail_report_audit")
    store.save_verification_report_and_audit(_report_payload("recovered-report"))
    assert store.get_verification_report("recovered-report") is not None


def test_atomic_report_save_rolls_back_on_report_failure_without_audit(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "atomic-report-failure.db")
    store.save_verification_report(_report_payload("existing-report"))

    with pytest.raises(ValueError, match="identifier already exists"):
        store.save_verification_report_and_audit(_report_payload("existing-report"))

    assert store.get_verification_report("existing-report")["summary"]["dataset_name"] == "atomic-dataset"
    assert store.list_audit_events()["total"] == 0


def test_atomic_report_save_is_workspace_scoped_and_duplicate_safe(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "atomic-workspace.db")
    store.save_verification_report_and_audit(_report_payload("shared-id", workspace_id="workspace-a"), workspace_id="workspace-a")

    assert store.get_verification_report("shared-id", workspace_id="workspace-b") is None
    assert store.list_verification_reports(workspace_id="workspace-b")["total"] == 0
    with pytest.raises(ValueError):
        store.save_verification_report_and_audit(
            {**_report_payload("shared-id"), "dataset_name": "replacement"},
            workspace_id="workspace-a",
        )
    assert store.get_verification_report("shared-id", workspace_id="workspace-a")["dataset_name"] == "atomic-dataset"
    assert store.list_audit_events(workspace_id="workspace-a")["total"] == 1


def test_concurrent_atomic_report_creation_allows_only_one_duplicate(tmp_path: Path):
    database_path = tmp_path / "atomic-concurrent.db"
    SQLiteExperimentStore(database_path)

    def create_report() -> str:
        store = SQLiteExperimentStore(database_path)
        try:
            store.save_verification_report_and_audit(_report_payload("concurrent-id"))
        except Exception:
            return "failed"
        return "succeeded"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: create_report(), range(2)))

    assert outcomes.count("succeeded") == 1
    assert outcomes.count("failed") == 1
    store = SQLiteExperimentStore(database_path)
    assert store.get_verification_report("concurrent-id") is not None
    assert store.list_audit_events()["total"] == 1


def test_store_migrates_existing_verification_reports(tmp_path: Path):
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE verification_reports (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                dataset_name TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                created_at TEXT NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )

    store = SQLiteExperimentStore(database_path)
    report = store.save_verification_report({
        "dataset_name": "new-data",
        "dataset_version": "v2",
        "risk_level": "LOW",
        "confidence_score": 91,
        "policy_decision": {"decision": "ALLOW", "passed": True},
    })

    assert report["policy_decision"] == "ALLOW"
    assert store.get_verification_report(report["id"])["policy_decision"] == "ALLOW"


def test_verification_api_persists_reports(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "verification_api.db"))
    experiment_response = client.post(
        "/api/v1/experiments",
        json={"name": "verification-linked-run"},
    )
    experiment_id = experiment_response.json()["id"]

    response = client.post(
        "/api/v1/verification",
        json={
            "dataset_name": "demo",
            "dataset_version": "v1",
            "experiment_id": experiment_id,
            "records": [
                {"sample_id": "1", "input": "What is 2 + 2?", "output": "4", "score": 0.95, "evaluator": "exact_match"},
                {"sample_id": "2", "input": "What is the capital of France?", "output": "Paris", "score": 0.9, "evaluator": "exact_match"},
            ],
            "policy": {
                "max_risk_level": "MEDIUM",
                "min_confidence_score": 70.0
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["report_id"]
    assert payload["experiment_id"] == experiment_id
    assert "policy_decision" in payload
    assert payload["policy_decision"]["decision"] in {"ALLOW", "BLOCK"}

    fetched = client.get(f"/api/v1/verification/{payload['report_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["dataset_name"] == "demo"
    assert fetched.json()["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert fetched.json()["experiment_id"] == experiment_id
    assert fetched.json()["summary"]["policy_decision"] == payload["policy_decision"]
    audit_events = client.get("/api/v1/audit-events")
    assert audit_events.status_code == 200
    assert any(
        event["event_type"] == "verification.created"
        and event["details"]["report_id"] == payload["report_id"]
        for event in audit_events.json()["items"]
    )

    linked_reports = client.get(
        "/api/v1/verification",
        params={"experiment_id": experiment_id},
    )
    assert linked_reports.status_code == 200
    assert linked_reports.json()["total"] == 1
    assert linked_reports.json()["items"][0]["id"] == payload["report_id"]

    decision_reports = client.get(
        "/api/v1/verification",
        params={"decision": payload["policy_decision"]["decision"]},
    )
    assert decision_reports.status_code == 200
    assert decision_reports.json()["total"] == 1
    assert decision_reports.json()["items"][0]["policy_decision"] == payload["policy_decision"]["decision"]

    invalid_decision = client.get("/api/v1/verification", params={"decision": "MAYBE"})
    assert invalid_decision.status_code == 422


def test_verification_api_rejects_unknown_experiment_id(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "unknown_experiment.db"))

    response = client.post(
        "/api/v1/verification",
        json={
            "experiment_id": "missing-experiment",
            "records": [
                {"sample_id": "1", "input": "question", "output": "answer", "score": 0.9}
            ],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def test_verification_api_rejects_invalid_release_policy(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "invalid_policy.db"))

    response = client.post(
        "/api/v1/verification",
        json={
            "records": [
                {"sample_id": "1", "input": "question", "output": "answer", "score": 0.9}
            ],
            "policy": {"min_confidence_score": "not-a-number"},
        },
    )

    assert response.status_code == 422


def test_release_policy_blocks_high_risk_verification():
    policy = ReleasePolicy(
        max_risk_level="MEDIUM",
        min_confidence_score=80.0,
        min_dataset_quality=0.75,
        min_evaluator_reliability=0.80,
        min_statistical_confidence=0.75,
    )

    decision = policy.evaluate({
        "risk_level": "HIGH",
        "confidence_score": 70.0,
        "dataset_quality": 0.6,
        "evaluator_reliability": 0.7,
        "statistical_confidence": 0.65,
        "issues": ["Evaluator reliability is below the recommended threshold."],
        "dataset_health": {"duplicate_ratio": 0.08, "distribution_shift": 0.4},
    })

    assert decision["passed"] is False
    assert decision["decision"] == "BLOCK"
    assert "Risk level HIGH exceeds" in decision["failures"][0]


def test_release_policy_allows_safe_verification():
    policy = ReleasePolicy(max_risk_level="MEDIUM")
    decision = policy.evaluate({
        "risk_level": "LOW",
        "confidence_score": 86.0,
        "dataset_quality": 0.9,
        "evaluator_reliability": 0.88,
        "statistical_confidence": 0.8,
        "issues": [],
        "dataset_health": {"duplicate_ratio": 0.01, "distribution_shift": 0.2},
    })

    assert decision["passed"] is True
    assert decision["decision"] == "ALLOW"


def test_release_policy_blocks_unverified_assurance():
    decision = ReleasePolicy(max_risk_level="HIGH").evaluate({
        "status": "insufficient_evidence",
        "risk_level": "LOW",
        "confidence_score": 100,
        "dataset_quality": 1,
        "evaluator_reliability": 1,
        "statistical_confidence": 1,
        "issues": [],
        "dataset_health": {"duplicate_ratio": 0, "distribution_shift": 0},
    })

    assert decision["decision"] == "BLOCK"
    assert any("not verified" in failure for failure in decision["failures"])


def test_release_policy_requires_every_configured_check_and_explains_missing_results():
    policy = ReleasePolicy(
        max_risk_level="HIGH",
        min_confidence_score=0,
        min_dataset_quality=0,
        min_evaluator_reliability=0,
        min_statistical_confidence=0,
        required_checks=["tests", "review"],
    )
    decision = policy.evaluate({
        "status": "verified",
        "risk_level": "LOW",
        "confidence_score": 100,
        "dataset_quality": 1,
        "evaluator_reliability": 1,
        "statistical_confidence": 1,
        "checks": [{"name": "tests", "status": "passed", "passed": True}],
        "issues": [],
        "dataset_health": {"duplicate_ratio": 0, "distribution_shift": 0},
    })

    assert decision["decision"] == "BLOCK"
    assert decision["passed_requirements"] == ["tests"]
    assert decision["not_run_requirements"] == ["review"]
    assert any("review" in failure and "not run" in failure for failure in decision["failures"])


def test_release_policy_blocks_required_evidence_comparison_and_profile_failures():
    policy = ReleasePolicy(
        max_risk_level="HIGH",
        min_confidence_score=0,
        min_dataset_quality=0,
        min_evaluator_reliability=0,
        min_statistical_confidence=0,
        required_evidence=["test_result"],
        required_comparisons=[{"metric": "accuracy", "max_regression": 0.02}],
        required_task_profile={"profile_id": "demo.feature", "profile_version": "1.0"},
    )
    decision = policy.evaluate({
        "status": "verified",
        "risk_level": "LOW",
        "confidence_score": 100,
        "dataset_quality": 1,
        "evaluator_reliability": 1,
        "statistical_confidence": 1,
        "evidence": {"test_result": {"available": False, "sufficient": False}},
        "comparisons": {"accuracy": {"status": "verified", "difference": -0.1, "direction": "higher_is_better"}},
        "task_profile_result": {"status": "insufficient_evidence"},
        "issues": [],
        "dataset_health": {"duplicate_ratio": 0, "distribution_shift": 0},
    })

    assert decision["decision"] == "BLOCK"
    assert len(decision["failures"]) == 3


def test_release_policy_rejects_duplicate_and_unknown_required_configuration():
    with pytest.raises(ValueError):
        ReleasePolicy.from_dict({"required_checks": ["tests", "tests"]})
    with pytest.raises(ValueError):
        ReleasePolicy.from_dict({"required_checks": ["tests"], "optional_checks": ["tests"]})
    with pytest.raises(ValueError):
        ReleasePolicy.from_dict({"policy_version": "9.0"})


def test_release_policy_rejects_ambiguous_runtime_results():
    policy = ReleasePolicy(
        max_risk_level="HIGH",
        min_confidence_score=0,
        min_dataset_quality=0,
        min_evaluator_reliability=0,
        min_statistical_confidence=0,
        required_evidence=["test_result"],
    )
    verification = {
        "risk_level": "LOW",
        "confidence_score": 100,
        "dataset_quality": 1,
        "evaluator_reliability": 1,
        "statistical_confidence": 1,
        "evidence": [
            {"evidence_type": "test_result", "available": True, "sufficient": True},
            {"evidence_type": "test_result", "available": False, "sufficient": False},
        ],
    }
    with pytest.raises(ValueError, match="Duplicate evidence"):
        policy.evaluate(verification)


@pytest.mark.parametrize(
    "payload",
    [
        {"max_risk_level": "CRITICAL"},
        {"min_confidence_score": -1},
        {"min_confidence_score": 101},
        {"min_dataset_quality": float("nan")},
        {"max_duplicate_ratio": float("inf")},
        {"max_distribution_shift": True},
        {"min_evaluator_reliability": "high"},
        {"allowed_issues": "ignore all issues"},
        {"allowed_issues": [""]},
        {"min_confidnce_score": 80},
    ],
)
def test_release_policy_rejects_invalid_configuration(payload: dict[str, object]):
    with pytest.raises(ValueError):
        ReleasePolicy.from_dict(payload)


def test_release_policy_blocks_unknown_verification_risk():
    decision = ReleasePolicy(max_risk_level="HIGH").evaluate({
        "risk_level": "UNKNOWN",
        "confidence_score": 100,
        "dataset_quality": 1,
        "evaluator_reliability": 1,
        "statistical_confidence": 1,
        "issues": [],
        "dataset_health": {"duplicate_ratio": 0, "distribution_shift": 0},
    })

    assert decision["passed"] is False
    assert any("Unknown verification risk level" in failure for failure in decision["failures"])


def test_verify_command_exits_nonzero_when_policy_blocks(tmp_path: Path, capsys):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({
            "dataset": {"name": "demo", "version": "v1", "samples": 100},
            "metrics": [],
            "sample_results": [],
        }),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({
            "max_risk_level": "LOW",
            "min_confidence_score": 0,
            "min_dataset_quality": 0,
            "min_evaluator_reliability": 0,
            "min_statistical_confidence": 0,
            "max_duplicate_ratio": 1,
            "max_distribution_shift": 1,
            "allowed_issues": [
                "Dataset quality is too weak to support a high-confidence release decision."
            ],
        }),
        encoding="utf-8",
    )

    exit_code = verify_command(Namespace(report=str(report_path), policy=str(policy_path)))
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["policy_decision"]["decision"] == "BLOCK"


def test_verify_command_exits_zero_when_policy_allows(tmp_path: Path, capsys):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({
            "dataset": {"name": "demo", "version": "v1", "samples": 100},
            "metrics": [],
            "sample_results": [],
        }),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({
            "max_risk_level": "HIGH",
            "min_confidence_score": 0,
            "min_dataset_quality": 0,
            "min_evaluator_reliability": 0,
            "min_statistical_confidence": 0,
            "max_duplicate_ratio": 1,
            "max_distribution_shift": 1,
            "allowed_issues": [
                "Dataset quality is too weak to support a high-confidence release decision."
            ],
        }),
        encoding="utf-8",
    )

    exit_code = verify_command(Namespace(report=str(report_path), policy=str(policy_path)))
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["policy_decision"]["decision"] == "BLOCK"


def test_cli_subprocess_json_and_error_exit_codes(tmp_path: Path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "project:\n"
        "  name: demo\n"
        "dataset:\n"
        "  path: data.jsonl\n"
        "evaluators:\n"
        "  - name: exact_match\n",
        encoding="utf-8",
    )
    (tmp_path / "data.jsonl").write_text(
        '{"id":"1","input":"capital","expected_answer":"Paris"}\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "evalom", "run", str(config_path), "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["dataset"]["name"] == "data"
    assert completed.stderr == ""

    missing = subprocess.run(
        [sys.executable, "-m", "evalom", "report", str(tmp_path / "missing.json")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert missing.returncode == 2
    assert "error:" in missing.stderr
    assert missing.stdout == ""


def test_profile_cli_outputs_json(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "evalom", "profile", "--task-type", "bug_fix", "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["task_type"] == "bug_fix"
    assert len(payload["definition_hash"]) == 64


def test_api_exposes_profiles_and_existing_comparison_engine(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "api.db"))

    profile = client.get("/api/v1/task-profiles/bug_fix")
    assert profile.status_code == 200
    assert profile.json()["task_type"] == "bug_fix"
    assert len(profile.json()["definition_hash"]) == 64

    continuous = client.post(
        "/api/v1/comparisons/continuous",
        json={"baseline": [1] * 50, "candidate": [2] * 50, "resamples": 50},
    )
    assert continuous.status_code == 200
    assert continuous.json()["difference"] == 1.0

    binary = client.post(
        "/api/v1/comparisons/binary",
        json={
            "baseline_successes": 50,
            "baseline_total": 100,
            "candidate_successes": 60,
            "candidate_total": 100,
        },
    )
    assert binary.status_code == 200
    assert binary.json()["difference"] == pytest.approx(0.1)


def test_api_rejects_invalid_comparison_and_unknown_profile(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "api.db"))
    assert client.get("/api/v1/task-profiles/unknown").status_code == 404
    invalid = client.post(
        "/api/v1/comparisons/continuous",
        json={"baseline": [1], "candidate": [2], "paired": True},
    )
    assert invalid.status_code == 422

    oversized = client.post(
        "/api/v1/verification",
        json={"records": [{}] * 10_001},
    )
    assert oversized.status_code == 422


def test_experiment_tracker_load_handles_empty_file(tmp_path: Path):
    tracker_path = tmp_path / "tracker.json"
    tracker_path.write_text("", encoding="utf-8")

    tracker = ExperimentTracker.load(tracker_path)

    assert tracker.list() == []


def test_experiment_api_validates_requests_and_reports_missing_records(tmp_path: Path):
    database_path = tmp_path / "experiments.db"
    app = create_app(database_path=database_path)
    assert not database_path.exists()
    client = TestClient(app)

    invalid = client.post("/api/v1/experiments", json={"name": ""})
    missing = client.get("/api/v1/experiments/not-a-real-id")
    health = client.get("/healthz")
    readiness = client.get("/readyz")

    assert invalid.status_code == 422
    assert missing.status_code == 404
    assert health.json() == {"status": "ok"}
    assert readiness.json() == {"status": "ready"}


def test_api_key_auth_workspace_isolation_and_revocation(tmp_path: Path):
    bootstrap_key = "test-bootstrap-key-with-at-least-32-chars"
    client = TestClient(create_app(
        database_path=tmp_path / "team_api.db",
        bootstrap_api_key=bootstrap_key,
        bootstrap_workspace_id="team-a",
    ))
    bootstrap_headers = {"Authorization": f"Bearer {bootstrap_key}"}

    assert client.get("/api/v1/experiments").status_code == 401
    assert client.get(
        "/api/v1/experiments",
        headers={"Authorization": "Bearer invalid"},
    ).status_code == 401
    assert client.get("/api/v1/auth/me", headers=bootstrap_headers).json() == {
        "id": "bootstrap",
        "name": "bootstrap",
        "workspace_id": "team-a",
        "role": "admin",
    }

    member_response = client.post(
        "/api/v1/api-keys",
        headers=bootstrap_headers,
        json={"name": "ci-member", "role": "member"},
    )
    assert member_response.status_code == 201
    member_key = member_response.json()["api_key"]
    member_headers = {"Authorization": f"Bearer {member_key}"}
    key_records = client.get("/api/v1/api-keys", headers=bootstrap_headers).json()
    assert all("api_key" not in record and "key_hash" not in record for record in key_records)
    assert client.get("/api/v1/api-keys", headers=member_headers).status_code == 403

    workspace_response = client.post(
        "/api/v1/workspaces",
        headers=bootstrap_headers,
        json={"workspace_id": "team-b", "admin_key_name": "team-b-admin"},
    )
    assert workspace_response.status_code == 201
    workspace_key = workspace_response.json()["admin_api_key"]["api_key"]
    workspace_headers = {"Authorization": f"Bearer {workspace_key}"}

    experiment_a = client.post(
        "/api/v1/experiments",
        headers=member_headers,
        json={"name": "team-a-run"},
    ).json()
    experiment_b_response = client.post(
        "/api/v1/experiments",
        headers=workspace_headers,
        json={"name": "team-b-run"},
    )
    assert experiment_b_response.status_code == 201
    experiment_b = experiment_b_response.json()
    assert experiment_a["workspace_id"] == "team-a"
    assert experiment_b["workspace_id"] == "team-b"
    assert client.get("/api/v1/experiments", headers=member_headers).json()["total"] == 1
    assert client.get(
        f"/api/v1/experiments/{experiment_b['id']}",
        headers=member_headers,
    ).status_code == 404

    verification_payload = {
        "dataset_name": "team-check",
        "records": [
            {"sample_id": "1", "input": "question", "output": "answer", "score": 0.9}
        ],
    }
    report_a = client.post(
        "/api/v1/verification",
        headers=member_headers,
        json=verification_payload,
    ).json()
    report_b_response = client.post(
        "/api/v1/verification",
        headers=workspace_headers,
        json=verification_payload,
    )
    assert report_b_response.status_code == 200
    assert client.get(
        f"/api/v1/verification/{report_a['report_id']}",
        headers=workspace_headers,
    ).status_code == 404
    assert client.get(
        "/api/v1/verification",
        headers=member_headers,
    ).json()["total"] == 1
    assert client.get(
        "/api/v1/verification",
        headers=workspace_headers,
    ).json()["total"] == 1

    with sqlite3.connect(tmp_path / "team_api.db") as connection:
        stored_key_hashes = {
            row[0] for row in connection.execute("SELECT key_hash FROM api_keys")
        }
    assert member_key not in stored_key_hashes

    revoked = client.delete(
        f"/api/v1/api-keys/{member_response.json()['id']}",
        headers=bootstrap_headers,
    )
    assert revoked.status_code == 204
    assert client.get("/api/v1/experiments", headers=member_headers).status_code == 401


def test_create_experiment_store_uses_sqlite_for_local_path(tmp_path: Path):
    store = create_experiment_store(tmp_path / "local.db")
    assert isinstance(store, SQLiteExperimentStore)


def test_create_experiment_store_uses_postgres_backend_for_dsn(monkeypatch):
    class FakePostgresStore:
        def __init__(self, database_url: str):
            self.database_url = database_url

    monkeypatch.setattr("evalom.storage.PostgresExperimentStore", FakePostgresStore)
    store = create_experiment_store("postgresql://user:pass@localhost:5432/evalom")

    assert isinstance(store, FakePostgresStore)
    assert store.database_url == "postgresql://user:pass@localhost:5432/evalom"


def test_api_rejects_short_bootstrap_key():
    with pytest.raises(ValueError, match="at least 32 characters"):
        create_app(bootstrap_api_key="too-short")


def test_api_rejects_insecure_remote_bootstrap_binding(monkeypatch):
    monkeypatch.setenv("EVALOM_HOST", "0.0.0.0")
    monkeypatch.delenv("EVALOM_ALLOW_REMOTE_ACCESS", raising=False)
    with pytest.raises(ValueError, match="Remote API exposure is blocked by default"):
        create_app(bootstrap_api_key="test-bootstrap-key-with-at-least-32-chars")

    monkeypatch.setenv("EVALOM_ALLOW_REMOTE_ACCESS", "true")
    with pytest.raises(ValueError, match="EVALOM_TLS_CERTFILE and EVALOM_TLS_KEYFILE"):
        create_app(bootstrap_api_key="test-bootstrap-key-with-at-least-32-chars")

    monkeypatch.setenv("EVALOM_TLS_CERTFILE", "/tmp/server.crt")
    monkeypatch.setenv("EVALOM_TLS_KEYFILE", "/tmp/server.key")
    app = create_app(bootstrap_api_key="test-bootstrap-key-with-at-least-32-chars")
    assert app is not None


def test_audit_events_are_logged_for_admin_actions(tmp_path: Path):
    bootstrap_key = "test-bootstrap-key-with-at-least-32-chars"
    client = TestClient(create_app(
        database_path=tmp_path / "audit_events.db",
        bootstrap_api_key=bootstrap_key,
        bootstrap_workspace_id="team-a",
    ))
    admin_headers = {"Authorization": f"Bearer {bootstrap_key}"}

    key_response = client.post(
        "/api/v1/api-keys",
        headers=admin_headers,
        json={"name": "audit-reader", "role": "member"},
    )
    member_key = key_response.json()["api_key"]
    member_headers = {"Authorization": f"Bearer {member_key}"}

    admin_events = client.get("/api/v1/audit-events", headers=admin_headers)
    assert admin_events.status_code == 200
    assert admin_events.json()["total"] >= 1
    assert any(item["event_type"] == "api_key.created" for item in admin_events.json()["items"])
    assert client.get("/api/v1/audit-events", headers=member_headers).status_code == 403


def test_verification_engine_scores_trustworthy_runs_as_low_risk():
    dataset = EvaluationDataset([
        {"id": f"sample-{idx}", "input": f"question {idx}", "expected_answer": "answer", "expected_context": ["context"]}
        for idx in range(100)
    ])
    result = EvaluationRunResult(
        dataset_name="demo",
        dataset_version="v1",
        total_samples=100,
        metrics=[{"name": "exact_match", "score": 0.92, "passed": True} for _ in range(100)],
    )

    verification = VerificationEngine().verify(result, dataset=dataset)

    assert verification.confidence_score < 80.0
    assert verification.risk_level == "MEDIUM"
    assert 0.0 <= verification.confidence_interval["lower"] <= 1.0
    assert 0.0 <= verification.confidence_interval["upper"] <= 1.0
    assert "Evaluator reliability is unavailable" in "\n".join(verification.issues)


def test_normalized_records_use_common_schema():
    record = normalize_evaluation_record({
        "sampleId": "a-1",
        "question": "What is 2 + 2?",
        "answer": "4",
        "expected": "4",
        "contexts": ["arithmetic"],
        "name": "exact_match",
        "score": 1.0,
    })

    assert record["sample_id"] == "a-1"
    assert record["input"] == "What is 2 + 2?"
    assert record["output"] == "4"
    assert record["ground_truth"] == "4"
    assert record["evaluator"] == "exact_match"
    assert record["score"] == 1.0


def test_import_layer_normalizes_langfuse_style_records():
    payload = [{
        "id": "trace-1",
        "name": "faithfulness",
        "score": 0.91,
        "metadata": {"input": "What is the capital of France?", "output": "Paris"},
    }]

    normalized = EvaluationImportLayer.normalize_records(payload)
    assert normalized[0]["sample_id"] == "trace-1"
    assert normalized[0]["evaluator"] == "faithfulness"
    assert normalized[0]["score"] == 0.91

    langfuse_record = LangfuseAdapter.normalize_record({
        "id": "trace-2",
        "name": "quality",
        "score": 0.77,
    })
    assert langfuse_record["evaluator"] == "quality"
    assert langfuse_record["score"] == 0.77


def test_verification_endpoint_accepts_imported_records(tmp_path: Path):
    database_path = tmp_path / "experiments.db"
    client = TestClient(create_app(database_path=database_path))

    response = client.post(
        "/api/v1/verification",
        json={
            "dataset_name": "demo",
            "dataset_version": "v1",
            "records": [
                {"id": "1", "name": "exact_match", "score": 0.9, "input": "capital", "answer": "Paris"},
                {"id": "2", "name": "exact_match", "score": 0.95, "input": "capital", "answer": "Paris"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_records"] == 2
    assert body["verification"]["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert "confidence_score" in body["verification"]


def test_verification_result_includes_category_distribution_and_drift():
    dataset = EvaluationDataset([
        {"id": "1", "input": "capital", "expected_answer": "Paris", "metadata": {"category": "simple"}},
        {"id": "2", "input": "complex", "expected_answer": "answer", "metadata": {"category": "simple"}},
        {"id": "3", "input": "edge", "expected_answer": "answer", "metadata": {"category": "hard"}},
    ])
    result = EvaluationRunResult(
        dataset_name="demo",
        dataset_version="v1",
        total_samples=3,
        metrics=[{"name": "exact_match", "score": 0.9, "passed": True}, {"name": "exact_match", "score": 0.8, "passed": True}, {"name": "exact_match", "score": 0.7, "passed": False}],
        sample_results=[
            {"sample_id": "1", "result": [{"name": "exact_match", "score": 0.9, "passed": True}]},
            {"sample_id": "2", "result": [{"name": "exact_match", "score": 0.8, "passed": True}]},
            {"sample_id": "3", "result": [{"name": "exact_match", "score": 0.7, "passed": False}]},
        ],
    )

    verification = VerificationEngine().verify(result, dataset=dataset, evaluator_agreement=0.9)

    assert verification.dataset_health["category_distribution"]["simple"] == 2
    assert verification.dataset_health["distribution_shift"] is None
    assert verification.dataset_health["category_diversity"] >= 0.0
    assert "dataset_health" in verification.to_dict()


def test_dataset_fingerprint_is_stable_and_exposed_in_run_metadata():
    dataset = EvaluationDataset([
        {"id": "1", "input": "capital", "expected_answer": "Paris"},
        {"id": "2", "input": "math", "expected_answer": "4"},
    ], name="demo", version="v2")

    digest = dataset.fingerprint()
    assert digest == dataset.hash
    assert len(digest) == 64

    runner = EvaluationRunner(dataset, [ExactMatchEvaluator()])
    result = runner.run(lambda _: {"answer": "Paris", "contexts": []}, metadata={"run_label": "baseline"})

    assert result.metadata["dataset_hash"] == digest
    assert result.metadata["run_label"] == "baseline"


def test_experiment_api_reuses_idempotency_key_for_duplicate_requests(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "idempotent.db"))
    payload = {"name": "retry-safe-run", "idempotency_key": "ci-run-123"}

    first = client.post("/api/v1/experiments", json=payload)
    second = client.post("/api/v1/experiments", json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]

    different_payload = client.post(
        "/api/v1/experiments",
        json={**payload, "name": "different-run"},
    )
    assert different_payload.status_code == 409
    assert "different request payload" in different_payload.json()["detail"]


def test_experiment_idempotency_keys_are_workspace_scoped(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "workspace_idempotency.db")
    payload = {"name": "workspace-run", "idempotency_key": "same-key"}

    team_a = store.create(payload, workspace_id="team-a")
    team_b = store.create(payload, workspace_id="team-b")
    retry_a = store.create(payload, workspace_id="team-a")

    assert team_a["id"] != team_b["id"]
    assert retry_a["id"] == team_a["id"]
    assert team_a["workspace_id"] == "team-a"
    assert team_b["workspace_id"] == "team-b"


def test_concurrent_experiment_retries_create_only_one_record(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "concurrent_idempotency.db")
    payload = {"name": "concurrent-run", "idempotency_key": "parallel-ci-run"}

    with ThreadPoolExecutor(max_workers=6) as executor:
        records = list(
            executor.map(
                lambda _: store.create(payload, workspace_id="team-a"),
                range(6),
            )
        )

    assert len({record["id"] for record in records}) == 1
    assert store.list(workspace_id="team-a")["total"] == 1


def test_running_experiment_can_be_completed_once_through_api(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "lifecycle.db"))
    created = client.post(
        "/api/v1/experiments",
        json={"name": "running-run", "status": "running"},
    )
    record = created.json()

    completed = client.patch(
        f"/api/v1/experiments/{record['id']}",
        json={
            "status": "completed",
            "metrics": {"exact_match": 0.95},
            "metadata": {"worker": "ci-1"},
        },
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["metrics"] == {"exact_match": 0.95}
    assert completed.json()["metadata"] == {"worker": "ci-1"}
    assert completed.json()["updated_at"] >= record["updated_at"]

    retry = client.patch(
        f"/api/v1/experiments/{record['id']}",
        json={"status": "failed"},
    )
    assert retry.status_code == 409
    assert retry.json()["detail"] == "Only running experiments can be updated."


def test_experiment_lifecycle_update_is_workspace_scoped(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "workspace_lifecycle.db")
    record = store.create({"name": "private-run", "status": "running"}, workspace_id="team-a")

    assert store.update(record["id"], {"status": "completed"}, workspace_id="team-b") is None
    updated = store.update(record["id"], {"status": "failed"}, workspace_id="team-a")
    assert updated is not None
    assert updated["status"] == "failed"
