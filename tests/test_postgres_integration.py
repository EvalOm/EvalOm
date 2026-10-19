from __future__ import annotations

import os
import threading
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from evalom.api import create_app
from evalom.evidence import EvidenceRecord, EvidenceRegistry, register_reference_evidence
from evalom.storage import create_experiment_store


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_api_storage_smoke():
    pytest.importorskip("psycopg")
    dsn = os.environ["EVALOM_TEST_POSTGRES_DSN"]
    bootstrap_key = "test-bootstrap-key-with-at-least-32-chars"
    suffix = uuid4().hex
    admin_workspace = f"pg-admin-{suffix}"
    team_workspace = f"pg-team-{suffix}"
    client = TestClient(
        create_app(
            database_path=dsn,
            bootstrap_api_key=bootstrap_key,
            bootstrap_workspace_id=admin_workspace,
        )
    )
    bootstrap_headers = {"Authorization": f"Bearer {bootstrap_key}"}

    assert client.get("/readyz").status_code == 200
    workspace = client.post(
        "/api/v1/workspaces",
        headers=bootstrap_headers,
        json={"workspace_id": team_workspace},
    )
    assert workspace.status_code == 201
    team_headers = {
        "Authorization": f"Bearer {workspace.json()['admin_api_key']['api_key']}"
    }

    payload = {
        "name": "postgres-smoke",
        "status": "running",
        "idempotency_key": suffix,
        "metrics": {"exact_match": 0.9},
    }
    created = client.post("/api/v1/experiments", headers=team_headers, json=payload)
    repeated = client.post("/api/v1/experiments", headers=team_headers, json=payload)
    assert created.status_code == 201
    assert repeated.json()["id"] == created.json()["id"]
    changed_replay = client.post(
        "/api/v1/experiments",
        headers=team_headers,
        json={**payload, "name": "changed-replay"},
    )
    assert changed_replay.status_code == 409

    completed = client.patch(
        f"/api/v1/experiments/{created.json()['id']}",
        headers=team_headers,
        json={"status": "completed", "metadata": {"backend": "postgres"}},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"

    verification = client.post(
        "/api/v1/verification",
        headers=team_headers,
        json={
            "dataset_name": "postgres-smoke",
            "records": [
                {
                    "sample_id": "sample-1",
                    "input": "question",
                    "output": "answer",
                    "score": 0.9,
                }
            ],
        },
    )
    assert verification.status_code == 200
    report = client.get(
        f"/api/v1/verification/{verification.json()['report_id']}",
        headers=team_headers,
    )
    assert report.status_code == 200
    assert report.json()["dataset_name"] == "postgres-smoke"
    assert client.get(
        "/api/v1/audit-events",
        headers=team_headers,
    ).json()["total"] >= 2

    assert client.get(
        f"/api/v1/experiments/{created.json()['id']}",
        headers=bootstrap_headers,
    ).status_code == 404


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_evidence_supersession_transaction():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    registry = EvidenceRegistry(store)
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://postgres-1", workspace_id="pg-evidence"
    ))
    replacement = register_reference_evidence(
        evidence_type="human_review",
        source="review://postgres-2",
        workspace_id="pg-evidence",
        evidence_id="postgres-replacement",
    )
    replacement = replacement.__class__(
        **{**replacement.to_dict(), "supersedes_id": original["evidence_id"]}
    )
    registry.supersede(original["evidence_id"], replacement)

    assert registry.get(original["evidence_id"], workspace_id="pg-evidence")["lifecycle_status"] == "superseded"
    assert registry.get("postgres-replacement", workspace_id="pg-evidence") is not None


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
@pytest.mark.parametrize("status", ["verified", "mismatch"])
def test_postgres_evidence_rejects_fabricated_digest_payload(status):
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    payload = register_reference_evidence(
        evidence_type="test_result",
        source="ci://forged-postgres",
        workspace_id=f"pg-forged-{uuid4().hex}",
    ).to_dict()
    payload.update({
        "digest_status": status,
        "verified_digest": "c" * 64,
        "_trusted_digest": True,
    })

    with pytest.raises(ValueError, match="Trusted digest"):
        store.save_evidence(payload)


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
@pytest.mark.parametrize("status", ["verified", "mismatch"])
def test_postgres_atomic_supersession_rejects_fabricated_digest_payload(status):
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    registry = EvidenceRegistry(store)
    workspace_id = f"pg-forged-supersede-{uuid4().hex}"
    original = registry.register(register_reference_evidence(
        evidence_type="human_review",
        source="review://postgres-original",
        workspace_id=workspace_id,
    ))
    replacement = register_reference_evidence(
        evidence_type="human_review",
        source="review://postgres-replacement",
        workspace_id=workspace_id,
        evidence_id=f"forged-replacement-{uuid4().hex}",
    ).to_dict()
    replacement.update({
        "supersedes_id": original["evidence_id"],
        "digest_status": status,
        "verified_digest": "d" * 64,
        "_trusted_digest": True,
    })

    with pytest.raises(ValueError, match="Trusted digest"):
        store.supersede_evidence_and_audit(original["evidence_id"], replacement)
    assert registry.get(original["evidence_id"], workspace_id=workspace_id)["lifecycle_status"] == "active"


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_evidence_validation_and_soft_delete():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    registry = EvidenceRegistry(store)
    workspace_id = f"pg-lifecycle-{uuid4().hex}"
    evidence = registry.register(register_reference_evidence(
        evidence_type="human_review",
        source="review://postgres-lifecycle",
        workspace_id=workspace_id,
    ))

    validation = registry.validate(evidence["evidence_id"], workspace_id=workspace_id)
    deleted = registry.delete(evidence["evidence_id"], workspace_id=workspace_id, actor_id="pg-actor")

    assert validation["status"] == "unavailable"
    assert deleted["lifecycle_status"] == "deleted"
    assert store.list_evidence(workspace_id=workspace_id) == []
    assert store.list_evidence(workspace_id=workspace_id, include_deleted=True)[0]["lifecycle_status"] == "deleted"


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_supersession_insert_failure_rolls_back():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    registry = EvidenceRegistry(store)
    workspace_id = f"pg-rollback-{uuid4().hex}"
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://original", workspace_id=workspace_id,
    ))
    duplicate = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://duplicate", workspace_id=workspace_id,
        evidence_id=f"pg-duplicate-{uuid4().hex}",
    ))
    replacement = EvidenceRecord(
        evidence_id=duplicate["evidence_id"], evidence_type="human_review",
        workspace_id=workspace_id, source="review://replacement",
        supersedes_id=original["evidence_id"],
    )

    with pytest.raises(ValueError):
        registry.supersede(original["evidence_id"], replacement)

    assert registry.get(original["evidence_id"], workspace_id=workspace_id)["lifecycle_status"] == "active"
    assert registry.get(duplicate["evidence_id"], workspace_id=workspace_id)["source"] == "review://duplicate"


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_supersession_audit_failure_rolls_back():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    registry = EvidenceRegistry(store)
    workspace_id = f"pg-audit-rollback-{uuid4().hex}"
    function_name = f"fail_evidence_audit_{uuid4().hex}"
    trigger_name = f"fail_evidence_audit_trigger_{uuid4().hex}"
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://original", workspace_id=workspace_id,
    ))
    replacement = EvidenceRecord(
        evidence_id=f"pg-replacement-{uuid4().hex}", evidence_type="human_review",
        workspace_id=workspace_id, source="review://replacement",
        supersedes_id=original["evidence_id"],
    )
    try:
        with store._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"""
                    CREATE FUNCTION {function_name}() RETURNS trigger AS $$
                    BEGIN
                        IF NEW.event_type = 'evidence.superseded' THEN
                            RAISE EXCEPTION 'supersession audit failure';
                        END IF;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql
                """)
                cursor.execute(
                    f"CREATE TRIGGER {trigger_name} BEFORE INSERT ON audit_events "
                    f"FOR EACH ROW EXECUTE FUNCTION {function_name}()"
                )
        with pytest.raises(Exception, match="supersession audit failure"):
            registry.supersede(original["evidence_id"], replacement)
    finally:
        with store._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON audit_events")
                cursor.execute(f"DROP FUNCTION IF EXISTS {function_name}()")

    assert registry.get(original["evidence_id"], workspace_id=workspace_id)["lifecycle_status"] == "active"
    assert registry.get(replacement.evidence_id, workspace_id=workspace_id) is None


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_soft_delete_audit_failure_rolls_back():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    registry = EvidenceRegistry(store)
    workspace_id = f"pg-delete-rollback-{uuid4().hex}"
    function_name = f"fail_evidence_delete_{uuid4().hex}"
    trigger_name = f"fail_evidence_delete_trigger_{uuid4().hex}"
    evidence = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://delete", workspace_id=workspace_id,
    ))
    try:
        with store._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"""
                    CREATE FUNCTION {function_name}() RETURNS trigger AS $$
                    BEGIN
                        IF NEW.event_type = 'evidence.deleted' THEN
                            RAISE EXCEPTION 'delete audit failure';
                        END IF;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql
                """)
                cursor.execute(
                    f"CREATE TRIGGER {trigger_name} BEFORE INSERT ON audit_events "
                    f"FOR EACH ROW EXECUTE FUNCTION {function_name}()"
                )
        with pytest.raises(Exception, match="delete audit failure"):
            registry.delete(evidence["evidence_id"], workspace_id=workspace_id)
    finally:
        with store._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON audit_events")
                cursor.execute(f"DROP FUNCTION IF EXISTS {function_name}()")

    assert registry.get(evidence["evidence_id"], workspace_id=workspace_id)["lifecycle_status"] == "active"


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_concurrent_supersession_only_one_attempt_succeeds():
    pytest.importorskip("psycopg")
    dsn = os.environ["EVALOM_TEST_POSTGRES_DSN"]
    setup_store = create_experiment_store(dsn)
    registry = EvidenceRegistry(setup_store)
    workspace_id = f"pg-concurrent-{uuid4().hex}"
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://concurrent", workspace_id=workspace_id,
    ))
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def attempt(index: int) -> None:
        store = create_experiment_store(dsn)
        candidate = EvidenceRecord(
            evidence_id=f"pg-concurrent-replacement-{index}-{uuid4().hex}",
            evidence_type="human_review", workspace_id=workspace_id,
            source=f"review://replacement-{index}", supersedes_id=original["evidence_id"],
        )
        try:
            barrier.wait(timeout=10)
            EvidenceRegistry(store).supersede(original["evidence_id"], candidate)
        except Exception:
            outcomes.append("failed")
        else:
            outcomes.append("succeeded")

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert outcomes.count("succeeded") == 1
    assert outcomes.count("failed") == 1
    assert registry.get(original["evidence_id"], workspace_id=workspace_id)["lifecycle_status"] == "superseded"
    assert len(registry.store.list_evidence(workspace_id=workspace_id)) == 2


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_evidence_workspace_isolation_for_lifecycle_operations():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    registry = EvidenceRegistry(store)
    workspace_a = f"pg-workspace-a-{uuid4().hex}"
    workspace_b = f"pg-workspace-b-{uuid4().hex}"
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://isolated", workspace_id=workspace_a,
    ))
    replacement = EvidenceRecord(
        evidence_id=f"pg-isolated-replacement-{uuid4().hex}", evidence_type="human_review",
        workspace_id=workspace_b, source="review://replacement",
        supersedes_id=original["evidence_id"],
    )

    assert registry.get(original["evidence_id"], workspace_id=workspace_b) is None
    with pytest.raises(ValueError):
        registry.supersede(original["evidence_id"], replacement)
    with pytest.raises(ValueError):
        registry.delete(original["evidence_id"], workspace_id=workspace_b)
    assert registry.get(original["evidence_id"], workspace_id=workspace_a)["lifecycle_status"] == "active"


def _postgres_report_payload(report_id: str) -> dict:
    return {
        "report_id": report_id,
        "dataset_name": "postgres-report",
        "dataset_version": "v1",
        "risk_level": "LOW",
        "confidence_score": 97.0,
        "policy_decision": {"decision": "ALLOW"},
        "verification": {
            "status": "verified",
            "checks": [{"name": "completeness", "status": "passed"}],
            "evidence_references": ["evidence-1"],
        },
    }


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_atomic_report_save_round_trip_and_workspace_isolation():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    workspace_id = f"pg-report-{uuid4().hex}"
    report_id = f"pg-report-{uuid4().hex}"

    report = store.save_verification_report_and_audit(
        _postgres_report_payload(report_id),
        workspace_id=workspace_id,
        actor_id="pg-actor",
        actor_name="integration",
    )

    assert store.get_verification_report(report_id, workspace_id=workspace_id) == report
    assert store.get_verification_report(report_id, workspace_id="other-workspace") is None
    assert store.list_verification_reports(workspace_id=workspace_id)["total"] == 1
    assert store.list_verification_reports(workspace_id="other-workspace")["total"] == 0
    events = store.list_audit_events(workspace_id=workspace_id)["items"]
    assert any(
        event["event_type"] == "verification.created"
        and event["details"]["report_id"] == report_id
        for event in events
    )


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_atomic_report_audit_failure_rolls_back_and_recovers():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    workspace_id = f"pg-report-rollback-{uuid4().hex}"
    function_name = f"fail_report_audit_{uuid4().hex}"
    trigger_name = f"fail_report_audit_trigger_{uuid4().hex}"
    failed_id = f"pg-failed-report-{uuid4().hex}"
    recovered_id = f"pg-recovered-report-{uuid4().hex}"
    try:
        with store._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"""
                    CREATE FUNCTION {function_name}() RETURNS trigger AS $$
                    BEGIN
                        IF NEW.event_type = 'verification.created' THEN
                            RAISE EXCEPTION 'report audit failure';
                        END IF;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql
                """)
                cursor.execute(
                    f"CREATE TRIGGER {trigger_name} BEFORE INSERT ON audit_events "
                    f"FOR EACH ROW EXECUTE FUNCTION {function_name}()"
                )
        with pytest.raises(Exception, match="report audit failure"):
            store.save_verification_report_and_audit(
                _postgres_report_payload(failed_id), workspace_id=workspace_id
            )
    finally:
        with store._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON audit_events")
                cursor.execute(f"DROP FUNCTION IF EXISTS {function_name}()")

    assert store.get_verification_report(failed_id, workspace_id=workspace_id) is None
    assert store.save_verification_report_and_audit(
        _postgres_report_payload(recovered_id), workspace_id=workspace_id
    )["id"] == recovered_id


@pytest.mark.skipif(
    not os.environ.get("EVALOM_TEST_POSTGRES_DSN"),
    reason="Set EVALOM_TEST_POSTGRES_DSN to run PostgreSQL integration coverage.",
)
def test_postgres_atomic_report_duplicate_id_does_not_overwrite_or_audit():
    pytest.importorskip("psycopg")
    store = create_experiment_store(os.environ["EVALOM_TEST_POSTGRES_DSN"])
    workspace_id = f"pg-report-duplicate-{uuid4().hex}"
    report_id = f"pg-duplicate-report-{uuid4().hex}"
    store.save_verification_report_and_audit(
        _postgres_report_payload(report_id), workspace_id=workspace_id
    )

    with pytest.raises(Exception):
        store.save_verification_report_and_audit(
            {**_postgres_report_payload(report_id), "dataset_name": "changed"},
            workspace_id=workspace_id,
        )

    persisted = store.get_verification_report(report_id, workspace_id=workspace_id)
    assert persisted["dataset_name"] == "postgres-report"
    assert store.list_audit_events(workspace_id=workspace_id)["total"] == 1
