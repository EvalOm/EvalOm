from __future__ import annotations

import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from evalom.api import create_app
from evalom.evidence import EvidenceRegistry, register_reference_evidence
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
