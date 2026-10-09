from __future__ import annotations

import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from evalom.api import create_app


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
