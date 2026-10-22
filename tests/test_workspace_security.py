from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evalom.api import _validate_tls_files, create_app

BOOTSTRAP = "b" * 40


def test_workspace_creation_cannot_mint_keys_for_existing_workspace(tmp_path: Path):
    app = create_app(database_path=tmp_path / "t.db", bootstrap_api_key=BOOTSTRAP)
    client = TestClient(app)
    admin = {"Authorization": f"Bearer {BOOTSTRAP}"}

    victim = client.post("/api/v1/workspaces", json={"workspace_id": "victim"}, headers=admin)
    attacker = client.post("/api/v1/workspaces", json={"workspace_id": "attacker"}, headers=admin)
    assert victim.status_code == 201
    assert attacker.status_code == 201
    attacker_key = attacker.json()["admin_api_key"]["api_key"]
    attacker_headers = {"Authorization": f"Bearer {attacker_key}"}

    stolen = client.post("/api/v1/workspaces", json={"workspace_id": "victim"}, headers=attacker_headers)
    assert stolen.status_code == 409

    default = client.post("/api/v1/workspaces", json={"workspace_id": "default"}, headers=attacker_headers)
    assert default.status_code == 409

def test_api_refuses_remote_binding_without_authentication(monkeypatch):
    monkeypatch.setenv("EVALOM_HOST", "0.0.0.0")
    monkeypatch.delenv("EVALOM_ADMIN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="without authentication"):
        create_app()


def test_api_allows_loopback_without_authentication(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("EVALOM_HOST", "127.0.0.1")
    monkeypatch.delenv("EVALOM_ADMIN_API_KEY", raising=False)
    assert create_app(database_path=tmp_path / "t.db") is not None


def test_api_rejects_malformed_verification_records_without_server_error(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "invalid-record.db"))
    response = client.post(
        "/api/v1/verification",
        json={
            "records": [
                {"sample_id": "1", "input": "question", "output": "answer", "score": "not-a-number"}
            ]
        },
    )
    assert response.status_code == 422
    assert "Invalid evaluation record payload" in response.json()["detail"]


def test_api_rejects_out_of_range_assurance_inputs(tmp_path: Path):
    client = TestClient(create_app(database_path=tmp_path / "invalid-input.db"))
    response = client.post(
        "/api/v1/verification",
        json={"records": [{"sample_id": "1", "score": 0.9}], "evaluator_agreement": 2},
    )
    assert response.status_code == 422


def test_remote_tls_configuration_requires_regular_files(tmp_path: Path):
    with pytest.raises(ValueError, match="missing or is not a regular file"):
        _validate_tls_files(str(tmp_path / "missing.crt"), str(tmp_path / "missing.key"))
    certfile = tmp_path / "server.crt"
    keyfile = tmp_path / "server.key"
    certfile.write_text("certificate", encoding="utf-8")
    keyfile.write_text("private key", encoding="utf-8")
    _validate_tls_files(str(certfile), str(keyfile))
