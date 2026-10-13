from pathlib import Path

from fastapi.testclient import TestClient

from evalom.api import create_app

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
