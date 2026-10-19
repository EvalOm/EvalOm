import hashlib
from pathlib import Path

import pytest

from evalom.evidence import EvidenceRecord, EvidenceRegistry, register_file_evidence, register_reference_evidence
from evalom.storage import SQLiteExperimentStore


def test_register_and_retrieve_file_evidence(tmp_path: Path):
    artifact = tmp_path / "result.txt"
    artifact.write_text("verified", encoding="utf-8")
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)

    evidence = register_file_evidence(
        "result.txt",
        artifact_root=tmp_path,
        workspace_id="workspace-a",
        task_id="task-1",
        execution_id="execution-1",
    )
    saved = registry.register(evidence)
    loaded = registry.get(saved["evidence_id"], workspace_id="workspace-a")

    assert loaded is not None
    assert loaded["verified_digest"] == hashlib.sha256(b"verified").hexdigest()
    assert loaded["digest_status"] == "verified"
    assert loaded["task_id"] == "task-1"


def test_digest_mismatch_is_explicit(tmp_path: Path):
    (tmp_path / "result.txt").write_text("actual", encoding="utf-8")
    evidence = register_file_evidence(
        "result.txt",
        artifact_root=tmp_path,
        claimed_digest=hashlib.sha256(b"claimed").hexdigest(),
    )
    assert evidence.digest_status == "mismatch"
    assert evidence.verified_digest != evidence.claimed_digest


def test_claimed_digest_match_is_verified(tmp_path: Path):
    (tmp_path / "result.txt").write_text("actual", encoding="utf-8")
    digest = hashlib.sha256(b"actual").hexdigest()
    evidence = register_file_evidence(
        "result.txt",
        artifact_root=tmp_path,
        claimed_digest=digest,
    )
    assert evidence.digest_status == "verified"
    assert evidence.claimed_digest == evidence.verified_digest


def test_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        register_file_evidence("missing.txt", artifact_root=tmp_path)


def test_validate_file_evidence_detects_change_without_mutating_history(tmp_path: Path):
    artifact = tmp_path / "result.txt"
    artifact.write_text("original", encoding="utf-8")
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_file_evidence("result.txt", artifact_root=tmp_path, workspace_id="a"))
    historical = evidence["verified_digest"]
    artifact.write_text("changed", encoding="utf-8")

    result = registry.validate(evidence["evidence_id"], workspace_id="a", artifact_root=tmp_path)

    assert result["status"] == "mismatch"
    assert result["historical_digest"] == historical
    assert registry.get(evidence["evidence_id"], workspace_id="a")["verified_digest"] == historical


def test_validate_removed_file_is_unavailable(tmp_path: Path):
    artifact = tmp_path / "result.txt"
    artifact.write_text("original", encoding="utf-8")
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_file_evidence("result.txt", artifact_root=tmp_path))
    artifact.unlink()

    result = registry.validate(evidence["evidence_id"], artifact_root=tmp_path)

    assert result["status"] == "unavailable"


def test_validate_enforces_size_and_path_safety(tmp_path: Path):
    artifact = tmp_path / "result.txt"
    artifact.write_text("content", encoding="utf-8")
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_file_evidence("result.txt", artifact_root=tmp_path))
    assert registry.validate(evidence["evidence_id"], artifact_root=tmp_path, max_file_size=2)["status"] == "unavailable"

    inside_target = tmp_path / "inside-target.txt"
    inside_target.write_text("inside", encoding="utf-8")
    inside_link = tmp_path / "inside-link.txt"
    inside_link.symlink_to(inside_target)
    with pytest.raises(ValueError):
        register_file_evidence("inside-link.txt", artifact_root=tmp_path)


def test_matching_digest_is_not_a_task_assurance_result(tmp_path: Path):
    artifact = tmp_path / "result.txt"
    artifact.write_text("correct-looking", encoding="utf-8")
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_file_evidence(
        "result.txt", artifact_root=tmp_path, task_id="task-1"
    ))

    result = registry.validate(evidence["evidence_id"], artifact_root=tmp_path)

    assert result["status"] == "matched"
    assert "task_status" not in result
    assert result["limitations"]


def test_validate_reference_evidence_is_explicitly_unavailable(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_reference_evidence(
        evidence_type="test_result", source="ci://run", workspace_id="a"
    ))

    result = registry.validate(evidence["evidence_id"], workspace_id="a")

    assert result["status"] == "unavailable"
    assert result["integrity_match"] is None


def test_validate_is_workspace_scoped_and_does_not_verify_tasks(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://1", workspace_id="a", task_id="task-1"
    ))

    assert registry.validate(evidence["evidence_id"], workspace_id="b")["status"] == "not_found"
    assert "verified" not in registry.validate(evidence["evidence_id"], workspace_id="a")["status"]


def test_soft_delete_is_atomic_audited_and_retains_artifact(tmp_path: Path):
    artifact = tmp_path / "result.txt"
    artifact.write_text("retain", encoding="utf-8")
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_file_evidence("result.txt", artifact_root=tmp_path, workspace_id="a"))

    deleted = registry.delete(evidence["evidence_id"], workspace_id="a", actor_id="actor-1")

    assert deleted["lifecycle_status"] == "deleted"
    assert artifact.read_text(encoding="utf-8") == "retain"
    assert store.list_evidence(workspace_id="a") == []
    assert store.list_evidence(workspace_id="a", include_deleted=True)[0]["lifecycle_status"] == "deleted"
    assert any(event["event_type"] == "evidence.deleted" for event in store.list_audit_events(workspace_id="a")["items"])


def test_soft_delete_rejects_missing_cross_workspace_and_repeated_requests(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://1", workspace_id="a"
    ))
    with pytest.raises(ValueError):
        registry.delete(evidence["evidence_id"], workspace_id="b")
    registry.delete(evidence["evidence_id"], workspace_id="a")
    with pytest.raises(ValueError):
        registry.delete(evidence["evidence_id"], workspace_id="a")
    with pytest.raises(ValueError):
        registry.delete("missing", workspace_id="a")


def test_soft_delete_audit_failure_rolls_back(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    evidence = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://1", workspace_id="a"
    ))
    with store._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_evidence_delete BEFORE INSERT ON audit_events
            WHEN NEW.event_type = 'evidence.deleted'
            BEGIN SELECT RAISE(ABORT, 'delete audit failure'); END
            """
        )
    with pytest.raises(Exception, match="delete audit failure"):
        registry.delete(evidence["evidence_id"], workspace_id="a")
    assert registry.get(evidence["evidence_id"], workspace_id="a")["lifecycle_status"] == "active"


def test_digest_status_cannot_be_forged_by_direct_or_serialized_construction():
    with pytest.raises(ValueError):
        EvidenceRecord(digest_status="verified", verified_digest="a" * 64)
    with pytest.raises(ValueError):
        EvidenceRecord.from_dict({
            "evidence_id": "serialized",
            "digest_status": "mismatch",
            "verified_digest": "a" * 64,
        })


@pytest.mark.parametrize("status", ["verified", "mismatch"])
def test_storage_rejects_fabricated_trusted_digest_payload(tmp_path: Path, status: str):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    payload = register_reference_evidence(
        evidence_type="test_result",
        source="ci://forged",
    ).to_dict()
    payload.update({
        "digest_status": status,
        "verified_digest": "a" * 64,
        "_trusted_digest": True,
    })

    with pytest.raises(ValueError, match="Trusted digest"):
        store.save_evidence(payload)


def test_reference_digest_verification_is_unavailable():
    evidence = register_reference_evidence(
        evidence_type="test_result",
        source="ci://run/123",
        claimed_digest="a" * 64,
    )
    assert evidence.digest_status == "unavailable"
    assert evidence.verified_digest is None


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside.txt"])
def test_file_evidence_rejects_unsafe_paths(tmp_path: Path, path: str):
    with pytest.raises(ValueError):
        register_file_evidence(path, artifact_root=tmp_path)


def test_file_evidence_rejects_symlink_escape_and_size_limit(tmp_path: Path):
    outside = tmp_path.parent / "outside-evalom.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        register_file_evidence("link.txt", artifact_root=tmp_path)

    large = tmp_path / "large.txt"
    large.write_text("12345", encoding="utf-8")
    with pytest.raises(ValueError):
        register_file_evidence("large.txt", artifact_root=tmp_path, max_file_size=4)


def test_file_growth_during_streaming_ingestion_fails_closed(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "growing.txt"
    artifact.write_bytes(b"initial")
    original_open = Path.open

    class GrowingHandle:
        def __init__(self, handle):
            self.handle = handle
            self.grown = False

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, size=-1):
            data = self.handle.read(size)
            if not self.grown:
                with artifact.open("ab") as writer:
                    writer.write(b"growth")
                self.grown = True
            return data

        def fileno(self):
            return self.handle.fileno()

        def write(self, data):
            return self.handle.write(data)

    def growing_open(self, *args, **kwargs):
        return GrowingHandle(original_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", growing_open)
    with pytest.raises(ValueError, match="changed during ingestion|exceeds"):
        register_file_evidence("growing.txt", artifact_root=tmp_path, max_file_size=100)


def test_workspace_isolation_and_supersession_preserve_history(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://1", workspace_id="a", task_id="task-1"
    ))
    replacement = EvidenceRecord(
        evidence_id="replacement",
        evidence_type="human_review",
        workspace_id="a",
        task_id="task-1",
        source="review://2",
        supersedes_id=original["evidence_id"],
    )
    registry.supersede(original["evidence_id"], replacement)

    assert registry.get(original["evidence_id"], workspace_id="b") is None
    assert registry.get(original["evidence_id"], workspace_id="a")["lifecycle_status"] == "superseded"
    assert registry.get("replacement", workspace_id="a")["supersedes_id"] == original["evidence_id"]
    events = store.list_audit_events(workspace_id="a")["items"]
    assert {event["event_type"] for event in events} >= {"evidence.registered", "evidence.superseded"}


def test_supersession_rejects_missing_inactive_and_cross_workspace_originals(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://1", workspace_id="a"
    ))

    replacement = EvidenceRecord(
        evidence_id="replacement-1", evidence_type="human_review", workspace_id="a",
        source="review://2", supersedes_id=original["evidence_id"],
    )
    with pytest.raises(ValueError):
        registry.supersede("missing", replacement)
    registry.supersede(original["evidence_id"], replacement)
    replacement_again = EvidenceRecord(
        evidence_id="replacement-2", evidence_type="human_review", workspace_id="a",
        source="review://3", supersedes_id=original["evidence_id"],
    )
    with pytest.raises(ValueError):
        registry.supersede(original["evidence_id"], replacement_again)
    cross_workspace = EvidenceRecord(
        evidence_id="replacement-3", evidence_type="human_review", workspace_id="b",
        source="review://4", supersedes_id=original["evidence_id"],
    )
    with pytest.raises(ValueError):
        registry.supersede(original["evidence_id"], cross_workspace)


@pytest.mark.parametrize("status", ["verified", "mismatch"])
def test_atomic_supersession_rejects_fabricated_trusted_digest_payload(tmp_path: Path, status: str):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://1", workspace_id="a"
    ))
    replacement = register_reference_evidence(
        evidence_type="human_review",
        source="review://2",
        workspace_id="a",
        evidence_id="forged-replacement",
    ).to_dict()
    replacement.update({
        "supersedes_id": original["evidence_id"],
        "digest_status": status,
        "verified_digest": "b" * 64,
        "_trusted_digest": True,
    })

    with pytest.raises(ValueError, match="Trusted digest"):
        store.supersede_evidence_and_audit(
            original["evidence_id"],
            replacement,
        )
    assert registry.get(original["evidence_id"], workspace_id="a")["lifecycle_status"] == "active"


def test_failed_audit_rolls_back_supersession(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://1", workspace_id="a"
    ))
    replacement = EvidenceRecord(
        evidence_id="replacement-audit-failure", evidence_type="human_review",
        workspace_id="a", source="review://2", supersedes_id=original["evidence_id"],
    )
    with store._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_evidence_audit BEFORE INSERT ON audit_events
            WHEN NEW.event_type = 'evidence.superseded'
            BEGIN SELECT RAISE(ABORT, 'audit failure'); END
            """
        )
    with pytest.raises(Exception, match="audit failure"):
        registry.supersede(original["evidence_id"], replacement)
    assert registry.get("replacement-audit-failure", workspace_id="a") is None
    assert registry.get(original["evidence_id"], workspace_id="a")["lifecycle_status"] == "active"


def test_failed_replacement_insert_rolls_back_supersession(tmp_path: Path):
    store = SQLiteExperimentStore(tmp_path / "evidence.db")
    registry = EvidenceRegistry(store)
    original = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://1", workspace_id="a"
    ))
    duplicate = registry.register(register_reference_evidence(
        evidence_type="human_review", source="review://duplicate", workspace_id="a",
        evidence_id="duplicate",
    ))
    replacement = EvidenceRecord(
        evidence_id=duplicate["evidence_id"], evidence_type="human_review",
        workspace_id="a", source="review://2", supersedes_id=original["evidence_id"],
    )
    with pytest.raises(ValueError):
        registry.supersede(original["evidence_id"], replacement)
    assert registry.get(original["evidence_id"], workspace_id="a")["lifecycle_status"] == "active"
    assert registry.get(duplicate["evidence_id"], workspace_id="a")["source"] == "review://duplicate"


def test_invalid_evidence_metadata_is_rejected():
    with pytest.raises(ValueError):
        EvidenceRecord(metadata={"bad": object()})
