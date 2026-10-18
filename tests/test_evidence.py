import hashlib
import os
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


def test_invalid_evidence_metadata_is_rejected():
    with pytest.raises(ValueError):
        EvidenceRecord(metadata={"bad": object()})

