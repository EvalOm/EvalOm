from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


EVIDENCE_TYPES = frozenset(
    {"artifact", "test_result", "execution_result", "evaluator_result", "dataset_result", "human_review", "other"}
)
DIGEST_STATUSES = frozenset({"verified", "mismatch", "unavailable", "not_applicable"})
LIFECYCLE_STATUSES = frozenset({"active", "superseded", "deleted"})
MAX_METADATA_KEYS = 50
MAX_METADATA_BYTES = 16_384
MAX_EVIDENCE_ID_LENGTH = 200
MAX_SOURCE_LENGTH = 500
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024


def _text(value: Any, name: str, maximum: int, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    if required and not value.strip():
        raise ValueError(f"{name} cannot be empty.")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds the maximum length of {maximum}.")
    return value


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > MAX_METADATA_KEYS:
        raise ValueError("metadata must be a dictionary with at most 50 keys.")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain JSON-serializable values.") from exc
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds the maximum size of {MAX_METADATA_BYTES} bytes.")
    return dict(value)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str = field(default_factory=lambda: str(uuid4()))
    evidence_type: str = "other"
    workspace_id: str = "default"
    task_id: str | None = None
    execution_id: str | None = None
    source: str | None = None
    claimed_digest: str | None = None
    verified_digest: str | None = None
    digest_status: str = "not_applicable"
    metadata: dict[str, Any] = field(default_factory=dict)
    lifecycle_status: str = "active"
    supersedes_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _text(self.evidence_id, "evidence_id", MAX_EVIDENCE_ID_LENGTH)
        _text(self.evidence_type, "evidence_type", 50)
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"Unsupported evidence type: {self.evidence_type!r}.")
        _text(self.workspace_id, "workspace_id", 100)
        _text(self.task_id, "task_id", 200, required=False)
        _text(self.execution_id, "execution_id", 200, required=False)
        _text(self.source, "source", MAX_SOURCE_LENGTH, required=False)
        for name, digest in (("claimed_digest", self.claimed_digest), ("verified_digest", self.verified_digest)):
            if digest is not None and (len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest)):
                raise ValueError(f"{name} must be a 64-character hexadecimal digest.")
        if self.digest_status not in DIGEST_STATUSES:
            raise ValueError(f"Unsupported digest status: {self.digest_status!r}.")
        if self.digest_status == "verified" and not self.verified_digest:
            raise ValueError("verified digest status requires a verified_digest.")
        _metadata(self.metadata)
        if self.lifecycle_status not in LIFECYCLE_STATUSES:
            raise ValueError(f"Unsupported lifecycle status: {self.lifecycle_status!r}.")
        _text(self.supersedes_id, "supersedes_id", MAX_EVIDENCE_ID_LENGTH, required=False)
        _text(self.created_at, "created_at", 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "workspace_id": self.workspace_id,
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "source": self.source,
            "claimed_digest": self.claimed_digest,
            "verified_digest": self.verified_digest,
            "digest_status": self.digest_status,
            "metadata": dict(self.metadata),
            "lifecycle_status": self.lifecycle_status,
            "supersedes_id": self.supersedes_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRecord":
        if not isinstance(payload, dict):
            raise ValueError("Evidence record must be a dictionary.")
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


def _safe_relative_path(root: Path, relative_path: str) -> Path:
    path = Path(_text(relative_path, "path", MAX_SOURCE_LENGTH) or "")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence path must be relative and cannot contain '..'.")
    root = root.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence path escapes the configured root.") from exc
    return candidate


def register_file_evidence(
    path: str,
    *,
    artifact_root: str | Path,
    workspace_id: str = "default",
    task_id: str | None = None,
    execution_id: str | None = None,
    evidence_id: str | None = None,
    claimed_digest: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> EvidenceRecord:
    if isinstance(max_file_size, bool) or not isinstance(max_file_size, int) or max_file_size < 0:
        raise ValueError("max_file_size must be a non-negative integer.")
    candidate = _safe_relative_path(Path(artifact_root), path)
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("evidence path must reference a regular file inside the configured root.")
    size = candidate.stat().st_size
    if size > max_file_size:
        raise ValueError(f"evidence file exceeds the maximum size of {max_file_size} bytes.")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    verified = digest.hexdigest()
    status = "verified" if claimed_digest is None or claimed_digest.lower() == verified else "mismatch"
    return EvidenceRecord(
        evidence_id=evidence_id or str(uuid4()),
        evidence_type="artifact",
        workspace_id=workspace_id,
        task_id=task_id,
        execution_id=execution_id,
        source=str(Path(path)),
        claimed_digest=claimed_digest,
        verified_digest=verified,
        digest_status=status,
        metadata={**(metadata or {}), "size_bytes": size},
    )


def register_reference_evidence(
    *,
    evidence_type: str,
    source: str,
    workspace_id: str = "default",
    task_id: str | None = None,
    execution_id: str | None = None,
    claimed_digest: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_type=evidence_type,
        workspace_id=workspace_id,
        task_id=task_id,
        execution_id=execution_id,
        source=source,
        claimed_digest=claimed_digest,
        digest_status="unavailable" if claimed_digest else "not_applicable",
        metadata=metadata or {},
    )


class EvidenceRegistry:
    def __init__(self, store: Any):
        self.store = store

    def register(self, evidence: EvidenceRecord, *, actor_id: str | None = None, actor_name: str | None = None) -> dict[str, Any]:
        record = self.store.save_evidence(evidence.to_dict())
        self.store.log_audit_event(
            event_type="evidence.registered",
            workspace_id=evidence.workspace_id,
            actor_id=actor_id,
            actor_name=actor_name,
            details={"evidence_id": evidence.evidence_id, "evidence_type": evidence.evidence_type},
        )
        return record

    def get(self, evidence_id: str, *, workspace_id: str = "default") -> dict[str, Any] | None:
        return self.store.get_evidence(evidence_id, workspace_id=workspace_id)

    def supersede(self, evidence_id: str, replacement: EvidenceRecord, *, actor_id: str | None = None) -> dict[str, Any]:
        if replacement.supersedes_id != evidence_id:
            raise ValueError("replacement.supersedes_id must identify the evidence being superseded.")
        if self.get(evidence_id, workspace_id=replacement.workspace_id) is None:
            raise ValueError("Evidence to supersede was not found in the workspace.")
        result = self.register(replacement, actor_id=actor_id)
        if not self.store.mark_evidence_superseded(evidence_id, workspace_id=replacement.workspace_id):
            raise ValueError("Evidence to supersede was not found in the workspace.")
        self.store.log_audit_event(
            event_type="evidence.superseded",
            workspace_id=replacement.workspace_id,
            actor_id=actor_id,
            details={"evidence_id": evidence_id, "replacement_id": replacement.evidence_id},
        )
        return result


__all__ = [
    "DEFAULT_MAX_FILE_SIZE",
    "DIGEST_STATUSES",
    "EVIDENCE_TYPES",
    "EvidenceRecord",
    "EvidenceRegistry",
    "register_file_evidence",
    "register_reference_evidence",
]
