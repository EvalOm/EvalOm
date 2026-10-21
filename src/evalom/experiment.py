from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import _normalize_json_value

_SECRET_FIELD_PARTS = ("secret", "token", "password", "credential", "api_key", "private_key")


def _without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_secrets(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in _SECRET_FIELD_PARTS)
        }
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


def metadata_fingerprint(metadata: dict[str, Any]) -> str:
    canonical = json.dumps(
        _without_secrets(_normalize_json_value(metadata)),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ExperimentRecord:
    name: str
    model: str | None = None
    evaluator: str | None = None
    dataset_version: str | None = None
    run_id: str | None = None
    report_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    def __init__(self, storage: list[dict[str, Any]] | None = None, path: str | Path | None = None):
        self.storage = storage or []
        self.path = Path(path) if path is not None else None

    def record(self, experiment: ExperimentRecord | dict[str, Any]) -> dict[str, Any]:
        if isinstance(experiment, dict):
            payload = dict(experiment)
        else:
            payload = {
                "name": experiment.name,
                "model": experiment.model,
                "evaluator": experiment.evaluator,
                "dataset_version": experiment.dataset_version,
                "run_id": experiment.run_id,
                "report_id": experiment.report_id,
                "timestamp": experiment.timestamp,
                "metadata": {
                    **experiment.metadata,
                    "metadata_fingerprint": metadata_fingerprint(experiment.metadata),
                },
            }

        if isinstance(payload.get("metadata"), dict):
            metadata = dict(payload["metadata"])
            metadata.setdefault("metadata_fingerprint", metadata_fingerprint(metadata))
            payload["metadata"] = metadata
        payload = _normalize_json_value(payload)
        self.storage.append(payload)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            temporary.write_text(json.dumps(self.storage, indent=2, default=str), encoding="utf-8")
            temporary.replace(self.path)
        return payload

    def list(self) -> list[dict[str, Any]]:
        return list(self.storage)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentTracker":
        tracker_path = Path(path)
        if tracker_path.exists():
            raw = tracker_path.read_text(encoding="utf-8").strip()
            if not raw:
                return cls(path=tracker_path)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return cls(path=tracker_path)
            if not isinstance(payload, list):
                return cls(path=tracker_path)
            return cls(storage=list(payload), path=tracker_path)
        return cls(path=tracker_path)
