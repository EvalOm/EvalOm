from __future__ import annotations

import json
import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _normalize_json_value(value: Any) -> Any:
    """Convert arbitrary Python values into JSON-serializable objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "value") and not isinstance(value, type):
        return _normalize_json_value(value.value)
    return str(value)


class SQLiteExperimentStore:
    """Small, durable experiment store for local and single-instance deployments."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL DEFAULT 'default',
                    name TEXT NOT NULL,
                    model TEXT,
                    evaluator TEXT,
                    dataset_version TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_created_at "
                "ON experiments(created_at DESC, id DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_reports (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL DEFAULT 'default',
                    experiment_id TEXT,
                    dataset_name TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    policy_decision TEXT,
                    created_at TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )
            experiment_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(experiments)").fetchall()
            }
            if "workspace_id" not in experiment_columns:
                connection.execute(
                    "ALTER TABLE experiments ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default'"
                )
            if "updated_at" not in experiment_columns:
                connection.execute("ALTER TABLE experiments ADD COLUMN updated_at TEXT")
            connection.execute(
                "UPDATE experiments SET updated_at = created_at WHERE updated_at IS NULL"
            )
            report_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(verification_reports)").fetchall()
            }
            if "workspace_id" not in report_columns:
                connection.execute(
                    "ALTER TABLE verification_reports ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default'"
                )
            report_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(verification_reports)"
                ).fetchall()
            }
            if "policy_decision" not in report_columns:
                connection.execute(
                    "ALTER TABLE verification_reports ADD COLUMN policy_decision TEXT"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_verification_reports_created_at "
                "ON verification_reports(created_at DESC, id DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_verification_reports_experiment_id "
                "ON verification_reports(experiment_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_verification_reports_policy_decision "
                "ON verification_reports(policy_decision)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_workspace_created "
                "ON experiments(workspace_id, created_at DESC, id DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_verification_workspace_created "
                "ON verification_reports(workspace_id, created_at DESC, id DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'member')),
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_workspace "
                "ON api_keys(workspace_id, revoked_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    request_hash TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            idempotency_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(run_idempotency_keys)"
                ).fetchall()
            }
            if "request_hash" not in idempotency_columns:
                connection.execute(
                    "ALTER TABLE run_idempotency_keys ADD COLUMN request_hash TEXT"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    actor_id TEXT,
                    actor_name TEXT,
                    event_type TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_events_workspace_created "
                "ON audit_events(workspace_id, created_at DESC, id DESC)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO workspaces (id, created_at) VALUES (?, ?)",
                ("default", datetime.now(timezone.utc).isoformat()),
            )

    @staticmethod
    def _hash_api_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def ensure_bootstrap_api_key(
        self,
        api_key: str,
        *,
        workspace_id: str = "default",
    ) -> None:
        if not api_key:
            raise ValueError("Bootstrap API key cannot be empty.")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO workspaces (id, created_at) VALUES (?, ?)",
                (workspace_id, now),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO api_keys (
                    id, name, workspace_id, role, key_hash, key_prefix, created_at
                ) VALUES (?, ?, ?, 'admin', ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    key_hash = excluded.key_hash,
                    key_prefix = excluded.key_prefix,
                    workspace_id = excluded.workspace_id,
                    revoked_at = NULL
                """,
                (
                    "bootstrap",
                    "bootstrap",
                    workspace_id,
                    self._hash_api_key(api_key),
                    api_key[:8],
                    now,
                ),
            )

    def create_api_key(
        self,
        *,
        workspace_id: str,
        name: str,
        role: str = "member",
    ) -> dict[str, Any]:
        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")
        if not name.strip():
            raise ValueError("API key name cannot be empty.")
        if role not in {"admin", "member"}:
            raise ValueError("API key role must be admin or member.")

        token = "evalom_" + secrets.token_urlsafe(32)
        key_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO workspaces (id, created_at) VALUES (?, ?)",
                (workspace_id, created_at),
            )
            connection.execute(
                """
                INSERT INTO api_keys (
                    id, name, workspace_id, role, key_hash, key_prefix, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key_id,
                    name.strip(),
                    workspace_id,
                    role,
                    self._hash_api_key(token),
                    token[:12],
                    created_at,
                ),
            )
        return {
            "id": key_id,
            "name": name.strip(),
            "workspace_id": workspace_id,
            "role": role,
            "created_at": created_at,
            "api_key": token,
        }
            def workspace_exists(self, workspace_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
        return row is not None

    def authenticate_api_key(self, api_key: str) -> dict[str, str] | None:
        key_hash = self._hash_api_key(api_key)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, workspace_id, role
                FROM api_keys
                WHERE key_hash = ? AND revoked_at IS NULL
                """,
                (key_hash,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "name": row["name"],
            "workspace_id": row["workspace_id"],
            "role": row["role"],
        }

    def list_api_keys(self, workspace_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, workspace_id, role, key_prefix, created_at, revoked_at
                FROM api_keys
                WHERE workspace_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (workspace_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def revoke_api_key(self, key_id: str, workspace_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE api_keys
                SET revoked_at = ?
                WHERE id = ? AND workspace_id = ? AND revoked_at IS NULL
                    AND id != 'bootstrap'
                """,
                (datetime.now(timezone.utc).isoformat(), key_id, workspace_id),
            )
        return cursor.rowcount > 0

    def log_audit_event(
        self,
        *,
        event_type: str,
        workspace_id: str,
        actor_id: str | None = None,
        actor_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not event_type or not event_type.strip():
            raise ValueError("event_type cannot be empty.")
        normalized_details = _normalize_json_value(details or {})
        event = {
            "id": str(uuid4()),
            "workspace_id": workspace_id,
            "actor_id": actor_id,
            "actor_name": actor_name,
            "event_type": event_type.strip(),
            "details": normalized_details,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    id, workspace_id, actor_id, actor_name, event_type, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["id"],
                    event["workspace_id"],
                    event["actor_id"],
                    event["actor_name"],
                    event["event_type"],
                    json.dumps(event["details"], sort_keys=True, default=str),
                    event["created_at"],
                ),
            )
        return event

    def list_audit_events(
        self,
        *,
        workspace_id: str = "default",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT * FROM audit_events
                WHERE workspace_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (workspace_id, limit, offset),
            ).fetchall()
        return {
            "items": [
                {
                    "id": row["id"],
                    "workspace_id": row["workspace_id"],
                    "actor_id": row["actor_id"],
                    "actor_name": row["actor_name"],
                    "event_type": row["event_type"],
                    "details": json.loads(row["details_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
            "total": total,
        }

    def create(
        self,
        payload: dict[str, Any],
        *,
        workspace_id: str = "default",
    ) -> dict[str, Any]:
        normalized_key, storage_key, request_hash = self._idempotency_details(
            payload,
            workspace_id,
        )
        experiment_status = payload.get("status", "completed")
        if experiment_status not in {"running", "completed", "failed"}:
            raise ValueError("status must be running, completed, or failed.")

        record = {
            "id": str(uuid4()),
            "workspace_id": workspace_id,
            "name": payload["name"],
            "model": payload.get("model"),
            "evaluator": payload.get("evaluator"),
            "dataset_version": payload.get("dataset_version"),
            "status": experiment_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": _normalize_json_value(payload.get("metrics", {})),
            "metadata": _normalize_json_value(payload.get("metadata", {})),
        }
        metrics_json = json.dumps(record["metrics"], sort_keys=True, default=str)
        metadata_json = json.dumps(record["metadata"], sort_keys=True, default=str)

        with self._connect() as connection:
            if storage_key is not None:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    """
                    SELECT experiment_id, request_hash
                    FROM run_idempotency_keys
                    WHERE idempotency_key IN (?, ?)
                    ORDER BY CASE WHEN idempotency_key = ? THEN 0 ELSE 1 END
                    LIMIT 1
                    """,
                    (storage_key, normalized_key, storage_key),
                ).fetchone()
                if existing is not None:
                    existing_record = self.get(
                        existing["experiment_id"],
                        workspace_id=workspace_id,
                    )
                    if existing_record is not None:
                        self._validate_idempotent_replay(
                            existing["request_hash"],
                            request_hash,
                        )
                        return existing_record
            connection.execute(
                """
                INSERT INTO experiments (
                    id, workspace_id, name, model, evaluator, dataset_version, status,
                    created_at, updated_at, metrics_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"], record["workspace_id"], record["name"], record["model"],
                    record["evaluator"], record["dataset_version"],
                    record["status"], record["created_at"], record["updated_at"], metrics_json,
                    metadata_json,
                ),
            )
            if storage_key is not None:
                connection.execute(
                    """
                    INSERT INTO run_idempotency_keys (
                        idempotency_key, experiment_id, request_hash, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (storage_key, record["id"], request_hash, record["created_at"]),
                )
        return record

    @staticmethod
    def _idempotency_details(
        payload: dict[str, Any],
        workspace_id: str,
    ) -> tuple[str | None, str | None, str | None]:
        idempotency_key = payload.get("idempotency_key")
        if idempotency_key is None:
            idempotency_key = payload.get("request_id")
        if idempotency_key is None:
            return None, None, None
        normalized_key = str(idempotency_key).strip()
        if not normalized_key:
            raise ValueError("idempotency_key cannot be empty.")
        scoped_digest = hashlib.sha256(
            f"{workspace_id}\0{normalized_key}".encode("utf-8")
        ).hexdigest()
        request_payload = {
            "name": payload.get("name"),
            "model": payload.get("model"),
            "evaluator": payload.get("evaluator"),
            "dataset_version": payload.get("dataset_version"),
            "status": payload.get("status", "completed"),
            "metrics": payload.get("metrics", {}),
            "metadata": payload.get("metadata", {}),
        }
        request_digest = hashlib.sha256(
            json.dumps(
                _normalize_json_value(request_payload),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return normalized_key, f"workspace:{scoped_digest}", request_digest

    @staticmethod
    def _validate_idempotent_replay(
        stored_hash: str | None,
        request_hash: str | None,
    ) -> None:
        if stored_hash is not None and stored_hash != request_hash:
            raise ValueError(
                "idempotency_key was already used with a different request payload."
            )

    def update(
        self,
        experiment_id: str,
        changes: dict[str, Any],
        *,
        workspace_id: str = "default",
    ) -> dict[str, Any] | None:
        allowed_fields = {"status", "metrics", "metadata"}
        if not changes or set(changes) - allowed_fields:
            raise ValueError("Provide status, metrics, or metadata updates.")
        if "status" in changes and changes["status"] not in {
            "running",
            "completed",
            "failed",
        }:
            raise ValueError("status must be running, completed, or failed.")

        assignments: list[str] = []
        values: list[Any] = []
        for field, column in (
            ("status", "status"),
            ("metrics", "metrics_json"),
            ("metadata", "metadata_json"),
        ):
            if field not in changes:
                continue
            value = changes[field]
            if field in {"metrics", "metadata"}:
                value = json.dumps(
                    _normalize_json_value(value),
                    sort_keys=True,
                    default=str,
                )
            assignments.append(f"{column} = ?")
            values.append(value)
        assignments.append("updated_at = ?")
        values.append(datetime.now(timezone.utc).isoformat())
        values.extend([experiment_id, workspace_id])

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT status FROM experiments WHERE id = ? AND workspace_id = ?",
                (experiment_id, workspace_id),
            ).fetchone()
            if existing is None:
                return None
            if existing["status"] != "running":
                raise ValueError("Only running experiments can be updated.")
            connection.execute(
                f"""
                UPDATE experiments
                SET {", ".join(assignments)}
                WHERE id = ? AND workspace_id = ? AND status = 'running'
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ? AND workspace_id = ?",
                (experiment_id, workspace_id),
            ).fetchone()
        return self._to_record(row)

    def get(
        self,
        experiment_id: str,
        *,
        workspace_id: str = "default",
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ? AND workspace_id = ?",
                (experiment_id, workspace_id),
            ).fetchone()
        return self._to_record(row) if row is not None else None

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        workspace_id: str = "default",
    ) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM experiments WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT id, workspace_id, name, model, evaluator, dataset_version,
                    status, created_at, updated_at, metrics_json, metadata_json
                FROM experiments
                WHERE workspace_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (workspace_id, limit, offset),
            ).fetchall()
        return {"items": [self._to_record(row) for row in rows], "total": total}

    def save_verification_report(
        self,
        payload: dict[str, Any],
        *,
        experiment_id: str | None = None,
        workspace_id: str = "default",
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("Verification payload must be a dictionary.")

        dt = datetime.now(timezone.utc).isoformat()
        verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else payload
        dataset_name = str(payload.get("dataset_name") or verification.get("dataset_name") or "unknown")
        dataset_version = str(payload.get("dataset_version") or verification.get("dataset_version") or "unknown")
        risk_level = str(payload.get("risk_level") or verification.get("risk_level") or "UNKNOWN").upper()
        confidence_score = float(payload.get("confidence_score") or verification.get("confidence_score") or 0.0)
        policy = payload.get("policy_decision")
        policy_decision = (
            str(policy.get("decision")).upper()
            if isinstance(policy, dict) and policy.get("decision") is not None
            else None
        )
        if policy_decision not in {None, "ALLOW", "BLOCK"}:
            raise ValueError("policy_decision must be ALLOW or BLOCK.")
        summary = _normalize_json_value(payload)
        record = {
            "id": str(uuid4()),
            "workspace_id": workspace_id,
            "experiment_id": experiment_id or payload.get("experiment_id"),
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "risk_level": risk_level,
            "confidence_score": confidence_score,
            "policy_decision": policy_decision,
            "created_at": dt,
            "summary": summary,
        }

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO verification_reports (
                    id, workspace_id, experiment_id, dataset_name, dataset_version,
                    risk_level, confidence_score, policy_decision, created_at, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["workspace_id"],
                    record["experiment_id"],
                    record["dataset_name"],
                    record["dataset_version"],
                    record["risk_level"],
                    record["confidence_score"],
                    record["policy_decision"],
                    record["created_at"],
                    json.dumps(record["summary"], sort_keys=True, default=str),
                ),
            )
        return record

    def get_verification_report(
        self,
        report_id: str,
        *,
        workspace_id: str = "default",
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM verification_reports WHERE id = ? AND workspace_id = ?",
                (report_id, workspace_id),
            ).fetchone()
        return self._to_verification_record(row) if row is not None else None

    def list_verification_reports(
        self,
        *,
        experiment_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        risk_level: str | None = None,
        decision: str | None = None,
        workspace_id: str = "default",
    ) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        clauses: list[str] = ["workspace_id = ?"]
        params: list[Any] = [workspace_id]
        if experiment_id is not None:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if risk_level is not None:
            clauses.append("risk_level = ?")
            params.append(str(risk_level).upper())
        if decision is not None:
            normalized_decision = str(decision).upper()
            if normalized_decision not in {"ALLOW", "BLOCK"}:
                raise ValueError("decision must be ALLOW or BLOCK.")
            clauses.append("policy_decision = ?")
            params.append(normalized_decision)

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM verification_reports{where_sql}",
                params,
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM verification_reports{where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return {
            "items": [self._to_verification_record(row) for row in rows],
            "total": total,
        }

    def check_ready(self) -> bool:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return True

    @staticmethod
    def _to_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "name": row["name"],
            "model": row["model"],
            "evaluator": row["evaluator"],
            "dataset_version": row["dataset_version"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metrics": json.loads(row["metrics_json"]),
            "metadata": json.loads(row["metadata_json"]),
        }

    @staticmethod
    def _to_verification_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "experiment_id": row["experiment_id"],
            "dataset_name": row["dataset_name"],
            "dataset_version": row["dataset_version"],
            "risk_level": row["risk_level"],
            "confidence_score": row["confidence_score"],
            "policy_decision": row["policy_decision"],
            "created_at": row["created_at"],
            "summary": json.loads(row["summary_json"]),
        }


class PostgresExperimentStore(SQLiteExperimentStore):
    """PostgreSQL-backed experiment store. This is intentionally minimal and keeps the same API surface as SQLite."""

    def __init__(self, database_url: str | Path):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "psycopg is required for the PostgreSQL backend. Install it with `pip install -e '.[postgres]'`."
            ) from exc
        self.database_url = str(database_url)
        self._psycopg = psycopg
        self._initialize()

    def _connect(self):
        return self._psycopg.connect(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiments (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL DEFAULT 'default',
                        name TEXT NOT NULL,
                        model TEXT,
                        evaluator TEXT,
                        dataset_version TEXT,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metrics_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS updated_at TEXT"
                )
                cursor.execute(
                    "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default'"
                )
                cursor.execute(
                    "UPDATE experiments SET updated_at = created_at WHERE updated_at IS NULL"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at DESC, id DESC)"
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS verification_reports (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL DEFAULT 'default',
                        experiment_id TEXT,
                        dataset_name TEXT NOT NULL,
                        dataset_version TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        confidence_score DOUBLE PRECISION NOT NULL,
                        policy_decision TEXT,
                        created_at TEXT NOT NULL,
                        summary_json TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    "ALTER TABLE verification_reports ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default'"
                )
                cursor.execute(
                    "ALTER TABLE verification_reports ADD COLUMN IF NOT EXISTS policy_decision TEXT"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_verification_reports_created_at ON verification_reports(created_at DESC, id DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_verification_reports_experiment_id ON verification_reports(experiment_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_verification_reports_policy_decision ON verification_reports(policy_decision)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_experiments_workspace_created ON experiments(workspace_id, created_at DESC, id DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_verification_workspace_created ON verification_reports(workspace_id, created_at DESC, id DESC)"
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        workspace_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('admin', 'member')),
                        key_hash TEXT NOT NULL UNIQUE,
                        key_prefix TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        revoked_at TEXT
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_api_keys_workspace ON api_keys(workspace_id, revoked_at)"
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workspaces (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_idempotency_keys (
                        idempotency_key TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        request_hash TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    "ALTER TABLE run_idempotency_keys ADD COLUMN IF NOT EXISTS request_hash TEXT"
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_events (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL,
                        actor_id TEXT,
                        actor_name TEXT,
                        event_type TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_events_workspace_created ON audit_events(workspace_id, created_at DESC, id DESC)"
                )
                cursor.execute(
                    "INSERT INTO workspaces (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    ("default", datetime.now(timezone.utc).isoformat()),
                )

    def ensure_bootstrap_api_key(self, api_key: str, *, workspace_id: str = "default") -> None:
        if not api_key:
            raise ValueError("Bootstrap API key cannot be empty.")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO workspaces (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (workspace_id, now),
                )
                cursor.execute(
                    """
                    INSERT INTO api_keys (id, name, workspace_id, role, key_hash, key_prefix, created_at)
                    VALUES (%s, %s, %s, 'admin', %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        key_hash = EXCLUDED.key_hash,
                        key_prefix = EXCLUDED.key_prefix,
                        workspace_id = EXCLUDED.workspace_id,
                        revoked_at = NULL
                    """,
                    ("bootstrap", "bootstrap", workspace_id, self._hash_api_key(api_key), api_key[:8], now),
                )

    def create_api_key(self, *, workspace_id: str, name: str, role: str = "member") -> dict[str, Any]:
        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")
        if not name.strip():
            raise ValueError("API key name cannot be empty.")
        if role not in {"admin", "member"}:
            raise ValueError("API key role must be admin or member.")
        token = "evalom_" + secrets.token_urlsafe(32)
        key_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO workspaces (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (workspace_id, created_at),
                )
                cursor.execute(
                    """
                    INSERT INTO api_keys (id, name, workspace_id, role, key_hash, key_prefix, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (key_id, name.strip(), workspace_id, role, self._hash_api_key(token), token[:12], created_at),
                )
        return {
            "id": key_id,
            "name": name.strip(),
            "workspace_id": workspace_id,
            "role": role,
            "created_at": created_at,
            "api_key": token,
        }
            def workspace_exists(self, workspace_id: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM workspaces WHERE id = %s", (workspace_id,))
                return cursor.fetchone() is not None

    def authenticate_api_key(self, api_key: str) -> dict[str, str] | None:
        key_hash = self._hash_api_key(api_key)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, name, workspace_id, role
                    FROM api_keys
                    WHERE key_hash = %s AND revoked_at IS NULL
                    """,
                    (key_hash,),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "workspace_id": row[2],
            "name": row[1],
            "role": row[3],
        }

    def list_api_keys(self, workspace_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, name, workspace_id, role, key_prefix, created_at, revoked_at
                    FROM api_keys
                    WHERE workspace_id = %s
                    ORDER BY created_at DESC, id DESC
                    """,
                    (workspace_id,),
                )
                rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "workspace_id": row[2],
                "role": row[3],
                "key_prefix": row[4],
                "created_at": row[5],
                "revoked_at": row[6],
            }
            for row in rows
        ]

    def revoke_api_key(self, key_id: str, workspace_id: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE api_keys
                    SET revoked_at = %s
                    WHERE id = %s AND workspace_id = %s AND revoked_at IS NULL AND id != 'bootstrap'
                    """,
                    (datetime.now(timezone.utc).isoformat(), key_id, workspace_id),
                )
                return cursor.rowcount > 0

    def log_audit_event(self, *, event_type: str, workspace_id: str, actor_id: str | None = None, actor_name: str | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
        if not event_type or not event_type.strip():
            raise ValueError("event_type cannot be empty.")
        normalized_details = _normalize_json_value(details or {})
        event = {
            "id": str(uuid4()),
            "workspace_id": workspace_id,
            "actor_id": actor_id,
            "actor_name": actor_name,
            "event_type": event_type.strip(),
            "details": normalized_details,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO audit_events (id, workspace_id, actor_id, actor_name, event_type, details_json, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event["id"],
                        event["workspace_id"],
                        event["actor_id"],
                        event["actor_name"],
                        event["event_type"],
                        json.dumps(event["details"], sort_keys=True, default=str),
                        event["created_at"],
                    ),
                )
        return event

    def list_audit_events(self, *, workspace_id: str = "default", limit: int = 50, offset: int = 0) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM audit_events WHERE workspace_id = %s", (workspace_id,))
                total = cursor.fetchone()[0]
                cursor.execute(
                    """
                    SELECT * FROM audit_events
                    WHERE workspace_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s OFFSET %s
                    """,
                    (workspace_id, limit, offset),
                )
                rows = cursor.fetchall()
        return {
            "items": [
                {
                    "id": row[0],
                    "workspace_id": row[1],
                    "actor_id": row[2],
                    "actor_name": row[3],
                    "event_type": row[4],
                    "details": json.loads(row[5]),
                    "created_at": row[6],
                }
                for row in rows
            ],
            "total": total,
        }

    def create(self, payload: dict[str, Any], *, workspace_id: str = "default") -> dict[str, Any]:
        normalized_key, storage_key, request_hash = self._idempotency_details(
            payload,
            workspace_id,
        )
        experiment_status = payload.get("status", "completed")
        if experiment_status not in {"running", "completed", "failed"}:
            raise ValueError("status must be running, completed, or failed.")
        record = {
            "id": str(uuid4()),
            "workspace_id": workspace_id,
            "name": payload["name"],
            "model": payload.get("model"),
            "evaluator": payload.get("evaluator"),
            "dataset_version": payload.get("dataset_version"),
            "status": experiment_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": _normalize_json_value(payload.get("metrics", {})),
            "metadata": _normalize_json_value(payload.get("metadata", {})),
        }
        metrics_json = json.dumps(record["metrics"], sort_keys=True, default=str)
        metadata_json = json.dumps(record["metadata"], sort_keys=True, default=str)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                if storage_key is not None:
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (storage_key,),
                    )
                    cursor.execute(
                        """
                        SELECT experiment_id, request_hash
                        FROM run_idempotency_keys
                        WHERE idempotency_key IN (%s, %s)
                        ORDER BY CASE WHEN idempotency_key = %s THEN 0 ELSE 1 END
                        LIMIT 1
                        """,
                        (storage_key, normalized_key, storage_key),
                    )
                    existing = cursor.fetchone()
                    if existing is not None:
                        existing_record = self.get(existing[0], workspace_id=workspace_id)
                        if existing_record is not None:
                            self._validate_idempotent_replay(
                                existing[1],
                                request_hash,
                            )
                            return existing_record
                cursor.execute(
                    """
                    INSERT INTO experiments (id, workspace_id, name, model, evaluator, dataset_version, status, created_at, updated_at, metrics_json, metadata_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record["id"], record["workspace_id"], record["name"], record["model"], record["evaluator"],
                        record["dataset_version"], record["status"], record["created_at"], record["updated_at"], metrics_json, metadata_json,
                    ),
                )
                if storage_key is not None:
                    cursor.execute(
                        "INSERT INTO run_idempotency_keys (idempotency_key, experiment_id, request_hash, created_at) VALUES (%s, %s, %s, %s)",
                        (storage_key, record["id"], request_hash, record["created_at"]),
                    )
        return record

    def update(
        self,
        experiment_id: str,
        changes: dict[str, Any],
        *,
        workspace_id: str = "default",
    ) -> dict[str, Any] | None:
        allowed_fields = {"status", "metrics", "metadata"}
        if not changes or set(changes) - allowed_fields:
            raise ValueError("Provide status, metrics, or metadata updates.")
        if "status" in changes and changes["status"] not in {
            "running",
            "completed",
            "failed",
        }:
            raise ValueError("status must be running, completed, or failed.")

        assignments: list[str] = []
        values: list[Any] = []
        for field, column in (
            ("status", "status"),
            ("metrics", "metrics_json"),
            ("metadata", "metadata_json"),
        ):
            if field not in changes:
                continue
            value = changes[field]
            if field in {"metrics", "metadata"}:
                value = json.dumps(
                    _normalize_json_value(value),
                    sort_keys=True,
                    default=str,
                )
            assignments.append(f"{column} = %s")
            values.append(value)
        assignments.append("updated_at = %s")
        values.append(datetime.now(timezone.utc).isoformat())
        values.extend([experiment_id, workspace_id])

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT status FROM experiments WHERE id = %s AND workspace_id = %s FOR UPDATE",
                    (experiment_id, workspace_id),
                )
                existing = cursor.fetchone()
                if existing is None:
                    return None
                if existing[0] != "running":
                    raise ValueError("Only running experiments can be updated.")
                cursor.execute(
                    f"""
                    UPDATE experiments
                    SET {", ".join(assignments)}
                    WHERE id = %s AND workspace_id = %s AND status = 'running'
                    """,
                    values,
                )
                cursor.execute(
                    """
                    SELECT id, workspace_id, name, model, evaluator, dataset_version,
                        status, created_at, updated_at, metrics_json, metadata_json
                    FROM experiments WHERE id = %s AND workspace_id = %s
                    """,
                    (experiment_id, workspace_id),
                )
                row = cursor.fetchone()
        return self._to_record(row)

    def get(self, experiment_id: str, *, workspace_id: str = "default") -> dict[str, Any] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, workspace_id, name, model, evaluator, dataset_version,
                        status, created_at, updated_at, metrics_json, metadata_json
                    FROM experiments WHERE id = %s AND workspace_id = %s
                    """,
                    (experiment_id, workspace_id),
                )
                row = cursor.fetchone()
        return self._to_record(row) if row is not None else None

    def list(self, limit: int = 50, offset: int = 0, *, workspace_id: str = "default") -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM experiments WHERE workspace_id = %s", (workspace_id,))
                total = cursor.fetchone()[0]
                cursor.execute(
                    """
                    SELECT id, workspace_id, name, model, evaluator, dataset_version,
                        status, created_at, updated_at, metrics_json, metadata_json
                    FROM experiments
                    WHERE workspace_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s OFFSET %s
                    """,
                    (workspace_id, limit, offset),
                )
                rows = cursor.fetchall()
        return {"items": [self._to_record(row) for row in rows], "total": total}

    def save_verification_report(self, payload: dict[str, Any], *, experiment_id: str | None = None, workspace_id: str = "default") -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("Verification payload must be a dictionary.")
        dt = datetime.now(timezone.utc).isoformat()
        verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else payload
        dataset_name = str(payload.get("dataset_name") or verification.get("dataset_name") or "unknown")
        dataset_version = str(payload.get("dataset_version") or verification.get("dataset_version") or "unknown")
        risk_level = str(payload.get("risk_level") or verification.get("risk_level") or "UNKNOWN").upper()
        confidence_score = float(payload.get("confidence_score") or verification.get("confidence_score") or 0.0)
        policy = payload.get("policy_decision")
        policy_decision = (
            str(policy.get("decision")).upper()
            if isinstance(policy, dict) and policy.get("decision") is not None
            else None
        )
        if policy_decision not in {None, "ALLOW", "BLOCK"}:
            raise ValueError("policy_decision must be ALLOW or BLOCK.")
        summary = _normalize_json_value(payload)
        record = {
            "id": str(uuid4()),
            "workspace_id": workspace_id,
            "experiment_id": experiment_id or payload.get("experiment_id"),
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "risk_level": risk_level,
            "confidence_score": confidence_score,
            "policy_decision": policy_decision,
            "created_at": dt,
            "summary": summary,
        }
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO verification_reports (id, workspace_id, experiment_id, dataset_name, dataset_version, risk_level, confidence_score, policy_decision, created_at, summary_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record["id"], record["workspace_id"], record["experiment_id"], record["dataset_name"], record["dataset_version"],
                        record["risk_level"], record["confidence_score"], record["policy_decision"], record["created_at"],
                        json.dumps(record["summary"], sort_keys=True, default=str),
                    ),
                )
        return record

    def get_verification_report(self, report_id: str, *, workspace_id: str = "default") -> dict[str, Any] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, workspace_id, experiment_id, dataset_name, dataset_version,
                        risk_level, confidence_score, policy_decision, created_at, summary_json
                    FROM verification_reports WHERE id = %s AND workspace_id = %s
                    """,
                    (report_id, workspace_id),
                )
                row = cursor.fetchone()
        return self._to_verification_record(row) if row is not None else None

    def list_verification_reports(self, *, experiment_id: str | None = None, limit: int = 50, offset: int = 0, risk_level: str | None = None, decision: str | None = None, workspace_id: str = "default") -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        clauses: list[str] = ["workspace_id = %s"]
        params: list[Any] = [workspace_id]
        if experiment_id is not None:
            clauses.append("experiment_id = %s")
            params.append(experiment_id)
        if risk_level is not None:
            clauses.append("risk_level = %s")
            params.append(str(risk_level).upper())
        if decision is not None:
            normalized_decision = str(decision).upper()
            if normalized_decision not in {"ALLOW", "BLOCK"}:
                raise ValueError("decision must be ALLOW or BLOCK.")
            clauses.append("policy_decision = %s")
            params.append(normalized_decision)
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM verification_reports{where_sql}", params)
                total = cursor.fetchone()[0]
                cursor.execute(
                    f"""
                    SELECT id, workspace_id, experiment_id, dataset_name, dataset_version,
                        risk_level, confidence_score, policy_decision, created_at, summary_json
                    FROM verification_reports{where_sql}
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s OFFSET %s
                    """,
                    [*params, limit, offset],
                )
                rows = cursor.fetchall()
        return {"items": [self._to_verification_record(row) for row in rows], "total": total}

    def check_ready(self) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        return True

    @staticmethod
    def _to_record(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "workspace_id": row[1],
            "name": row[2],
            "model": row[3],
            "evaluator": row[4],
            "dataset_version": row[5],
            "status": row[6],
            "created_at": row[7],
            "updated_at": row[8],
            "metrics": json.loads(row[9]),
            "metadata": json.loads(row[10]),
        }

    @staticmethod
    def _to_verification_record(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "workspace_id": row[1],
            "experiment_id": row[2],
            "dataset_name": row[3],
            "dataset_version": row[4],
            "risk_level": row[5],
            "confidence_score": row[6],
            "policy_decision": row[7],
            "created_at": row[8],
            "summary": json.loads(row[9]),
        }


def _is_postgres_dsn(candidate: str) -> bool:
    normalized = candidate.strip().lower()
    return normalized.startswith(("postgres://", "postgresql://", "postgresql+psycopg://"))


def create_experiment_store(database_path: str | Path | None) -> SQLiteExperimentStore | PostgresExperimentStore:
    if database_path is None:
        return SQLiteExperimentStore("evalom.db")
    candidate = str(database_path)
    if _is_postgres_dsn(candidate):
        return PostgresExperimentStore(candidate)
    return SQLiteExperimentStore(candidate)
