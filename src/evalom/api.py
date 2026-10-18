from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from .adapters import EvaluationImportLayer
from .policy import evaluate_verification_policy
from .runner import EvaluationRunResult
from .storage import SQLiteExperimentStore, create_experiment_store
from .verification import VerificationEngine
from . import __version__


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    model: str | None = Field(default=None, max_length=200)
    evaluator: str | None = Field(default=None, max_length=200)
    dataset_version: str | None = Field(default=None, max_length=200)
    idempotency_key: str | None = Field(default=None, max_length=200)
    status: Literal["running", "completed", "failed"] = "completed"
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentResponse(ExperimentCreate):
    id: str
    workspace_id: str
    created_at: str
    updated_at: str


class ExperimentListResponse(BaseModel):
    items: list[ExperimentResponse]
    total: int


class ExperimentUpdate(BaseModel):
    status: Literal["running", "completed", "failed"] | None = None
    metrics: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class VerificationRequest(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)
    dataset_name: str = "unknown"
    dataset_version: str = "v1"
    experiment_id: str | None = None
    sample_size: int | None = None
    evaluator_agreement: float | None = None
    human_agreement: float | None = None
    policy: dict[str, Any] | None = None


class VerificationSummaryResponse(BaseModel):
    total_records: int
    evaluators: dict[str, dict[str, Any]]
    verification: dict[str, Any]
    report_id: str | None = None
    experiment_id: str | None = None
    policy_decision: dict[str, Any] | None = None


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    role: Literal["admin", "member"] = "member"


class WorkspaceCreate(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    admin_key_name: str = Field(default="workspace-admin", min_length=1, max_length=100)


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1", "[::1]"}


def _validate_runtime_security(
    *,
    bootstrap_api_key: str | None,
    host: str,
    allow_remote_access: bool,
    certfile: str | None,
    keyfile: str | None,
) -> None:
        if _is_loopback_host(host):
        return
    if not bootstrap_api_key:
        raise ValueError(
            "Refusing to bind to a non-loopback host without authentication. Set EVALOM_ADMIN_API_KEY (at least 32 characters) or use EVALOM_HOST=127.0.0.1."
        )
    if not allow_remote_access:
        raise ValueError(
            "Remote API exposure is blocked by default. Set EVALOM_ALLOW_REMOTE_ACCESS=1 and provide TLS certificate/key files before binding to a non-loopback host."
        )
    if not certfile or not keyfile:
        raise ValueError(
            "Remote API exposure requires both EVALOM_TLS_CERTFILE and EVALOM_TLS_KEYFILE to be set when EVALOM_ALLOW_REMOTE_ACCESS=1."
        )


def create_app(
    database_path: str | Path | None = None,
    store: SQLiteExperimentStore | None = None,
    bootstrap_api_key: str | None = None,
    bootstrap_workspace_id: str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="EvalOm API",
        version=__version__,
        description="Self-hosted experiment tracking API for LLM evaluations.",
    )
    app.state.experiment_store = store
    app.state.database_path = database_path
    configured_bootstrap_key = (
        bootstrap_api_key
        if bootstrap_api_key is not None
        else os.environ.get("EVALOM_ADMIN_API_KEY")
    )
    if configured_bootstrap_key and len(configured_bootstrap_key) < 32:
        raise ValueError("EVALOM_ADMIN_API_KEY must contain at least 32 characters.")
    host = os.environ.get("EVALOM_HOST", "127.0.0.1")
    allow_remote_access = os.environ.get("EVALOM_ALLOW_REMOTE_ACCESS", "0").strip().lower() in {"1", "true", "yes", "on"}
    certfile = os.environ.get("EVALOM_TLS_CERTFILE")
    keyfile = os.environ.get("EVALOM_TLS_KEYFILE")
    _validate_runtime_security(
        bootstrap_api_key=configured_bootstrap_key,
        host=host,
        allow_remote_access=allow_remote_access,
        certfile=certfile,
        keyfile=keyfile,
    )
    app.state.bootstrap_api_key = configured_bootstrap_key
    app.state.bootstrap_workspace_id = (
        bootstrap_workspace_id
        or os.environ.get("EVALOM_DEFAULT_WORKSPACE", "default")
    )
    app.state.bootstrap_key_initialized = False

    def get_store(request: Request) -> SQLiteExperimentStore:
        current_store = request.app.state.experiment_store
        if current_store is None:
            configured_path = (
                request.app.state.database_path
                or os.environ.get("EVALOM_DB_PATH", "evalom.db")
            )
            current_store = create_experiment_store(configured_path)
            request.app.state.experiment_store = current_store
        bootstrap_key = request.app.state.bootstrap_api_key
        if bootstrap_key and not request.app.state.bootstrap_key_initialized:
            current_store.ensure_bootstrap_api_key(
                bootstrap_key,
                workspace_id=request.app.state.bootstrap_workspace_id,
            )
            request.app.state.bootstrap_key_initialized = True
        return current_store

    def get_identity(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        bootstrap_key = request.app.state.bootstrap_api_key
        if not bootstrap_key:
            return {
                "id": "local",
                "name": "local",
                "workspace_id": "default",
                "role": "admin",
            }
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="A valid bearer API key is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        identity = get_store(request).authenticate_api_key(authorization[7:])
        if identity is None:
            raise HTTPException(
                status_code=401,
                detail="A valid bearer API key is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return identity

    def require_admin(identity: dict[str, str]) -> None:
        if identity["role"] != "admin":
            raise HTTPException(status_code=403, detail="Workspace admin role required")

    @app.get("/api/v1/auth/me", tags=["authentication"])
    def get_current_identity(
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, str]:
        return identity

    @app.post("/api/v1/workspaces", status_code=status.HTTP_201_CREATED, tags=["authentication"])
    def create_workspace(
        payload: WorkspaceCreate,
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        require_admin(identity)
        store = get_store(request)
        if store.workspace_exists(payload.workspace_id):
            raise HTTPException(status_code=409, detail="Workspace already exists")
        try:
            admin_key = store.create_api_key(
                workspace_id=payload.workspace_id,
                name=payload.admin_key_name,
                role="admin",
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        store.log_audit_event(
            event_type="workspace.created",
            workspace_id=payload.workspace_id,
            actor_id=identity["id"],
            actor_name=identity["name"],
            details={"workspace_id": payload.workspace_id, "admin_key_name": payload.admin_key_name},
        )
        return {
            "workspace_id": payload.workspace_id,
            "admin_api_key": admin_key,
        }

    @app.get("/api/v1/api-keys", tags=["authentication"])
    def list_api_keys(
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> list[dict[str, Any]]:
        require_admin(identity)
        return get_store(request).list_api_keys(identity["workspace_id"])

    @app.get("/api/v1/audit-events", tags=["authentication"])
    def list_audit_events(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        require_admin(identity)
        return get_store(request).list_audit_events(
            workspace_id=identity["workspace_id"],
            limit=limit,
            offset=offset,
        )

    @app.post("/api/v1/api-keys", status_code=status.HTTP_201_CREATED, tags=["authentication"])
    def create_api_key(
        payload: ApiKeyCreate,
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        require_admin(identity)
        try:
            created = get_store(request).create_api_key(
                workspace_id=identity["workspace_id"],
                name=payload.name,
                role=payload.role,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        get_store(request).log_audit_event(
            event_type="api_key.created",
            workspace_id=identity["workspace_id"],
            actor_id=identity["id"],
            actor_name=identity["name"],
            details={"key_id": created["id"], "key_name": created["name"], "role": created["role"]},
        )
        return created

    @app.delete("/api/v1/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["authentication"])
    def revoke_api_key(
        key_id: str,
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> Response:
        require_admin(identity)
        if key_id == identity["id"]:
            raise HTTPException(status_code=400, detail="Cannot revoke the API key used for this request")
        revoked = get_store(request).revoke_api_key(key_id, identity["workspace_id"])
        if not revoked:
            raise HTTPException(status_code=404, detail="API key not found")
        get_store(request).log_audit_event(
            event_type="api_key.revoked",
            workspace_id=identity["workspace_id"],
            actor_id=identity["id"],
            actor_name=identity["name"],
            details={"key_id": key_id},
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/healthz", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["operations"])
    def readiness(request: Request) -> dict[str, str]:
        try:
            get_store(request).check_ready()
        except Exception as error:
            raise HTTPException(status_code=503, detail="Experiment storage unavailable") from error
        return {"status": "ready"}

    @app.post(
        "/api/v1/experiments",
        response_model=ExperimentResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["experiments"],
    )
    def create_experiment(
        experiment: ExperimentCreate,
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        try:
            created = get_store(request).create(
                experiment.model_dump(),
                workspace_id=identity["workspace_id"],
            )
        except ValueError as error:
            status_code = (
                status.HTTP_409_CONFLICT
                if "already used with a different request payload" in str(error)
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(status_code=status_code, detail=str(error)) from error
        get_store(request).log_audit_event(
            event_type="experiment.created",
            workspace_id=identity["workspace_id"],
            actor_id=identity["id"],
            actor_name=identity["name"],
            details={
                "experiment_id": created["id"],
                "name": created["name"],
                "dataset_version": created["dataset_version"],
                "idempotency_key": experiment.idempotency_key,
            },
        )
        return created

    @app.get(
        "/api/v1/experiments",
        response_model=ExperimentListResponse,
        tags=["experiments"],
    )
    def list_experiments(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        return get_store(request).list(
            limit=limit,
            offset=offset,
            workspace_id=identity["workspace_id"],
        )

    @app.get(
        "/api/v1/experiments/{experiment_id}",
        response_model=ExperimentResponse,
        tags=["experiments"],
    )
    def get_experiment(
        experiment_id: str,
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        experiment = get_store(request).get(
            experiment_id,
            workspace_id=identity["workspace_id"],
        )
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment

    @app.patch(
        "/api/v1/experiments/{experiment_id}",
        response_model=ExperimentResponse,
        tags=["experiments"],
    )
    def update_experiment(
        experiment_id: str,
        payload: ExperimentUpdate,
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        changes = {
            field: getattr(payload, field)
            for field in payload.model_fields_set
            if getattr(payload, field) is not None
        }
        if not changes:
            raise HTTPException(
                status_code=422,
                detail="Provide status, metrics, or metadata updates.",
            )
        try:
            updated = get_store(request).update(
                experiment_id,
                changes,
                workspace_id=identity["workspace_id"],
            )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        if updated is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        get_store(request).log_audit_event(
            event_type="experiment.updated",
            workspace_id=identity["workspace_id"],
            actor_id=identity["id"],
            actor_name=identity["name"],
            details={
                "experiment_id": experiment_id,
                "changed_fields": sorted(changes),
                "status": updated["status"],
            },
        )
        return updated

    @app.get(
        "/api/v1/verification",
        tags=["verification"],
    )
    def list_verification_reports(
        request: Request,
        experiment_id: str | None = None,
        risk_level: str | None = None,
        decision: Literal["ALLOW", "BLOCK"] | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        return get_store(request).list_verification_reports(
            experiment_id=experiment_id,
            risk_level=risk_level,
            decision=decision,
            limit=limit,
            offset=offset,
            workspace_id=identity["workspace_id"],
        )

    @app.get(
        "/api/v1/verification/{report_id}",
        tags=["verification"],
    )
    def get_verification_report(
        report_id: str,
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        report = get_store(request).get_verification_report(
            report_id,
            workspace_id=identity["workspace_id"],
        )
        if report is None:
            raise HTTPException(status_code=404, detail="Verification report not found")
        return report

    @app.post(
        "/api/v1/verification",
        response_model=VerificationSummaryResponse,
        tags=["verification"],
    )
    def verify_run(
        payload: VerificationRequest,
        request: Request,
        identity: dict[str, str] = Depends(get_identity),
    ) -> dict[str, Any]:
        if not payload.records:
            raise HTTPException(status_code=400, detail="At least one evaluation record is required.")

        store = get_store(request)
        if payload.experiment_id is not None and store.get(
            payload.experiment_id,
            workspace_id=identity["workspace_id"],
        ) is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        normalized = EvaluationImportLayer.normalize_records(payload.records)
        summary = EvaluationImportLayer.summarize(normalized)
        metric_scores = []
        sample_results = []
        grouped: dict[str, list[float]] = {}
        for item in normalized:
            evaluator = str(item["evaluator"])
            grouped.setdefault(evaluator, []).append(float(item["score"]))
            sample_result = {
                "sample_id": item["sample_id"],
                "input": item["input"],
                "output": item["output"],
                "result": [{"name": evaluator, "score": float(item["score"]), "passed": bool(item["score"] >= 0.8)}],
            }
            sample_results.append(sample_result)
            metric_scores.append({"name": evaluator, "score": float(item["score"]), "passed": bool(item["score"] >= 0.8)})

        result = EvaluationRunResult(
            dataset_name=payload.dataset_name,
            dataset_version=payload.dataset_version,
            total_samples=max(payload.sample_size or 0, len(normalized)),
            metrics=metric_scores,
            sample_results=sample_results,
        )
        verification = VerificationEngine().verify(
            result,
            sample_size=payload.sample_size,
            evaluator_agreement=payload.evaluator_agreement,
            human_agreement=payload.human_agreement,
        )

        if payload.evaluator_agreement is None and len(grouped) > 1:
            per_evaluator_scores = {
                name: sum(values) / len(values) for name, values in grouped.items()
            }
            agreement = 1.0 - (max(per_evaluator_scores.values()) - min(per_evaluator_scores.values()))
            verification.evaluator_agreement = round(max(0.0, min(1.0, agreement)), 4)
            verification.metadata["inferred_evaluator_agreement"] = verification.evaluator_agreement

        verification_summary = verification.to_dict()
        policy_decision = None
        if payload.policy is not None:
            try:
                policy_decision = evaluate_verification_policy(
                    payload.policy,
                    verification_summary,
                )
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

        report = store.save_verification_report(
            {
                "dataset_name": payload.dataset_name,
                "dataset_version": payload.dataset_version,
                "risk_level": verification.risk_level,
                "confidence_score": verification.confidence_score,
                "verification": verification_summary,
                "policy_decision": policy_decision,
                "total_records": len(normalized),
                "evaluators": summary["evaluators"],
            },
            experiment_id=payload.experiment_id,
            workspace_id=identity["workspace_id"],
        )
        store.log_audit_event(
            event_type="verification.created",
            workspace_id=identity["workspace_id"],
            actor_id=identity["id"],
            actor_name=identity["name"],
            details={
                "report_id": report["id"],
                "experiment_id": payload.experiment_id,
                "dataset_name": payload.dataset_name,
                "risk_level": verification.risk_level,
                "decision": policy_decision["decision"] if policy_decision else None,
            },
        )

        return {
            "total_records": len(normalized),
            "evaluators": summary["evaluators"],
            "verification": verification_summary,
            "report_id": report["id"],
            "experiment_id": report["experiment_id"],
            "policy_decision": policy_decision,
        }

    return app


app = create_app()


def run() -> None:
    host = os.environ.get("EVALOM_HOST", "127.0.0.1")
    allow_remote_access = os.environ.get("EVALOM_ALLOW_REMOTE_ACCESS", "0").strip().lower() in {"1", "true", "yes", "on"}
    certfile = os.environ.get("EVALOM_TLS_CERTFILE")
    keyfile = os.environ.get("EVALOM_TLS_KEYFILE")
        _validate_runtime_security(
        bootstrap_api_key=app.state.bootstrap_api_key,
        host=host,
        allow_remote_access=allow_remote_access,
        certfile=certfile,
        keyfile=keyfile,
        )
    uvicorn.run(
        "evalom.api:app",
        host=host,
        port=int(os.environ.get("EVALOM_PORT", "8000")),
        ssl_keyfile=keyfile,
        ssl_certfile=certfile,
    )
