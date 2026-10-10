# EvalOm

EvalOm is an open-source evaluation platform for LLM and RAG applications.

This repository is intentionally being built in small, business-relevant sections. The first slice focuses on the product foundation:

- evaluation dataset model
- evaluator interface
- deterministic evaluation logic
- sample-level and aggregate results
- CLI entry point

## Product direction

The goal is to help teams answer this question reliably:

> Did my prompt, model, or retrieval configuration improve or degrade quality?

The open-source version is designed to stay local-first and enterprise-friendly, without requiring a hosted platform.

The initial customer is a team already shipping an LLM or RAG feature that needs repeatable release checks. EvalOm focuses on dataset-based evaluation, regression gates, and retained experiment records; it does not host models or run customer code on a server. See the [product specification](docs/PRODUCT_SPEC.md), [architecture](docs/ARCHITECTURE.md), and [deployment guide](docs/DEPLOYMENT.md) for the product boundary and deployment assumptions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
evalom init --project-name demo
```

For API development and tests, install the development extra with `pip install -e ".[dev]"`. Save a local run to SQLite with `evalom run eval.yaml --experiment-store evalom.db`. Start the self-hosted experiment API with `evalom-api`; it binds to `127.0.0.1:8000` and uses `evalom.db` by default. Configure `EVALOM_DB_PATH` to choose another SQLite file or a PostgreSQL DSN such as `postgresql://user:pass@host:5432/evalom`. For PostgreSQL support, install the optional extra with `pip install -e ".[postgres]"` and use the local Postgres smoke-test stack:

```bash
export EVALOM_POSTGRES_PASSWORD="$(openssl rand -hex 24)"
docker compose -f docker-compose.postgres.yml up -d
export EVALOM_DB_PATH="postgresql://evalom:${EVALOM_POSTGRES_PASSWORD}@127.0.0.1:5432/evalom"
export EVALOM_ADMIN_API_KEY="$(openssl rand -hex 32)"
export EVALOM_ALLOW_REMOTE_ACCESS=0
export EVALOM_HOST=127.0.0.1
evalom-api
```

This gives a concrete local Postgres validation path before broader production deployment. To run the API/storage integration checks against that database, use a second terminal:

```bash
EVALOM_TEST_POSTGRES_DSN="postgresql://evalom:evalom@127.0.0.1:5432/evalom" \
  .venv/bin/pytest -q tests/test_postgres_integration.py
```

The integration test is skipped unless `EVALOM_TEST_POSTGRES_DSN` is set. Interactive API docs are available at `/docs`.

## API authentication and workspaces

The API keeps unauthenticated local-development behavior when no admin key is configured. Do not expose that mode to a network or use it for shared data. To enable bearer API-key authentication, configure a high-entropy bootstrap key (at least 32 characters) before starting the API:

```bash
export EVALOM_ADMIN_API_KEY="$(openssl rand -hex 32)"
export EVALOM_DEFAULT_WORKSPACE="default"
evalom-api
```

When enabled, all `/api/v1` data and key-management routes require `Authorization: Bearer <api-key>`; `/healthz` and `/readyz` remain public. Existing records are assigned to the `default` workspace during SQLite/PostgreSQL schema migration. Experiments and verification reports are isolated by workspace, and a cross-workspace record lookup returns `404`.

The bootstrap key is an admin for the configured default workspace. For remote deployments, set both `EVALOM_TLS_CERTFILE` and `EVALOM_TLS_KEYFILE` when `EVALOM_ALLOW_REMOTE_ACCESS=1`; otherwise the API refuses to start. This keeps the service fail-closed for external exposure and avoids starting an authenticated API without TLS.

Workspace admins can create and revoke scoped API keys:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/api-keys \
  -H "Authorization: Bearer $EVALOM_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"ci-member","role":"member"}'
```

The newly created key is returned once; the database stores only its SHA-256 hash. Admins may also create a workspace and its initial admin key with `POST /api/v1/workspaces`. Members can access workspace data but cannot manage API keys or create workspaces. Revoke a key with `DELETE /api/v1/api-keys/{key_id}`. Protect the bootstrap secret and use TLS when accessing the API remotely.

Workspace admins can also review operational history through `GET /api/v1/audit-events`, which records key creation/revocation, workspace creation, experiment creation, and verification events in the authenticated workspace. This gives a lightweight audit trail before broader service hardening.

To make repeated CI runs safe and auditable, each experiment may also include an `idempotency_key`. Keys are scoped to the workspace; reusing a key with the same payload returns the original experiment, while reusing it with a different payload returns `409 Conflict`. The dataset itself also supports a stable fingerprint via `EvaluationDataset.fingerprint()`; the runner stores that hash in experiment metadata so you can trace which version of the dataset produced a result.

Experiments created with `status: "running"` can be updated through `PATCH /api/v1/experiments/{id}` with final `status` (`completed` or `failed`), metrics, or metadata. Only running experiments can be updated; terminal experiment records are immutable. Each record exposes `updated_at` alongside `created_at`.

## Part 1 scope

This first implementation includes:

- a dataset abstraction for structured QA-style samples
- built-in deterministic evaluators
- a basic runner that executes an app against the dataset
- aggregate metrics and pass/fail summary
- a CLI with an `init` command

## Example usage

```python
from evalom.dataset import EvaluationDataset
from evalom.evaluators import ExactMatchEvaluator
from evalom.runner import EvaluationRunner


def app(question: str):
    return {"answer": "Paris", "contexts": ["Paris is the capital of France."]}


dataset = EvaluationDataset.from_jsonl("datasets/demo.jsonl")
runner = EvaluationRunner(dataset, [ExactMatchEvaluator()])
result = runner.run(app)
print(result.summary())
```

## Trust-layer direction

The current product foundation is being extended into an evaluation verification layer that answers a more important question than "did the model score well?": "how trustworthy is that score?"

The project now supports:

- an import/normalization layer for external evaluation payloads
- evaluator reliability checks
- dataset quality and duplicate coverage analysis
- sample-size and statistical confidence scoring
- overall release risk and recommendation reporting

This is the first step toward the broader product vision described in the project specification: a system that sits between existing evaluation infrastructure and production release decisions.

## Release policy layer

The latest milestone adds a policy layer that turns verification output into a deployment recommendation. Instead of using raw score averages alone, EvalOm evaluates the lower confidence bound from a confidence interval against explicit policy thresholds.

For binary or pass/fail metrics, the engine uses a Wilson score interval; for continuous metrics it uses a bootstrap percentile interval. The default confidence level is 95%, and release gates compare the interval's lower bound rather than a heuristic sample-size score.

Example:

```python
from evalom.policy import ReleasePolicy

policy = ReleasePolicy(
    max_risk_level="MEDIUM",
    min_confidence_score=80.0,
    min_dataset_quality=0.75,
    min_evaluator_reliability=0.80,
    min_statistical_confidence=0.75,
    confidence_level=0.95,
)

result = verification_engine.verify(run_result)
decision = policy.evaluate(result)
print(decision["decision"])  # ALLOW or BLOCK
print(decision["lower_confidence_bound"])  # lower bound used for the release gate
```

This lets the project move from "we scored this run" to "should we allow this run to gate release?" without yet adding a full multi-tenant SaaS control plane.

For CI, save the policy thresholds as JSON or YAML and pass the file to `verify`:

```bash
evalom verify report.json --policy release-policy.yaml
```

The command prints the verification and policy decision, exits with status `1` for `BLOCK`, and exits with status `0` for `ALLOW`. Without `--policy`, `verify` only reports the verification result and retains its normal success exit status.

You can also make the policy part of `eval.yaml` so every `run` applies it:

```yaml
release_policy:
  max_risk_level: MEDIUM
  min_confidence_score: 80
  min_dataset_quality: 0.75
  min_evaluator_reliability: 0.8
  min_statistical_confidence: 0.75
  confidence_level: 0.95
  max_duplicate_ratio: 0.05
  max_distribution_shift: 0.35
```

When configured, `evalom run` includes verification and policy results in its JSON report and stored experiment metadata. A `BLOCK` decision exits with status `1`, making the configured policy an effective release gate. Configurations without `release_policy` keep their existing run behavior.

The policy evaluates `lower_confidence_bound` from the verification result and compares it directly to `min_statistical_confidence`. The verification object also exposes `confidence_level` and `lower_confidence_bound` so callers can inspect the exact statistical gate being enforced.

Policy configuration is validated before use: risk levels must be `LOW`, `MEDIUM`, or `HIGH`; confidence thresholds must be between `0` and `100`; confidence levels must be between `0` and `1` or `1` and `100`; quality, reliability, duplicate, and distribution thresholds must be between `0` and `1`; and unknown keys or malformed `allowed_issues` entries are rejected.

Configured metric quality gates also fail when an evaluation produces no scores for that metric, rather than silently skipping the gate.

API verification reports can be linked to an existing experiment by passing its ID as `experiment_id` in the request. Reports can be queried by experiment or policy result, for example `GET /api/v1/verification?experiment_id=<id>&decision=BLOCK`. Existing SQLite databases are upgraded automatically when the policy-decision field is added.

## Product status

The package currently provides the local evaluation runner, dataset validation, retrieval metrics, experiment tracking, quality gates, JSON/HTML reports, a verification engine, persisted verification reports, and a lightweight release policy engine for gating deployment decisions. The API is intended for local or single-instance use; it is not yet suitable for exposure as a shared or public service. Authentication, authorization, PostgreSQL support, dataset version hashing, and collaborative review remain future work.
