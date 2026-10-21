# EvalOm Architecture

## Current MVP shape

```text
Python app callable + dataset + eval.yaml
                 |
                 v
       EvaluationRunner / evaluators
                 |
          run result + gates
                /           \\
               v             v
             JSON/HTML reports   SQLiteExperimentStore
                     ^
                     |
                  Experiment API
```

The Python package remains the evaluation engine. The CLI and API are separate adapters around the same domain concepts. SQLite is the default single-instance persistence layer; it avoids a service dependency for local use and makes API records durable across restarts.

## Boundaries

- `dataset.py`, `evaluators.py`, and `runner.py` own evaluation behavior and result generation.
- `config.py` owns evaluation configuration parsing.
- `storage.py` owns SQLite schema initialization and experiment persistence.
- `api.py` validates HTTP input, enforces bearer-key authentication and workspace roles when configured, exposes versioned routes, and translates missing records/storage failures into HTTP responses.
- `reports.py` remains responsible for report rendering.

## Reporting and experiment metadata

JSON reports use schema version `1.0` and retain the existing dataset,
metric, sample-result, quality-gate, and metadata fields. Imported reports
are validated before CLI comparison or verification; unsupported versions,
malformed required fields, and non-finite numbers are rejected. JSON file
exports use a sibling temporary file and atomic replacement, while HTML is
intended for human inspection and does not alter an assurance decision.

Reports may carry run and report identifiers, verification/policy results,
uncertainty, limitations, and comparison metadata. Missing or undefined
metrics are displayed as missing rather than as zero. A policy `BLOCK` and
mandatory failures remain visible in both machine-readable metadata and
human-readable output.

Experiment metadata fingerprints canonicalize JSON metadata and exclude
fields whose names indicate secrets (for example tokens, passwords, and API
keys). This supports comparison and reproduction checks without exporting
credentials; it is not proof that an experiment or report is correct.
- `cli.py` remains useful for local and CI execution; it should call shared domain/storage interfaces rather than embed server behavior.

Assurance results use `verified`, `failed`, and `insufficient_evidence` as
distinct assurance statuses. Individual agent checks retain the legacy
boolean `passed` field and also expose `passed`, `failed`, `not_run`, or
`indeterminate` in their `status` field. A release policy decision is a
separate `ALLOW` or `BLOCK` value; any assurance result whose status is not
`verified` is blocked, regardless of aggregate scores. Artifact hashes
establish integrity only and do not establish artifact correctness.

The local evidence registry stores typed, workspace-scoped evidence metadata
in the existing SQLite or PostgreSQL store. File evidence must be relative to
an explicitly supplied artifact root; paths are resolved before access, files
are streamed for SHA-256 calculation, symlink escapes are rejected, and a
configurable size limit is enforced. Claimed and verified digests remain
separate, with `verified`, `mismatch`, and `unavailable` states. Replacing
evidence creates a new record and marks the prior record `superseded`;
registration and supersession use the existing audit-event model.
Only the in-process file-ingestion factory may create `verified` or `mismatch`
records; deserializing caller-controlled dictionaries cannot establish those
states, and storage methods require an in-process capability passed separately
from the serialized payload. Historical persisted records are preserved as
stored and are not retroactively reclassified by this boundary change. This
is an application trust boundary, not a Python security sandbox:
trusted in-process code can still bypass conventions, so persisted evidence
must be treated according to its recorded provenance and validation history.

`EvidenceRegistry.validate()` rechecks current artifact bytes without
rewriting historical registration facts. It reports `matched`, `mismatch`,
`unavailable`, `invalid`, or `not_found`; a matching digest establishes
integrity only and does not establish task correctness or assurance
sufficiency. `EvidenceRegistry.delete()` performs a workspace-scoped soft
delete for active evidence, retains the database record and physical artifact,
and atomically records `evidence.deleted` with the lifecycle update. Normal
evidence listings exclude deleted records; explicit historical retrieval can
still inspect them.

Evidence lifecycle states are `active`, `superseded`, and `deleted`.
Supersession inserts replacement evidence while preserving the original
historical record. Deletion is soft deletion: it changes only the evidence
metadata and does not remove the underlying artifact. Supersession and
deletion lifecycle changes and their audit events share one database
transaction; evidence registration currently persists the record and writes
its audit event separately. These audit events are transactionally consistent,
not tamper-proof. Digest integrity does not establish correctness or
authenticity of the original claim. Permanent erasure and automated retention
policies are not implemented in this work package.

Verification reports are persisted through a dedicated report-plus-audit
operation used by the verification API. The report row and its required
`verification.created` audit event are inserted on the same SQLite or
PostgreSQL transaction; a report or audit insertion failure rolls back both.
Report IDs are database-constrained, retrieval and listing are
workspace-scoped, and serialized report summaries preserve historical
assurance results without rerunning evaluation or recalculating decisions.
The existing direct report-save method remains available for compatibility
and does not itself create an audit event; callers requiring the atomic
guarantee must use the dedicated operation. Evidence registration and its
audit event likewise remain separate operations.

Binary statistical assurance is provided by a deterministic Wilson score
confidence interval for a binomial success proportion. Results record the
eligible observation count, successes, failures, observed rate, confidence
level, bounds, method, and an explicit `insufficient_data` status for zero
observations. Confidence level is separate from any release-policy threshold.
The interval describes sampling uncertainty under independent, representative
observations; it does not prove correctness, authenticity, future reliability,
or evidence sufficiency, and does not correct bias, correlation, or
distribution shift. Release policy blocks an explicitly insufficient binary
result rather than treating zero observations as approval.

Continuous statistical assurance is available through a deterministic
percentile bootstrap of the arithmetic mean. Results preserve summary
statistics, confidence level, resample count, seed, method, and limitations.
Empty and singleton samples are explicitly insufficient for uncertainty
estimation; a configured minimum sample size may likewise mark a computable
result insufficient without fabricating an interval. The bootstrap resamples
individual observations and therefore assumes an appropriate independent,
representative sampling design. It does not correct correlated observations,
biased samples, label errors, or distribution shift. Statistical uncertainty
and evidence sufficiency remain distinct from assurance status and release
decisions.

Statistical comparisons use the explicit `candidate - baseline` difference
convention. Continuous comparisons support independent-sample normal
approximations and paired percentile bootstrap only when stable unique
identifiers establish the pairs; equal sample counts do not establish
pairing. Binary comparisons reuse Wilson single-proportion results and use an
explicit independent two-proportion approximation for the difference. A
comparison records observed values, uncertainty, evidence adequacy,
statistical support, metric direction, and practical thresholds separately.
It does not authorize release, establish equivalence, or correct sampling
bias, correlation, or distribution shift. Multiple-comparison correction is
not implemented, so results do not claim family-wise error control.

Declarative task profiles are data-only, versioned definitions of required
and optional assurance requirements for `feature_implementation`, `bug_fix`,
`test_generation`, and `code_review`. Profiles are validated strictly,
resolved deterministically, and identified by a canonical definition hash.
They reference only supported evaluators, metrics, and evidence categories;
they cannot contain executable code, arbitrary imports, commands, or trusted
filesystem paths. Missing, unavailable, or insufficient required evidence
produces `insufficient_evidence`; failed required checks produce `failed`.
Profile selection cannot override mandatory checks or manufacture evidence.

Release policies preserve the distinction between assurance, check status,
evidence adequacy, comparison outcomes, and the final decision. Policies are
versioned and declarative, validate required and optional check identifiers,
evidence requirements, comparison thresholds, and task-profile requirements.
Missing, duplicate, failed, not-run, indeterminate, invalid, or unsupported
mandatory results block release and are listed in the decision explanation.
Optional check failures do not become mandatory failures unless explicitly
configured. A policy decision does not execute evaluators or profile content,
and a favorable aggregate metric cannot override an unresolved mandatory
requirement.

Research-Informed Assurance Validation (RIAV) compares normalized binary
reference outcomes and assessments deterministically. It reports TP, FP, TN,
and FN together with precision, recall, specificity, F1, and explicit
denominators; undefined zero-denominator metrics are `null`, not perfect
scores. Missing or unverified outcomes are counted as exclusions and produce
`insufficient_evidence`, while malformed labels are rejected. RIAV attaches
the shared Wilson statistical result to the eligible correctness proportion;
this uncertainty estimate remains distinct from correctness, evidence
sufficiency, and release policy. The comparison contract assumes aligned,
independent records and does not correct label error, correlation, sampling
bias, or distribution shift.

## API contract

- `GET /healthz`: process health, independent of database availability.
- `GET /readyz`: readiness check that verifies the configured storage backend can be queried.
- `POST /api/v1/experiments`: create a record; the server assigns an ID and UTC creation timestamp.
- `PATCH /api/v1/experiments/{id}`: update a running experiment's status, metrics, or metadata; completed and failed records are immutable.
- `GET /api/v1/experiments?limit=50&offset=0`: return newest records and total count.
- `GET /api/v1/experiments/{id}`: fetch a record or return 404.
- `GET /api/v1/auth/me`: return the authenticated key identity and workspace.
- `GET|POST /api/v1/api-keys`, `DELETE /api/v1/api-keys/{key_id}`: workspace-admin key management; raw keys are returned only at creation.
- `GET /api/v1/audit-events`: workspace-admin audit stream for key, workspace, experiment, and verification actions.
- `POST /api/v1/workspaces`: create a workspace and its initial admin key.
- Verification reports and experiment records are scoped to the authenticated workspace.
- `GET /docs`: interactive OpenAPI documentation provided by FastAPI.

Idempotency keys are workspace-scoped and bind to the normalized create payload; replaying a key with a different payload returns a conflict. The API stores run metadata and metric summaries, not arbitrary application code or model credentials. Metric and metadata payloads must be JSON-serializable.

## Persistence and deployment

Set `EVALOM_DB_PATH` to choose a SQLite file, or set a PostgreSQL DSN such as `postgresql://user:pass@host:5432/evalom` to use the PostgreSQL-compatible backend. `evalom-api` binds to `127.0.0.1:8000` by default; `EVALOM_HOST` and `EVALOM_PORT` configure the listener. Setting `EVALOM_ADMIN_API_KEY` enables bearer-key authentication for API data routes. Without it, the API remains in unauthenticated local-development mode and must not be exposed to a network. SQLite migrations assign existing records to the `default` workspace. Use a durable mounted volume for the database in a container. This implementation is for local or single-instance deployments, not concurrent multi-node service operation.

The store layer is selected by the DSN/connection string: SQLite paths keep the current local mode and PostgreSQL DSNs route to the PostgreSQL-compatible `PostgresExperimentStore` implementation with the same API surface; both are intentionally still single-writer and single-instance oriented.
PostgreSQL integration tests are opt-in through `EVALOM_TEST_POSTGRES_DSN` and
are skipped when no DSN is configured.

## Production hardening sequence

1. Require TLS for any non-loopback bootstrap/API-key deployment: set `EVALOM_ALLOW_REMOTE_ACCESS=1` only with both `EVALOM_TLS_CERTFILE` and `EVALOM_TLS_KEYFILE`, and keep the secret material in a protected secret manager or mounted file system.
2. Review secret rotation and reverse-proxy configuration before exposing a shared service.
3. Run the opt-in PostgreSQL integration suite against a real database, then add concurrency/load and backup/restore drills before multi-node deployments.
4. Add evaluator version hashes, then persist full per-sample artifacts with configurable retention and redaction.
5. Add background workers only when evaluation execution is moved into the service; keep long-running model calls out of request handlers.

The initial API intentionally stores experiment records only. Evaluation execution still happens through the local runner/CLI, avoiding a premature remote code-execution surface.