# EvalOm Product Specification

## Product

EvalOm helps teams detect quality regressions in LLM and retrieval-augmented generation (RAG) applications before a prompt, model, or retrieval change reaches production. It is a self-hostable evaluation and experiment-tracking service, not a model host or chatbot builder.

## Customer and problem

The initial customer is a software team already operating an LLM feature and releasing prompt, model, or retrieval changes. The day-to-day user is an applied AI engineer or ML platform engineer; the buyer is the engineering leader accountable for reliability and release risk.

These teams often rely on manual spot checks, scattered spreadsheets, or production complaints. Those approaches make regressions difficult to reproduce, compare, and block in CI.

## Initial value proposition

Run a repeatable evaluation against a versioned dataset, inspect metric and sample-level outcomes, retain the experiment record, and fail CI when an explicit quality gate is missed. The team keeps its application code, model credentials, and evaluation data in its own environment.

## MVP workflows

1. Import and validate a JSONL, JSON, or CSV evaluation dataset.
2. Run a Python application callable against that dataset with deterministic and retrieval metrics.
3. Save a run to SQLite with its evaluator, dataset version, metrics, status, and quality-gate metadata, then expose it through the API.
4. Compare candidate reports with a baseline and enforce configured quality gates in CI.
5. Retrieve run records through a local API and produce JSON or HTML reports for review.

## MVP acceptance criteria

- A new team can install the package, initialize a project, and complete an evaluation without a hosted EvalOm account.
- A run has a stable ID, timestamp, dataset version, metric results, and user-provided model/config metadata.
- A failed configured gate has an explicit, testable CI exit status.
- Persisted records survive service restarts and can be listed and fetched by ID.
- Invalid requests and missing records return useful 4xx responses; health and storage readiness are independently observable.
- Evaluation data is not transmitted to EvalOm or any external service by default.

## Commercial boundary

The local-first core can be open source. A paid offering can focus on supported private deployment, team collaboration, access control, auditability, managed upgrades, and integrations. These are hypotheses to validate with design partners, not promises implied by the current code.

## Not in the first sellable slice

- Hosted multi-tenant service, billing, or a public signup flow.
- User authentication, organization membership, or role-based permissions.
- Model inference, prompt management, or automatic production traffic capture.
- Claims of regulatory compliance or automatic safety certification.

Do not expose the API beyond a trusted local network until authentication, authorization, transport security, secret handling, and tenant isolation have been implemented and reviewed.

## Pilot success measures

- Time for an engineer to add a representative evaluation to CI.
- Fraction of meaningful prompt/model changes evaluated before rollout.
- Regressions caught before release and time to diagnose a failed run.
- Repeat weekly usage by at least one product team.
- Willingness of a design partner to pay for deployment, collaboration, or support.