from __future__ import annotations

import json
import math
import os
import tempfile
from html import escape
from pathlib import Path
from typing import Any

from .runner import EvaluationRunResult

REPORT_SCHEMA_VERSION = "1.0"


def generate_json_report(result: EvaluationRunResult) -> dict[str, Any]:
    metadata = dict(result.metadata)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": metadata.get("report_id"),
        "run_id": metadata.get("run_id"),
        "dataset": {
            "name": result.dataset_name,
            "version": result.dataset_version,
            "samples": result.total_samples,
        },
        "average_score": result.average_score(),
        "quality_gates": result.check_quality_gates(),
        "metrics": result.metrics,
        "sample_results": result.sample_results,
        "metadata": result.metadata,
    }
    _validate_finite(report)
    return report


def validate_report(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Report must be a JSON object.")
    version = payload.get("schema_version", REPORT_SCHEMA_VERSION)
    if version != REPORT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported report schema_version: {version!r}.")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("name"), str) or not isinstance(dataset.get("version"), str):
        raise ValueError("Report dataset must contain string name and version.")
    if not isinstance(dataset.get("samples"), int) or isinstance(dataset["samples"], bool) or dataset["samples"] < 0:
        raise ValueError("Report dataset samples must be a non-negative integer.")
    for field_name in ("metrics", "sample_results"):
        if not isinstance(payload.get(field_name, []), list):
            raise ValueError(f"Report {field_name} must be a list.")
    _validate_finite(payload)
    return payload


def _validate_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Report contains a non-finite numeric value.")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_finite(item)


def write_json_report(result: EvaluationRunResult, path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(generate_json_report(result), indent=2, sort_keys=True, allow_nan=False)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{report_path.name}.", dir=report_path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, report_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return report_path


def generate_html_report(result: EvaluationRunResult) -> str:
    metrics_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(str(metric["name"])),
            escape(_display_number(metric.get("score"))),
            escape(str(metric.get("status", "valid")) if metric.get("status") != "valid"
                   else ("PASS" if metric.get("passed") else "FAIL")),
        )
        for metric in result.metrics
    )
    gate_summary = result.check_quality_gates()
    policy_decision = result.metadata.get("policy_decision")
    passed = gate_summary["passed"] and (
        not isinstance(policy_decision, dict) or policy_decision.get("passed", True)
    )
    status = "PASS" if passed else "FAIL"
    policy_failures = (
        "".join(
            f"<li>{escape(str(failure))}</li>"
            for failure in policy_decision.get("failures", [])
        )
        if isinstance(policy_decision, dict)
        else ""
    )
    policy_section = (
        f"<h2>Release policy: {escape(str(policy_decision.get('decision', 'UNKNOWN')))}</h2>"
        f"<ul>{policy_failures}</ul>"
        if isinstance(policy_decision, dict)
        else ""
    )
    limitations = result.metadata.get("limitations", [])
    limitation_section = (
        "<h2>Limitations</h2><ul>" + "".join(
            f"<li>{escape(str(item))}</li>" for item in limitations
        ) + "</ul>"
        if isinstance(limitations, list) and limitations else ""
    )
    uncertainty = result.metadata.get("uncertainty")
    uncertainty_section = (
        f"<p><strong>Uncertainty:</strong> {escape(json.dumps(uncertainty, sort_keys=True))}</p>"
        if uncertainty is not None else ""
    )
    return f"""
    <!doctype html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\">
      <title>EvalOm Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
        th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
        th {{ background: #f3f4f6; }}
        .status {{ font-weight: bold; color: {'#15803d' if status == 'PASS' else '#b91c1c'}; }}
      </style>
    </head>
    <body>
      <h1>EvalOm Evaluation Report</h1>
      <p><strong>Dataset:</strong> {escape(result.dataset_name)} ({escape(result.dataset_version)})</p>
      <p><strong>Samples:</strong> {result.total_samples}</p>
      <p><strong>Average score:</strong> {result.average_score():.4f}</p>
      <p class=\"status\"><strong>Status:</strong> {status}</p>
      {policy_section}
      {uncertainty_section}
      {limitation_section}
      <table>
        <thead>
          <tr><th>Metric</th><th>Score</th><th>Status</th></tr>
        </thead>
        <tbody>
          {metrics_rows}
        </tbody>
      </table>
    </body>
    </html>
    """


def _display_number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    if value is None:
        return "missing"
    return str(value)


def write_html_report(result: EvaluationRunResult, path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(generate_html_report(result), encoding="utf-8")
    return report_path


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "generate_html_report",
    "generate_json_report",
    "validate_report",
    "write_html_report",
    "write_json_report",
]
