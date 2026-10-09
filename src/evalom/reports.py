from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from .runner import EvaluationRunResult


def generate_json_report(result: EvaluationRunResult) -> dict[str, Any]:
    return {
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


def write_json_report(result: EvaluationRunResult, path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(generate_json_report(result), indent=2), encoding="utf-8")
    return report_path


def generate_html_report(result: EvaluationRunResult) -> str:
    metrics_rows = "".join(
        "<tr><td>{}</td><td>{:.4f}</td><td>{}</td></tr>".format(
            escape(str(metric["name"])),
            float(metric["score"]),
            "PASS" if metric.get("passed") else "FAIL",
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


def write_html_report(result: EvaluationRunResult, path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(generate_html_report(result), encoding="utf-8")
    return report_path
