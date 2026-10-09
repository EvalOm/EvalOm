from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

from . import __version__
from .config import EvaluationConfig
from .dataset import EvaluationDataset
from .evaluators import build_evaluator
from .experiment import ExperimentRecord, ExperimentTracker
from .policy import evaluate_verification_policy
from .reports import generate_json_report, write_html_report, write_json_report
from .runner import EvaluationRunResult, EvaluationRunner
from .storage import SQLiteExperimentStore
from .verification import VerificationEngine


def init_command(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir)
    project_name = str(getattr(args, "project_name", "demo") or "demo").strip() or "demo"
    dataset_dir = project_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / f"{project_name}.jsonl"
    if not dataset_path.exists():
        dataset_path.write_text(
            '\n'.join([
                '{"id": "1", "input": "What is the capital of France?", "expected_answer": "Paris"}',
                '{"id": "2", "input": "What is 2 + 2?", "expected_answer": "4"}',
            ]) + "\n",
            encoding="utf-8",
        )

    config_path = project_dir / "eval.yaml"
    config_path.write_text(
        f"project:\n  name: {project_name}\n\ndataset:\n  path: datasets/{project_name}.jsonl\n\nevaluators:\n  - name: exact_match\n    type: deterministic\n\nquality_gates:\n  exact_match:\n    minimum: 0.80\n",
        encoding="utf-8",
    )
    print(f"Created evaluation config at {config_path}")
    return 0


def validate_dataset_command(args: argparse.Namespace) -> int:
    dataset = EvaluationDataset.from_file(args.dataset)
    dataset.validate()
    print(f"Dataset {dataset.name} is valid ({len(dataset)} samples).")
    return 0


def _load_config(path: str | Path) -> EvaluationConfig:
    return EvaluationConfig.load(path)


def _load_app(args: argparse.Namespace):
    if getattr(args, "app_module", None) and getattr(args, "app_function", None):
        module = importlib.import_module(args.app_module)
        return getattr(module, args.app_function)

    def default_app(question: str):
        return {"answer": "", "contexts": []}

    return default_app


def run_command(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    dataset_path = (Path(args.config).parent / config.dataset_path).resolve() if not Path(config.dataset_path).is_absolute() else Path(config.dataset_path)
    dataset = EvaluationDataset.from_file(dataset_path)
    evaluators = [build_evaluator(item) for item in config.evaluators or [{"name": "exact_match"}]]
    runner = EvaluationRunner(dataset, evaluators, quality_gates={
        name: {"minimum": gate.minimum, "maximum": gate.maximum}
        for name, gate in config.quality_gates.items()
    })
    app = _load_app(args)
    result = runner.run(app, metadata={"config": config.project_name})
    quality_status = result.check_quality_gates()
    policy_status = None
    if config.release_policy is not None:
        verification = VerificationEngine().verify(result, dataset=dataset)
        policy_status = config.release_policy.evaluate(verification)
        result.metadata["verification"] = verification.to_dict()
        result.metadata["policy_decision"] = policy_status

    print(result.summary())
    if policy_status is not None:
        print(f"Release policy: {policy_status['decision']}")
        for failure in policy_status["failures"]:
            print(f"- {failure}")

    experiment_store_path = getattr(args, "experiment_store", None)
    if experiment_store_path:
        metric_averages = {}
        for metric in result.metrics:
            metric_averages.setdefault(metric["name"], []).append(float(metric["score"]))
        record = SQLiteExperimentStore(experiment_store_path).create(
            {
                "name": config.project_name,
                "evaluator": ",".join(
                    str(item.get("name", "unknown"))
                    for item in config.evaluators
                ) or "exact_match",
                "dataset_version": result.dataset_version,
                "status": "completed",
                "metrics": {
                    name: sum(scores) / len(scores)
                    for name, scores in metric_averages.items()
                },
                "metadata": {
                    "config": config.project_name,
                    "quality_gates": quality_status,
                    "verification": result.metadata.get("verification"),
                    "policy_decision": policy_status,
                },
            },
            workspace_id=os.environ.get("EVALOM_DEFAULT_WORKSPACE", "default"),
        )
        print(f"Experiment stored with ID {record['id']}")
    if args.output:
        output_path = Path(args.output)
        write_json_report(result, output_path)
        print(f"JSON report written to {output_path}")
    if args.html_output:
        html_path = Path(args.html_output)
        write_html_report(result, html_path)
        print(f"HTML report written to {html_path}")
    if policy_status is not None and not policy_status["passed"]:
        return 1
    if not quality_status["passed"] and getattr(args, "fail_on_regression", False):
        return 1
    return 0 if quality_status["passed"] else 0


def compare_command(args: argparse.Namespace) -> int:
    first = json.loads(Path(args.first).read_text(encoding="utf-8"))
    second = json.loads(Path(args.second).read_text(encoding="utf-8"))
    first_result = EvaluationRunResult(
        dataset_name=first["dataset"]["name"],
        dataset_version=first["dataset"]["version"],
        total_samples=first["dataset"]["samples"],
        metrics=first["metrics"],
        sample_results=first["sample_results"],
        quality_gates=first.get("quality_gates", {}),
    )
    second_result = EvaluationRunResult(
        dataset_name=second["dataset"]["name"],
        dataset_version=second["dataset"]["version"],
        total_samples=second["dataset"]["samples"],
        metrics=second["metrics"],
        sample_results=second["sample_results"],
        quality_gates=second.get("quality_gates", {}),
    )
    comparison = EvaluationRunner.compare_runs(first_result, second_result)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def report_command(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = EvaluationRunResult(
        dataset_name=payload["dataset"]["name"],
        dataset_version=payload["dataset"]["version"],
        total_samples=payload["dataset"]["samples"],
        metrics=payload["metrics"],
        sample_results=payload["sample_results"],
        quality_gates=payload.get("quality_gates", {}),
    )
    if args.format == "json":
        print(json.dumps(generate_json_report(result), indent=2, sort_keys=True))
    elif args.format == "html":
        if args.output:
            write_html_report(result, args.output)
            print(f"HTML report written to {args.output}")
        else:
            print(result.summary())
    return 0


def experiment_command(args: argparse.Namespace) -> int:
    tracker = ExperimentTracker.load(args.path)
    record = tracker.record({
        "name": args.name,
        "model": args.model,
        "evaluator": args.evaluator,
        "dataset_version": args.dataset_version,
        "metadata": {"report": args.report} if args.report else {},
    })
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _load_policy_file(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    raw = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(raw) or {}
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            raise ValueError("Policy file must be valid JSON or YAML")


def verify_command(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    result = EvaluationRunResult(
        dataset_name=payload["dataset"]["name"],
        dataset_version=payload["dataset"]["version"],
        total_samples=payload["dataset"]["samples"],
        metrics=payload.get("metrics", []),
        sample_results=payload.get("sample_results", []),
        quality_gates=payload.get("quality_gates", {}),
    )
    verification = VerificationEngine().verify(result)

    policy = _load_policy_file(getattr(args, "policy", None))
    out = verification.to_dict()
    if policy is not None:
        policy_result = evaluate_verification_policy(policy, out)
        out["policy_decision"] = policy_result

    print(json.dumps(out, indent=2, sort_keys=True))
    if policy is not None and not policy_result["passed"]:
        return 1
    return 0


def version_command(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="evalom")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starter evaluation config")
    init_parser.add_argument("--project-name", default="demo")
    init_parser.add_argument("--project-dir", default=".")
    init_parser.set_defaults(func=init_command)

    validate_parser = subparsers.add_parser("dataset", help="Validate a dataset file")
    validate_parser.add_argument("dataset", help="Path to dataset JSONL/JSON/CSV file")
    validate_parser.set_defaults(func=validate_dataset_command)

    run_parser = subparsers.add_parser("run", help="Run an evaluation")
    run_parser.add_argument("config", help="Path to eval.yaml")
    run_parser.add_argument("--app-module", default=None, help="Optional Python module containing the app callable")
    run_parser.add_argument("--app-function", default=None, help="Optional callable name within the module")
    run_parser.add_argument("--output", default=None, help="Optional JSON report path")
    run_parser.add_argument("--html-output", default=None, help="Optional HTML report path")
    run_parser.add_argument("--experiment-store", default=None, help="Optional SQLite path for saving this run")
    run_parser.add_argument("--fail-on-regression", action="store_true", help="Exit with status 1 if a configured quality gate fails")
    run_parser.set_defaults(func=run_command)

    compare_parser = subparsers.add_parser("compare", help="Compare two evaluation reports")
    compare_parser.add_argument("first", help="First JSON report path")
    compare_parser.add_argument("second", help="Second JSON report path")
    compare_parser.set_defaults(func=compare_command)

    report_parser = subparsers.add_parser("report", help="Render a report")
    report_parser.add_argument("report", help="Path to JSON report")
    report_parser.add_argument("--format", choices=["json", "html"], default="json")
    report_parser.add_argument("--output", default=None)
    report_parser.set_defaults(func=report_command)

    experiment_parser = subparsers.add_parser("experiment", help="Track an experiment record")
    experiment_parser.add_argument("--path", default="experiments.json", help="Where to persist experiment records")
    experiment_parser.add_argument("--name", required=True, help="Experiment name")
    experiment_parser.add_argument("--model", default=None)
    experiment_parser.add_argument("--evaluator", default=None)
    experiment_parser.add_argument("--dataset-version", default=None)
    experiment_parser.add_argument("--report", default=None)
    experiment_parser.set_defaults(func=experiment_command)

    verify_parser = subparsers.add_parser("verify", help="Assess the trustworthiness of an evaluation report")
    verify_parser.add_argument("report", help="Path to a JSON report generated by evalom run")
    verify_parser.add_argument("--policy", default=None, help="Path to a JSON/YAML release policy to evaluate against")
    verify_parser.set_defaults(func=verify_command)

    version_parser = subparsers.add_parser("version", help="Print version")
    version_parser.set_defaults(func=version_command)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
