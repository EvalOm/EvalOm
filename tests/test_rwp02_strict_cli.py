from argparse import Namespace
from pathlib import Path

import pytest

from evalom.cli import (
    EXIT_ASSURANCE_FAILURE,
    EXIT_INVALID_INPUT,
    EXIT_OK,
    compare_command,
    init_command,
    run_command,
)
from evalom.config import EvaluationConfig
from evalom.runner import EvaluationRunResult


def test_configuration_rejects_unknown_nested_keys_and_empty_evaluators(tmp_path: Path):
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("dataset:\n  path: data.jsonl\n  patth: typo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown dataset key"):
        EvaluationConfig.load(unknown)

    empty = tmp_path / "empty.yaml"
    empty.write_text("dataset:\n  path: data.jsonl\nevaluators: []\n", encoding="utf-8")
    assert EvaluationConfig.load(empty).evaluators == []


def test_configuration_rejects_malformed_yaml_and_invalid_gate(tmp_path: Path):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("project:\n  name: [broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid YAML.*line"):
        EvaluationConfig.load(malformed)

    invalid_gate = tmp_path / "invalid-gate.yaml"
    invalid_gate.write_text(
        "dataset:\n  path: data.jsonl\nquality_gates:\n  exact_match:\n    minimum: 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="quality gate minimum"):
        EvaluationConfig.load(invalid_gate)


def test_run_requires_explicit_application_configuration(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("dataset:\n  path: missing.jsonl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Application configuration is required"):
        run_command(
            Namespace(
                config=str(config),
                app_module=None,
                app_function=None,
                output=None,
                html_output=None,
                experiment_store=None,
                json=False,
            )
        )


def test_init_does_not_overwrite_existing_configuration(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("project:\n  name: keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        init_command(Namespace(project_dir=str(tmp_path), project_name="replace"))
    assert "keep" in config.read_text(encoding="utf-8")


def test_compare_fail_on_regression_controls_assurance_exit(monkeypatch):
    baseline = EvaluationRunResult(
        "data", "v1", 50,
        metrics=[{"name": "accuracy", "score": 1.0}] * 50,
    )
    candidate = EvaluationRunResult(
        "data", "v1", 50,
        metrics=[{"name": "accuracy", "score": 0.0}] * 50,
    )
    monkeypatch.setattr(
        "evalom.cli._load_report",
        lambda _: ({}, baseline if _ == "baseline" else candidate),
    )

    assert compare_command(Namespace(first="baseline", second="candidate", fail_on_regression=True)) == EXIT_ASSURANCE_FAILURE
    assert compare_command(Namespace(first="baseline", second="candidate", fail_on_regression=False)) == EXIT_OK


def test_exit_code_constants_preserve_input_semantics():
    assert EXIT_INVALID_INPUT == 2
