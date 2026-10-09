from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .policy import ReleasePolicy


@dataclass
class QualityGate:
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QualityGate":
        if not isinstance(payload, dict):
            raise ValueError("Quality gate definition must be a dictionary.")
        return cls(
            minimum=payload.get("minimum"),
            maximum=payload.get("maximum"),
        )


@dataclass
class EvaluationConfig:
    project_name: str = "demo"
    dataset_path: str = "datasets/demo.jsonl"
    evaluators: list[dict[str, Any]] = field(default_factory=list)
    quality_gates: dict[str, QualityGate] = field(default_factory=dict)
    release_policy: ReleasePolicy | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationConfig":
        if not isinstance(payload, dict):
            raise ValueError("Configuration must be a dictionary.")

        project = payload.get("project", {})
        dataset = payload.get("dataset", {})
        quality = payload.get("quality_gates", {})
        release_policy = payload.get("release_policy", payload.get("policy"))

        evaluators = payload.get("evaluators", [])
        if not isinstance(evaluators, list):
            evaluators = [evaluators]

        return cls(
            project_name=str((project or {}).get("name", "demo") or "demo"),
            dataset_path=str((dataset or {}).get("path", "datasets/demo.jsonl") or "datasets/demo.jsonl"),
            evaluators=[dict(item) for item in evaluators],
            quality_gates={
                name: QualityGate.from_dict(value)
                for name, value in (quality or {}).items()
            },
            release_policy=(
                ReleasePolicy.from_dict(release_policy)
                if release_policy is not None
                else None
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationConfig":
        config_path = Path(path)
        raw_text = config_path.read_text(encoding="utf-8")

        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw_text) or {}
        except Exception:
            data = _simple_yaml_load(raw_text)

        return cls.from_dict(data)


def _simple_yaml_load(raw_text: str) -> dict[str, Any]:
    lines = raw_text.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]

    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_value = line.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if key_value.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Invalid YAML list item at line: {line!r}")
            item_text = key_value[2:].strip()
            if not item_text:
                item = {}
            elif ":" in item_text:
                key, value = item_text.split(":", 1)
                item = {key.strip(): _parse_scalar(value.strip())}
                parent.append(item)
                stack.append((indent, item))
                continue
            else:
                item = _parse_scalar(item_text)
            parent.append(item)
            continue

        if ":" not in key_value:
            raise ValueError(f"Invalid YAML line: {line!r}")

        key, value = key_value.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            child: Any
            next_nonempty_index = None
            for candidate in range(index + 1, len(lines)):
                if lines[candidate].strip() and not lines[candidate].lstrip().startswith("#"):
                    next_nonempty_index = candidate
                    break
            next_nonempty = lines[next_nonempty_index] if next_nonempty_index is not None else ""
            if next_nonempty and next_nonempty.lstrip().startswith("- "):
                child = []
            else:
                child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
