from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .verification import normalize_evaluation_record


class EvaluationImportLayer:
    """Normalize external evaluation payloads into the internal schema used by EvalOm."""

    @staticmethod
    def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        return normalize_evaluation_record(record)

    @classmethod
    def normalize_records(cls, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [cls.normalize_record(record) for record in records]

    @classmethod
    def from_json_file(cls, path: str | Path) -> list[dict[str, Any]]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return cls.normalize_records(data)
        if isinstance(data, dict):
            for key in ("records", "results", "items", "scores"):
                if isinstance(data.get(key), list):
                    return cls.normalize_records(data[key])
        raise ValueError("Unsupported import file format; expected a list of records or a wrapper object with a records/results/items/scores list.")

    @staticmethod
    def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        normalized = list(records)
        metrics = defaultdict(list)
        for item in normalized:
            metrics[str(item.get("evaluator") or "unknown")].append(float(item.get("score") or 0.0))
        return {
            "total_records": len(normalized),
            "evaluators": {
                name: {
                    "samples": len(values),
                    "average_score": sum(values) / len(values) if values else 0.0,
                }
                for name, values in metrics.items()
            },
        }


class LangfuseAdapter(EvaluationImportLayer):
    @staticmethod
    def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        nested = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if isinstance(payload.get("metadata"), dict):
            nested = {**nested, **payload["metadata"]}
        normalized = EvaluationImportLayer.normalize_record({**nested, **payload})
        if normalized.get("evaluator") == "unknown":
            normalized["evaluator"] = str(payload.get("name") or payload.get("metric") or "langfuse")
        return normalized


class LangSmithAdapter(EvaluationImportLayer):
    @staticmethod
    def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        eval_data = payload.get("eval") if isinstance(payload.get("eval"), dict) else {}
        nested = {**result, **eval_data, **payload}
        return EvaluationImportLayer.normalize_record(nested)


class OpenTelemetryAdapter(EvaluationImportLayer):
    @staticmethod
    def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        attributes = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else {}
        if isinstance(payload.get("resource"), dict):
            attributes = {**attributes, **payload["resource"]}
        flattened = {**attributes, **payload}
        return EvaluationImportLayer.normalize_record(flattened)


__all__ = [
    "EvaluationImportLayer",
    "LangfuseAdapter",
    "LangSmithAdapter",
    "OpenTelemetryAdapter",
]
