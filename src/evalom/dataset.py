from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


@dataclass
class Sample:
    id: str
    input: str
    expected_answer: str | None = None
    expected_context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Sample":
        if "id" not in payload or "input" not in payload:
            raise ValueError("Each sample must have an 'id' and 'input' field.")

        expected_context = payload.get("expected_context", [])
        if expected_context is None:
            expected_context = []
        return cls(
            id=str(payload["id"]),
            input=str(payload["input"]),
            expected_answer=payload.get("expected_answer"),
            expected_context=list(expected_context),
            metadata=dict(payload.get("metadata", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "input": self.input,
            "metadata": self.metadata,
        }
        if self.expected_answer is not None:
            payload["expected_answer"] = self.expected_answer
        if self.expected_context:
            payload["expected_context"] = self.expected_context
        return payload


class EvaluationDataset:
    def __init__(self, samples: Iterable[Sample | dict[str, Any]], name: str = "dataset", version: str = "v1"):
        self.name = name
        self.version = version
        self.samples = [Sample.from_dict(sample) if isinstance(sample, dict) else sample for sample in samples]

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None, version: str = "v1") -> "EvaluationDataset":
        dataset_path = Path(path)
        samples = []
        for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}") from exc
            samples.append(Sample.from_dict(payload))
        return cls(samples=samples, name=name or dataset_path.stem, version=version)

    @classmethod
    def from_json(cls, path: str | Path, name: str | None = None, version: str = "v1") -> "EvaluationDataset":
        dataset_path = Path(path)
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "samples" in payload:
            samples = [Sample.from_dict(item) for item in payload["samples"]]
        elif isinstance(payload, list):
            samples = [Sample.from_dict(item) for item in payload]
        else:
            raise ValueError("Dataset JSON must be either a list of samples or an object with a 'samples' list.")
        return cls(samples=samples, name=name or dataset_path.stem, version=version)

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None, version: str = "v1") -> "EvaluationDataset":
        dataset_path = Path(path)
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            samples = [Sample.from_dict({
                "id": row.get("id", idx),
                "input": row.get("input", ""),
                "expected_answer": row.get("expected_answer"),
                "expected_context": row.get("expected_context", "").split("|") if row.get("expected_context") else [],
                "metadata": {key: value for key, value in row.items() if key not in {"id", "input", "expected_answer", "expected_context"}},
            }) for idx, row in enumerate(reader, start=1)]
        return cls(samples=samples, name=name or dataset_path.stem, version=version)

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None, version: str = "v1") -> "EvaluationDataset":
        dataset_path = Path(path)
        suffix = dataset_path.suffix.lower()
        if suffix == ".jsonl":
            return cls.from_jsonl(dataset_path, name=name, version=version)
        if suffix == ".json":
            return cls.from_json(dataset_path, name=name, version=version)
        if suffix == ".csv":
            return cls.from_csv(dataset_path, name=name, version=version)
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    def validate(self) -> None:
        if not self.samples:
            raise ValueError("Dataset cannot be empty.")
        for sample in self.samples:
            if not sample.id or not sample.input:
                raise ValueError(f"Sample {sample.id or '<unknown>'} is missing required fields.")

    def filter(self, predicate: Callable[[Sample], bool]) -> "EvaluationDataset":
        return EvaluationDataset([sample for sample in self.samples if predicate(sample)], name=self.name, version=self.version)

    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "version": self.version,
            "samples": [sample.as_dict() for sample in self.samples],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def hash(self) -> str:
        return self.fingerprint()

    def sample(self, limit: int, seed: int | None = None) -> "EvaluationDataset":
        if limit <= 0:
            return EvaluationDataset([], name=self.name, version=self.version)
        ordered = list(self.samples)
        if seed is not None:
            random.Random(seed).shuffle(ordered)
        return EvaluationDataset(ordered[:limit], name=self.name, version=self.version)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]
