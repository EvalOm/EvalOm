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
        if not isinstance(payload, dict):
            raise ValueError("Each sample must be an object.")
        if "id" not in payload or "input" not in payload:
            raise ValueError("Each sample must have an 'id' and 'input' field.")
        if payload["id"] is None or payload["input"] is None:
            raise ValueError("Sample 'id' and 'input' fields cannot be null.")

        expected_context = payload.get("expected_context", [])
        if expected_context is None:
            expected_context = []
        if not isinstance(expected_context, list):
            raise ValueError("Sample 'expected_context' must be a list when provided.")
        metadata = payload.get("metadata", {})
        if metadata is None or not isinstance(metadata, dict):
            raise ValueError("Sample 'metadata' must be an object when provided.")
        return cls(
            id=str(payload["id"]),
            input=str(payload["input"]),
            expected_answer=payload.get("expected_answer"),
            expected_context=list(expected_context),
            metadata=dict(metadata),
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
        self.samples = []
        for sample in samples:
            if isinstance(sample, dict):
                self.samples.append(Sample.from_dict(sample))
            elif isinstance(sample, Sample):
                self.samples.append(sample)
            else:
                raise ValueError("Dataset samples must be objects or Sample instances.")

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None, version: str = "v1") -> "EvaluationDataset":
        dataset_path = Path(path)
        samples = []
        for line_number, line in enumerate(
            dataset_path.read_text(encoding="utf-8-sig").split("\n"), start=1
        ):
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
            if not reader.fieldnames:
                raise ValueError(f"CSV dataset must include a header: {dataset_path}")
            missing = {"id", "input"} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV dataset is missing required column(s): {', '.join(sorted(missing))}."
                )
            samples = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"CSV row {row_number} has more fields than the header.")
                if any(value is None for value in row.values()):
                    raise ValueError(f"CSV row {row_number} has fewer fields than the header.")
                samples.append(Sample.from_dict({
                    "id": row["id"],
                    "input": row["input"],
                    "expected_answer": row.get("expected_answer"),
                    "expected_context": row.get("expected_context", "").split("|")
                    if row.get("expected_context") else [],
                    "metadata": {
                        key: value for key, value in row.items()
                        if key not in {"id", "input", "expected_answer", "expected_context"}
                    },
                }))
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
        identifiers: set[str] = set()
        for sample in self.samples:
            if not sample.id or not sample.input:
                raise ValueError(f"Sample {sample.id or '<unknown>'} is missing required fields.")
            if sample.id in identifiers:
                raise ValueError(f"Duplicate sample identifier: {sample.id!r}.")
            identifiers.add(sample.id)
            if not isinstance(sample.metadata, dict):
                raise ValueError(f"Sample {sample.id!r} metadata must be an object.")
            if not isinstance(sample.expected_context, list):
                raise ValueError(f"Sample {sample.id!r} expected_context must be a list.")

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
