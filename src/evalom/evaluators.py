from __future__ import annotations

import re
import math
from typing import Any


def _answer(output: dict[str, Any], evaluator: str) -> tuple[bool, Any]:
    if "answer" not in output or output["answer"] is None:
        return False, None
    return True, output["answer"]


def _contexts(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence, not a scalar.")
    return list(value)


class BaseEvaluator:
    name: str = "base"

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement evaluate().")


class ExactMatchEvaluator(BaseEvaluator):
    name = "exact_match"

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        if "expected_answer" not in sample or sample["expected_answer"] is None:
            return {
                "name": self.name,
                "score": 0.0,
                "passed": False,
                "status": "insufficient_evidence",
                "reason": "Missing expected answer reference.",
            }
        has_answer, raw_actual = _answer(output, self.name)
        if not has_answer:
            return {
                "name": self.name, "score": 0.0, "passed": False,
                "status": "insufficient_evidence",
                "reason": "Missing application answer.",
            }
        expected = str(sample["expected_answer"])
        actual = str(raw_actual)
        normalized_expected = expected.strip().lower()
        normalized_actual = actual.strip().lower()
        passed = normalized_expected == normalized_actual
        return {
            "name": self.name,
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "status": "valid",
            "expected": expected,
            "actual": actual,
        }


class ContainsEvaluator(BaseEvaluator):
    name = "contains"

    def __init__(self, required_terms: list[str] | None = None):
        if required_terms is not None and (
            not isinstance(required_terms, list)
            or any(not isinstance(term, str) or not term.strip() for term in required_terms)
        ):
            raise ValueError("required_terms must be a list of non-empty strings.")
        self.required_terms = required_terms or []

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        if not self.required_terms:
            return {
                "name": self.name,
                "score": 0.0,
                "passed": False,
                "status": "insufficient_evidence",
                "terms": [],
                "matched": [],
                "reason": "No required terms were configured.",
            }
        has_answer, raw_actual = _answer(output, self.name)
        if not has_answer:
            return {
                "name": self.name, "score": 0.0, "passed": False,
                "status": "insufficient_evidence",
                "terms": self.required_terms, "matched": [],
                "reason": "Missing application answer.",
            }
        actual = str(raw_actual)
        actual_lower = actual.lower()
        matched = []
        for term in self.required_terms:
            term_lower = term.lower()
            if term_lower in actual_lower:
                matched.append(term)
        passed = len(matched) == len(self.required_terms)
        return {
            "name": self.name,
            "score": (len(matched) / len(self.required_terms)) if self.required_terms else 1.0,
            "passed": passed,
            "status": "valid",
            "terms": self.required_terms,
            "matched": matched,
        }


class RegexEvaluator(BaseEvaluator):
    name = "regex"

    def __init__(self, pattern: str, flags: int = 0):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Regex pattern must be a non-empty string.")
        try:
            self._compiled = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc
        self.pattern = pattern
        self.flags = flags

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        has_answer, raw_actual = _answer(output, self.name)
        if not has_answer:
            return {
                "name": self.name, "score": 0.0, "passed": False,
                "status": "insufficient_evidence", "pattern": self.pattern,
                "matched": False, "reason": "Missing application answer.",
            }
        match = self._compiled.search(str(raw_actual))
        passed = match is not None
        return {
            "name": self.name,
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "status": "valid",
            "pattern": self.pattern,
            "matched": bool(match),
        }


class NumericToleranceEvaluator(BaseEvaluator):
    name = "numeric_tolerance"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = float(tolerance)
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be a finite non-negative number.")

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        expected = sample.get("expected_answer")
        actual = output.get("answer")
        if expected is None:
            return {
                "name": self.name,
                "score": 0.0,
                "passed": False,
                "status": "insufficient_evidence",
                "expected": expected,
                "actual": actual,
                "reason": "Missing expected numeric reference.",
            }
        if "answer" not in output or actual is None:
            return {
                "name": self.name, "score": 0.0, "passed": False,
                "status": "insufficient_evidence", "expected": expected,
                "actual": actual, "reason": "Missing application answer.",
            }
        try:
            expected_value = float(expected)
            actual_value = float(actual)
        except (TypeError, ValueError):
            return {
                "name": self.name,
                "score": 0.0,
                "passed": False,
                "status": "invalid",
                "expected": expected,
                "actual": actual,
            }

        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            return {
                "name": self.name, "score": 0.0, "passed": False,
                "status": "invalid", "expected": expected,
                "actual": actual, "reason": "Numeric values must be finite.",
            }
        delta = abs(actual_value - expected_value)
        passed = delta <= self.tolerance
        return {
            "name": self.name,
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "expected": expected_value,
            "actual": actual_value,
            "delta": delta,
        }


class RetrievalRecallEvaluator(BaseEvaluator):
    name = "retrieval_recall"

    def __init__(self, k: int = 5):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        expected_contexts = _contexts(sample.get("expected_context"), "expected_context")
        if "contexts" not in output or output["contexts"] is None:
            return {
                "name": self.name, "score": 0.0, "passed": False,
                "status": "insufficient_evidence", "retrieved": [],
                "reason": "Missing retrieval results.",
            }
        retrieved_contexts = _contexts(output.get("contexts"), "contexts")
        if not expected_contexts:
            return {
                "name": self.name,
                "score": 0.0,
                "passed": False,
                "status": "insufficient_evidence",
                "retrieved": retrieved_contexts[: self.k],
                "reason": "No expected relevant contexts were provided.",
            }

        relevant = set(str(item) for item in expected_contexts)
        top_k = []
        for item in retrieved_contexts:
            normalized = str(item)
            if normalized not in top_k:
                top_k.append(normalized)
        top_k = top_k[: self.k]
        hits = sum(1 for item in relevant if item in top_k)
        score = hits / len(relevant)
        passed = score >= 0.8
        return {
            "name": self.name,
            "score": score,
            "passed": passed,
            "status": "valid",
            "hits": hits,
            "expected": list(relevant),
            "retrieved": top_k,
        }


class MRREvaluator(BaseEvaluator):
    name = "mrr"

    def __init__(self, k: int = 10):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        expected_contexts = {str(item) for item in _contexts(sample.get("expected_context"), "expected_context")}
        if "contexts" not in output or output["contexts"] is None:
            return {
                "name": self.name, "score": 0.0, "passed": False,
                "status": "insufficient_evidence", "rank": None,
                "reason": "Missing retrieval results.",
            }
        retrieved_contexts = []
        for item in _contexts(output.get("contexts"), "contexts"):
            normalized = str(item)
            if normalized not in retrieved_contexts:
                retrieved_contexts.append(normalized)
        retrieved_contexts = retrieved_contexts[: self.k]
        if not expected_contexts:
            return {
                "name": self.name,
                "score": 0.0,
                "passed": False,
                "status": "insufficient_evidence",
                "rank": None,
                "reason": "No expected relevant contexts were provided.",
            }

        for rank, item in enumerate(retrieved_contexts, start=1):
            if item in expected_contexts:
                score = 1.0 / rank
                return {"name": self.name, "score": score, "passed": score >= 0.5, "rank": rank, "status": "valid"}
        return {"name": self.name, "score": 0.0, "passed": False, "rank": None, "status": "valid"}


class NDCGEvaluator(BaseEvaluator):
    name = "ndcg"

    def __init__(self, k: int = 10):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        expected_contexts = {str(item) for item in _contexts(sample.get("expected_context"), "expected_context")}
        if "contexts" not in output or output["contexts"] is None:
            return {
                "name": self.name, "score": 0.0, "passed": False,
                "status": "insufficient_evidence",
                "reason": "Missing retrieval results.",
            }
        retrieved_contexts = []
        for item in _contexts(output.get("contexts"), "contexts"):
            normalized = str(item)
            if normalized not in retrieved_contexts:
                retrieved_contexts.append(normalized)
        retrieved_contexts = retrieved_contexts[: self.k]
        if not expected_contexts:
            return {
                "name": self.name,
                "score": 0.0,
                "passed": False,
                "status": "insufficient_evidence",
                "reason": "No expected relevant contexts were provided.",
            }

        dcg = 0.0
        for idx, item in enumerate(retrieved_contexts, start=1):
            if item in expected_contexts:
                dcg += 1.0 / __import__("math").log2(idx + 1)

        idcg = 0.0
        for idx in range(1, min(len(expected_contexts), self.k) + 1):
            idcg += 1.0 / __import__("math").log2(idx + 1)

        score = dcg / idcg if idcg else 0.0
        return {
            "name": self.name, "score": score, "passed": score >= 0.5,
            "status": "valid", "dcg": dcg, "idcg": idcg,
        }


def build_evaluator(definition: dict[str, Any]) -> BaseEvaluator:
    evaluator_name = str(definition.get("name") or definition.get("type") or "exact_match")
    evaluator_type = str(definition.get("type") or evaluator_name)

    registry = {
        "exact_match": lambda: ExactMatchEvaluator(),
        "contains": lambda: ContainsEvaluator(required_terms=definition.get("terms", definition.get("required_terms", []))),
        "regex": lambda: RegexEvaluator(pattern=definition.get("pattern", "")),
        "numeric_tolerance": lambda: NumericToleranceEvaluator(tolerance=definition.get("tolerance", 0.0)),
        "retrieval_recall": lambda: RetrievalRecallEvaluator(k=definition.get("k", 5)),
        "recall": lambda: RetrievalRecallEvaluator(k=definition.get("k", 5)),
        "mrr": lambda: MRREvaluator(k=definition.get("k", 10)),
        "ndcg": lambda: NDCGEvaluator(k=definition.get("k", 10)),
    }
    factory = registry.get(evaluator_name) or registry.get(evaluator_type)
    if factory is None:
        raise ValueError(f"Unsupported evaluator: {definition}")
    evaluator = factory()
    if "name" in definition:
        evaluator.name = evaluator_name
    return evaluator
