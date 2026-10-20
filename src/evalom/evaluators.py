from __future__ import annotations

import re
from typing import Any


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
        expected = str(sample["expected_answer"])
        actual = str(output.get("answer") or "")
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
        actual = str(output.get("answer") or "")
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
        self.pattern = pattern
        self.flags = flags

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        actual = str(output.get("answer") or "")
        match = re.search(self.pattern, actual, self.flags)
        passed = match is not None
        return {
            "name": self.name,
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "pattern": self.pattern,
            "matched": bool(match),
        }


class NumericToleranceEvaluator(BaseEvaluator):
    name = "numeric_tolerance"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = float(tolerance)

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
        try:
            expected_value = float(str(expected).strip())
            actual_value = float(str(actual).strip())
        except (TypeError, ValueError):
            return {
                "name": self.name,
                "score": 0.0,
                "passed": False,
                "status": "invalid",
                "expected": expected,
                "actual": actual,
            }

        delta = abs(actual_value - expected_value)
        passed = delta <= self.tolerance
        return {
            "name": self.name,
            "score": 1.0 if passed else max(0.0, 1.0 - (delta / max(self.tolerance, 1.0))),
            "passed": passed,
            "expected": expected_value,
            "actual": actual_value,
            "delta": delta,
        }


class RetrievalRecallEvaluator(BaseEvaluator):
    name = "retrieval_recall"

    def __init__(self, k: int = 5):
        self.k = max(1, int(k))

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        expected_contexts = list(sample.get("expected_context") or [])
        retrieved_contexts = list(output.get("contexts") or [])
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
        top_k = [str(item) for item in retrieved_contexts[: self.k]]
        hits = sum(1 for item in top_k if item in relevant)
        score = hits / max(len(expected_contexts), 1)
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
        self.k = max(1, int(k))

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        expected_contexts = {str(item) for item in (sample.get("expected_context") or [])}
        retrieved_contexts = [str(item) for item in (output.get("contexts") or [])[: self.k]]
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
                return {"name": self.name, "score": score, "passed": score >= 0.5, "rank": rank}
        return {"name": self.name, "score": 0.0, "passed": False, "rank": None}


class NDCGEvaluator(BaseEvaluator):
    name = "ndcg"

    def __init__(self, k: int = 10):
        self.k = max(1, int(k))

    def evaluate(self, sample: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        expected_contexts = {str(item) for item in (sample.get("expected_context") or [])}
        retrieved_contexts = [str(item) for item in (output.get("contexts") or [])[: self.k]]
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
        return {"name": self.name, "score": score, "passed": score >= 0.5, "dcg": dcg, "idcg": idcg}


def build_evaluator(definition: dict[str, Any]) -> BaseEvaluator:
    evaluator_name = str(definition.get("name") or definition.get("type") or "exact_match")
    evaluator_type = str(definition.get("type") or evaluator_name)

    registry = {
        "exact_match": lambda: ExactMatchEvaluator(),
        "contains": lambda: ContainsEvaluator(required_terms=list(definition.get("terms") or definition.get("required_terms") or [])),
        "regex": lambda: RegexEvaluator(pattern=str(definition.get("pattern") or "")),
        "numeric_tolerance": lambda: NumericToleranceEvaluator(tolerance=float(definition.get("tolerance", 0.0))),
        "retrieval_recall": lambda: RetrievalRecallEvaluator(k=int(definition.get("k", 5))),
        "recall": lambda: RetrievalRecallEvaluator(k=int(definition.get("k", 5))),
        "mrr": lambda: MRREvaluator(k=int(definition.get("k", 10))),
        "ndcg": lambda: NDCGEvaluator(k=int(definition.get("k", 10))),
    }
    factory = registry.get(evaluator_name) or registry.get(evaluator_type)
    if factory is None:
        raise ValueError(f"Unsupported evaluator: {definition}")
    return factory()
