"""Public package interface for EvalOm."""

from .adapters import EvaluationImportLayer, LangSmithAdapter, LangfuseAdapter, OpenTelemetryAdapter
from .config import EvaluationConfig, QualityGate
from .dataset import EvaluationDataset, Sample
from .evaluators import BaseEvaluator, ContainsEvaluator, ExactMatchEvaluator, MRREvaluator, NDCGEvaluator, NumericToleranceEvaluator, RegexEvaluator, RetrievalRecallEvaluator
from .experiment import ExperimentRecord, ExperimentTracker
from .policy import ReleasePolicy, evaluate_verification_policy
from .reports import generate_html_report, generate_json_report
from .runner import EvaluationRunResult, EvaluationRunner
from .verification import (
    DatasetHealth,
    EvaluationVerificationResult,
    VerificationEngine,
    normalize_evaluation_record,
    verify_evaluation_result,
)

__version__ = "0.3.0"

__all__ = [
    "Sample",
    "EvaluationDataset",
    "BaseEvaluator",
    "ExactMatchEvaluator",
    "ContainsEvaluator",
    "RegexEvaluator",
    "NumericToleranceEvaluator",
    "RetrievalRecallEvaluator",
    "MRREvaluator",
    "NDCGEvaluator",
    "EvaluationRunner",
    "EvaluationRunResult",
    "EvaluationConfig",
    "QualityGate",
    "ExperimentRecord",
    "ExperimentTracker",
    "ReleasePolicy",
    "evaluate_verification_policy",
    "generate_json_report",
    "generate_html_report",
    "EvaluationImportLayer",
    "LangfuseAdapter",
    "LangSmithAdapter",
    "OpenTelemetryAdapter",
    "DatasetHealth",
    "EvaluationVerificationResult",
    "VerificationEngine",
    "normalize_evaluation_record",
    "verify_evaluation_result",
    "__version__",
]
