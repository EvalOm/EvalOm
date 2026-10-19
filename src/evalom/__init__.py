"""Public package interface for EvalOm."""

from .adapters import EvaluationImportLayer, LangSmithAdapter, LangfuseAdapter, OpenTelemetryAdapter
from .agent_assurance import (
    AgentAssuranceResult,
    AgentExecution,
    AgentTaskSpec,
    ArtifactExpectation,
    ExecutionEvent,
    verify_agent_execution,
)
from .config import EvaluationConfig, QualityGate
from .continuous import ContinuousStatisticalResult, bootstrap_mean_interval
from .dataset import EvaluationDataset, Sample
from .evaluators import BaseEvaluator, ContainsEvaluator, ExactMatchEvaluator, MRREvaluator, NDCGEvaluator, NumericToleranceEvaluator, RegexEvaluator, RetrievalRecallEvaluator
from .experiment import ExperimentRecord, ExperimentTracker
from .evidence import (
    EvidenceRecord,
    EvidenceRegistry,
    register_file_evidence,
    register_reference_evidence,
)
from .policy import ReleasePolicy, evaluate_verification_policy
from .reports import generate_html_report, generate_json_report
from .riav import (
    AssessmentStatus,
    ConfusionMatrixMetrics,
    EvalOmAssessment,
    GroundTruthComparisonResult,
    ReferenceOutcome,
    assess_ground_truth,
    calculate_confusion_matrix_metrics,
    compare_assessment_to_reference,
    compare_ground_truth,
    compute_confusion_matrix_metrics,
    normalize_assessment_decision,
    normalize_reference_label,
)
from .runner import EvaluationRunResult, EvaluationRunner
from .statistical import (
    BinaryStatisticalResult,
    calculate_binary_statistics,
    calculate_wilson_interval,
    wilson_score_interval,
)
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
    "ContinuousStatisticalResult",
    "bootstrap_mean_interval",
    "ExperimentRecord",
    "ExperimentTracker",
    "EvidenceRecord",
    "EvidenceRegistry",
    "register_file_evidence",
    "register_reference_evidence",
    "ReleasePolicy",
    "evaluate_verification_policy",
    "AssessmentStatus",
    "ReferenceOutcome",
    "EvalOmAssessment",
    "GroundTruthComparisonResult",
    "ConfusionMatrixMetrics",
    "compare_assessment_to_reference",
    "compare_ground_truth",
    "assess_ground_truth",
    "compute_confusion_matrix_metrics",
    "calculate_confusion_matrix_metrics",
    "normalize_reference_label",
    "normalize_assessment_decision",
    "generate_json_report",
    "generate_html_report",
    "EvaluationImportLayer",
    "LangfuseAdapter",
    "LangSmithAdapter",
    "OpenTelemetryAdapter",
    "AgentAssuranceResult",
    "AgentExecution",
    "AgentTaskSpec",
    "ArtifactExpectation",
    "ExecutionEvent",
    "verify_agent_execution",
    "DatasetHealth",
    "EvaluationVerificationResult",
    "VerificationEngine",
    "normalize_evaluation_record",
    "verify_evaluation_result",
    "BinaryStatisticalResult",
    "calculate_binary_statistics",
    "calculate_wilson_interval",
    "wilson_score_interval",
    "__version__",
]
