"""
Module 12 & 13: Research Reproducibility, Statistical Validation & Empirical GPU Validation.

Public API:
- Reproducibility : SeedManager, EnvironmentFingerprint, ResearchConfigHasher
- Manifest        : ResearchManifest, ManifestTrialRecord, CANONICAL_METRIC_DEFINITIONS
- Statistics      : StatisticalAnalyzer, MetricSummaryStats, PairedComparisonResult, ConfidenceInterval
- Comparisons     : PolicyComparisonEngine, ScenarioComparisonReport
- Ablations       : ResearchAblationRunner, ResearchAblationHypothesis, AblationAnalysisResult
- Sensitivity     : ResearchSensitivityRunner, SensitivityDimensionAnalysis, SensitivitySweepPoint
- Validation      : ResearchValidator, ResearchValidationReport, ResearchValidationIssue
- Artifacts       : PublicationArtifactGenerator
- Report          : ResearchReportGenerator
- Runner          : ResearchRunner
- Empirical Classes (Module 13): ExperimentClass, DevicePlacementMode, EvidenceQualityDescriptor
- Split Inference (Module 13) : SplitInferenceVerifier, SplitInferenceVerificationReport, PartitionPlacementAudit
- GPU Validation (Module 13)  : GPUValidationEngine, GPUValidationReport, PhaseMemoryRecord, FeasibilitySafetyEvaluation, BenchmarkMatrixRunner, WorkloadMatrixPoint
"""

from src.research.reproducibility import (
    EnvironmentFingerprint,
    ResearchConfigHasher,
    SeedManager,
)
from src.research.manifest import (
    CANONICAL_METRIC_DEFINITIONS,
    ManifestTrialRecord,
    ResearchManifest,
)
from src.research.statistics import (
    ConfidenceInterval,
    MetricSummaryStats,
    PairedComparisonResult,
    StatisticalAnalyzer,
)
from src.research.comparisons import (
    PolicyComparisonEngine,
    ScenarioComparisonReport,
)
from src.research.ablations import (
    AblationAnalysisResult,
    ResearchAblationHypothesis,
    ResearchAblationRunner,
)
from src.research.sensitivity import (
    ResearchSensitivityRunner,
    SensitivityDimensionAnalysis,
    SensitivitySweepPoint,
)
from src.research.validation import (
    ResearchValidationIssue,
    ResearchValidationReport,
    ResearchValidator,
)
from src.research.artifacts import PublicationArtifactGenerator
from src.research.report import ResearchReportGenerator
from src.research.runner import ResearchRunner

from src.research.empirical_classes import (
    DevicePlacementMode,
    EvidenceQualityDescriptor,
    ExperimentClass,
)
from src.research.split_verification import (
    PartitionPlacementAudit,
    SplitInferenceVerificationReport,
    SplitInferenceVerifier,
    TransferAuditRecord,
)
from src.research.gpu_validation import (
    BenchmarkMatrixRunner,
    FeasibilitySafetyEvaluation,
    GPUValidationEngine,
    GPUValidationReport,
    PhaseMemoryRecord,
    WorkloadMatrixPoint,
)

__all__ = [
    # Reproducibility
    "SeedManager",
    "EnvironmentFingerprint",
    "ResearchConfigHasher",
    # Manifest
    "ResearchManifest",
    "ManifestTrialRecord",
    "CANONICAL_METRIC_DEFINITIONS",
    # Statistics
    "StatisticalAnalyzer",
    "MetricSummaryStats",
    "PairedComparisonResult",
    "ConfidenceInterval",
    # Comparisons
    "PolicyComparisonEngine",
    "ScenarioComparisonReport",
    # Ablations
    "ResearchAblationRunner",
    "ResearchAblationHypothesis",
    "AblationAnalysisResult",
    # Sensitivity
    "ResearchSensitivityRunner",
    "SensitivityDimensionAnalysis",
    "SensitivitySweepPoint",
    # Validation
    "ResearchValidator",
    "ResearchValidationReport",
    "ResearchValidationIssue",
    # Artifacts & Reporting
    "PublicationArtifactGenerator",
    "ResearchReportGenerator",
    # Runner
    "ResearchRunner",
    # Empirical & GPU (Module 13)
    "ExperimentClass",
    "DevicePlacementMode",
    "EvidenceQualityDescriptor",
    "SplitInferenceVerifier",
    "SplitInferenceVerificationReport",
    "PartitionPlacementAudit",
    "TransferAuditRecord",
    "GPUValidationEngine",
    "GPUValidationReport",
    "PhaseMemoryRecord",
    "FeasibilitySafetyEvaluation",
    "BenchmarkMatrixRunner",
    "WorkloadMatrixPoint",
]
