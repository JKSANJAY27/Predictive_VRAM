"""
Module 10: Experimental Harness, Baseline Policies, Ablation Framework,
and Statistical Evaluation.

This module provides the scientific evaluation layer for the Predictive VRAM
and Network-Aware Dynamic Split Inference project. It does NOT introduce any
new runtime optimisation algorithm — it reuses Modules 1-9 via configuration.

Public API summary:
  Baselines     : BaselinePolicy, BaselinePolicyFactory
  Traces        : CombinedEnvironmentTrace, NetworkTrace, MemoryTrace, ComputeTrace, TraceGenerator
  Scenarios     : ScenarioCatalog, ScenarioMetadata
  Config        : ExperimentConfig, ExperimentMatrix, TrialConfig, make_canonical_experiment
  Runner        : ExperimentRunner, TrialResult
  Metrics       : MetricCalculator, EvaluationMetrics
  Aggregation   : ResultAggregator, AggregatedResult, PairedComparison, EffectSizeResult
  Ablations     : AblationSuite, AblationSpec
  Sensitivity   : SensitivityAnalyzer, SensitivityDimension, SensitivitySweepResult
  Validation    : ResultValidator, SanityChecker, ValidationReport
  Manifest      : ExperimentManifest, DuplicateRunGuard, ConfigHasher
  Export        : CSVExporter, JSONExporter, ResearchDatasetBuilder
  Visualization : 12 scientific plotting functions
"""

from src.evaluation.ablations import AblationSpec, AblationSuite
from src.evaluation.aggregation import (
    AggregatedResult,
    EffectSizeResult,
    MetricStats,
    PairedComparison,
    ResultAggregator,
)
from src.evaluation.baselines import BaselinePolicy, BaselinePolicyFactory, PolicySpec
from src.evaluation.experiment_config import (
    ExperimentConfig,
    ExperimentMatrix,
    TrialConfig,
    make_canonical_experiment,
)
from src.evaluation.export import CSVExporter, JSONExporter, ResearchDatasetBuilder
from src.evaluation.manifest import ConfigHasher, DuplicateRunGuard, ExperimentManifest
from src.evaluation.metrics import EvaluationMetrics, MetricCalculator
from src.evaluation.runner import ExperimentRunner, TrialResult
from src.evaluation.scenario_catalog import ScenarioCatalog, ScenarioMetadata
from src.evaluation.sensitivity import (
    SensitivityAnalyzer,
    SensitivityCurve,
    SensitivityDimension,
    SensitivitySweepResult,
)
from src.evaluation.traces import (
    CombinedEnvironmentTrace,
    ComputeTrace,
    MemoryTrace,
    NetworkTrace,
    TraceGenerator,
    WorkloadTrace,
)
from src.evaluation.validation import ResultValidator, SanityChecker, ValidationReport

__all__ = [
    # Baselines
    "BaselinePolicy",
    "BaselinePolicyFactory",
    "PolicySpec",
    # Traces
    "CombinedEnvironmentTrace",
    "ComputeTrace",
    "MemoryTrace",
    "NetworkTrace",
    "WorkloadTrace",
    "TraceGenerator",
    # Scenarios
    "ScenarioCatalog",
    "ScenarioMetadata",
    # Config
    "ExperimentConfig",
    "ExperimentMatrix",
    "TrialConfig",
    "make_canonical_experiment",
    # Runner
    "ExperimentRunner",
    "TrialResult",
    # Metrics
    "EvaluationMetrics",
    "MetricCalculator",
    # Aggregation
    "MetricStats",
    "AggregatedResult",
    "PairedComparison",
    "EffectSizeResult",
    "ResultAggregator",
    # Ablations
    "AblationSpec",
    "AblationSuite",
    # Sensitivity
    "SensitivityAnalyzer",
    "SensitivityCurve",
    "SensitivityDimension",
    "SensitivitySweepResult",
    # Validation
    "ResultValidator",
    "SanityChecker",
    "ValidationReport",
    # Manifest
    "ExperimentManifest",
    "DuplicateRunGuard",
    "ConfigHasher",
    # Export
    "CSVExporter",
    "JSONExporter",
    "ResearchDatasetBuilder",
]
