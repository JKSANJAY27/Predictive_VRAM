"""
Module 9: Closed-Loop Predictive Runtime Integration.

Integrates Modules 1 through 8 into an operational closed-loop runtime:
- ClosedLoopRuntime
- ControlCycle and ControlCycleRunner
- OrchestrationConfig and RuntimeTrace
- ControlScheduler
- TelemetryAgent
- Synthetic Scenarios and ScenarioRegistry
"""

from src.orchestration.adapters import TelemetryAgent
from src.orchestration.config import OrchestrationConfig
from src.orchestration.cycle import ControlCycleRunner
from src.orchestration.errors import (
    CandidateError,
    ControllerError,
    CostModelError,
    ErrorCategory,
    MigrationError,
    OrchestrationError,
    PredictionError,
    StateError,
    TelemetryError,
    VerificationError,
)
from src.orchestration.runtime import ClosedLoopRuntime
from src.orchestration.scenarios import SCENARIOS, ScenarioConditions, ScenarioRegistry, SyntheticScenario
from src.orchestration.scheduler import ControlScheduler
from src.orchestration.trace import RuntimeTrace
from src.orchestration.types import (
    ControlCycle,
    ExecutionMode,
    MeasuredOutcome,
    MigrationWindowRecord,
    OrchestrationMetrics,
    OrchestrationMode,
    PredictionLeadRecord,
    TokenRecord,
)

__all__ = [
    # Runtime
    "ClosedLoopRuntime",
    "OrchestrationConfig",
    "OrchestrationMode",
    "ExecutionMode",
    "ControlCycle",
    "ControlCycleRunner",
    "ControlScheduler",
    "RuntimeTrace",
    "MeasuredOutcome",
    "PredictionLeadRecord",
    "MigrationWindowRecord",
    "TokenRecord",
    "OrchestrationMetrics",
    "TelemetryAgent",
    # Scenarios
    "ScenarioRegistry",
    "SyntheticScenario",
    "ScenarioConditions",
    "SCENARIOS",
    # Errors
    "ErrorCategory",
    "OrchestrationError",
    "TelemetryError",
    "StateError",
    "PredictionError",
    "CandidateError",
    "CostModelError",
    "ControllerError",
    "MigrationError",
    "VerificationError",
]
