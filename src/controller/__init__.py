"""
Adaptive Partition Controller package.

Module 7 implements the predictive control policy:
    - Multi-horizon candidate selection: a* = argmin J(a)
    - Anti-thrashing stability: Gain > theta, minimum dwell time, cooldown, and hysteresis
    - Emergency safety recovery and safe fallback
    - Pure decision-making without physical model migration (produces MigrationRequest for Module 8)
"""

from src.controller.controller import AdaptivePartitionController
from src.controller.history import ControllerHistory
from src.controller.safety import SafetyEvaluation, SafetyPolicy
from src.controller.stability import StabilityChecker, StabilityResult
from src.controller.types import (
    ControlAction,
    ControlDecision,
    ControllerConfig,
    ControllerMode,
    ControllerState,
    DecisionReason,
    MigrationRequest,
    SafetyStatus,
)

__all__ = [
    "AdaptivePartitionController",
    "ControlAction",
    "ControlDecision",
    "ControllerConfig",
    "ControllerHistory",
    "ControllerMode",
    "ControllerState",
    "DecisionReason",
    "MigrationRequest",
    "SafetyEvaluation",
    "SafetyPolicy",
    "SafetyStatus",
    "StabilityChecker",
    "StabilityResult",
]
