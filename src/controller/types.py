"""
Data types and representations for Module 7: Adaptive Predictive Partition Controller.

Defines control actions, decision reason codes, migration requests, controller states,
and configuration contracts with full serialization and auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.runtime.partition import PartitionPlan


class ControlAction(str, Enum):
    """Decision action produced by the controller."""
    KEEP_CURRENT = "keep_current"
    SWITCH = "switch"
    SAFE_FALLBACK = "safe_fallback"
    NO_ACTION = "no_action"


class DecisionReason(str, Enum):
    """Machine-readable reason code explaining the controller decision."""
    STATIC_POLICY = "static_policy"
    BEST_CANDIDATE_IS_CURRENT = "best_candidate_is_current"
    NO_IMPROVEMENT = "no_improvement"
    BELOW_THRESHOLD = "below_threshold"
    COOLDOWN_ACTIVE = "cooldown_active"
    DWELL_TIME_ACTIVE = "dwell_time_active"
    HYSTERESIS_NOT_SATISFIED = "hysteresis_not_satisfied"
    CURRENT_PLAN_UNSAFE = "current_plan_unsafe"
    EMERGENCY_RECOVERY = "emergency_recovery"
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    UNKNOWN_RESOURCE = "unknown_resource"
    SAFETY_FALLBACK = "safety_fallback"
    INVALID_FORECAST = "invalid_forecast"
    INSUFFICIENT_HISTORY = "insufficient_history"
    PREDICTIVE_SWITCH_APPROVED = "predictive_switch_approved"
    REACTIVE_SWITCH_APPROVED = "reactive_switch_approved"


class ControllerMode(str, Enum):
    """Operational mode of the partition controller."""
    STATIC = "static"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"


class SafetyStatus(str, Enum):
    """Safety state of the currently active partition."""
    SAFE = "safe"
    WARNING = "warning"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class MigrationRequest:
    """
    Strongly typed request produced for Module 8 (MigrationManager).

    Contains all structural delta and sizing information needed to physically
    execute a repartition without embedding migration logic inside Module 7.
    """
    source_plan: PartitionPlan
    target_plan: PartitionPlan
    source_plan_id: str
    target_plan_id: str
    expected_gain: float
    switching_cost: float
    controller_cycle: int
    timestamp: float
    changed_layers: List[int]
    affected_tiers: List[str]
    estimated_kv_transfer_bytes: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_plan_id": self.source_plan_id,
            "target_plan_id": self.target_plan_id,
            "expected_gain": self.expected_gain,
            "switching_cost": self.switching_cost,
            "controller_cycle": self.controller_cycle,
            "timestamp": self.timestamp,
            "changed_layers": self.changed_layers,
            "affected_tiers": self.affected_tiers,
            "estimated_kv_transfer_bytes": self.estimated_kv_transfer_bytes,
            "reason": self.reason,
            "source_plan": self.source_plan.to_dict(),
            "target_plan": self.target_plan.to_dict(),
        }


@dataclass(frozen=True)
class ControlDecision:
    """
    Comprehensive, inspectable control decision produced by the controller.
    """
    action: ControlAction
    current_plan_id: str
    selected_plan_id: str
    current_cost: float
    selected_cost: float
    expected_gain: float
    threshold: float
    switch_allowed: bool
    reason: DecisionReason
    explanation: str
    timestamp: float
    controller_cycle: int
    prediction_used: bool
    is_proactive: bool = False
    is_safety_override: bool = False
    safety_status: SafetyStatus = SafetyStatus.SAFE
    stability_status: Dict[str, Any] = field(default_factory=dict)
    migration_request: Optional[MigrationRequest] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "current_plan_id": self.current_plan_id,
            "selected_plan_id": self.selected_plan_id,
            "current_cost": self.current_cost,
            "selected_cost": self.selected_cost,
            "expected_gain": self.expected_gain,
            "threshold": self.threshold,
            "switch_allowed": self.switch_allowed,
            "reason": self.reason.value,
            "explanation": self.explanation,
            "timestamp": self.timestamp,
            "controller_cycle": self.controller_cycle,
            "prediction_used": self.prediction_used,
            "is_proactive": self.is_proactive,
            "is_safety_override": self.is_safety_override,
            "safety_status": self.safety_status.value,
            "stability_status": self.stability_status,
            "migration_request": self.migration_request.to_dict() if self.migration_request else None,
        }


@dataclass
class ControllerState:
    """
    Persistent runtime state for controller cycle tracking and anti-thrashing.
    """
    current_plan: PartitionPlan
    current_plan_id: str
    last_decision: Optional[ControlDecision] = None
    last_switch_timestamp: float = 0.0
    last_switch_cycle: int = 0
    switch_count: int = 0
    consecutive_preference_plan_id: Optional[str] = None
    consecutive_preference_count: int = 0
    cooldown_remaining_seconds: float = 0.0
    controller_cycle: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_plan_id": self.current_plan_id,
            "last_switch_timestamp": self.last_switch_timestamp,
            "last_switch_cycle": self.last_switch_cycle,
            "switch_count": self.switch_count,
            "consecutive_preference_plan_id": self.consecutive_preference_plan_id,
            "consecutive_preference_count": self.consecutive_preference_count,
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds,
            "controller_cycle": self.controller_cycle,
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
            "current_plan": self.current_plan.to_dict(),
        }


@dataclass(frozen=True)
class ControllerConfig:
    """
    Configurable parameters for controller stability, thresholding, and safety.
    """
    mode: ControllerMode = ControllerMode.PREDICTIVE
    switch_threshold: float = 0.25
    minimum_dwell_seconds: float = 5.0
    cooldown_seconds: float = 5.0
    hysteresis_cycles: int = 2
    allow_unknown_switches: bool = False
    emergency_override: bool = True
    fallback_plan_id: str = "local"
    max_memory_pressure: float = 0.90
    allow_reactive_fallback: bool = True

    def __post_init__(self) -> None:
        if self.switch_threshold < 0.0:
            raise ValueError(f"switch_threshold cannot be negative: {self.switch_threshold}")
        if self.minimum_dwell_seconds < 0.0:
            raise ValueError(f"minimum_dwell_seconds cannot be negative: {self.minimum_dwell_seconds}")
        if self.cooldown_seconds < 0.0:
            raise ValueError(f"cooldown_seconds cannot be negative: {self.cooldown_seconds}")
        if self.hysteresis_cycles < 1:
            raise ValueError(f"hysteresis_cycles must be >= 1: {self.hysteresis_cycles}")
        if not (0.0 <= self.max_memory_pressure <= 1.0):
            raise ValueError(f"max_memory_pressure must be in [0.0, 1.0]: {self.max_memory_pressure}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "switch_threshold": self.switch_threshold,
            "minimum_dwell_seconds": self.minimum_dwell_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "hysteresis_cycles": self.hysteresis_cycles,
            "allow_unknown_switches": self.allow_unknown_switches,
            "emergency_override": self.emergency_override,
            "fallback_plan_id": self.fallback_plan_id,
            "max_memory_pressure": self.max_memory_pressure,
            "allow_reactive_fallback": self.allow_reactive_fallback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ControllerConfig:
        return cls(
            mode=ControllerMode(data.get("mode", "predictive")),
            switch_threshold=float(data.get("switch_threshold", 0.25)),
            minimum_dwell_seconds=float(data.get("minimum_dwell_seconds", 5.0)),
            cooldown_seconds=float(data.get("cooldown_seconds", 5.0)),
            hysteresis_cycles=int(data.get("hysteresis_cycles", 2)),
            allow_unknown_switches=bool(data.get("allow_unknown_switches", False)),
            emergency_override=bool(data.get("emergency_override", True)),
            fallback_plan_id=str(data.get("fallback_plan_id", "local")),
            max_memory_pressure=float(data.get("max_memory_pressure", 0.90)),
            allow_reactive_fallback=bool(data.get("allow_reactive_fallback", True)),
        )
