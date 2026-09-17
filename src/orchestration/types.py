"""
Data types and schemas for Module 9: Closed-Loop Predictive Runtime Integration.

Defines the fundamental record units:
- OrchestrationMode and ExecutionMode enums
- ControlCycle: the atomic unit of closed-loop observation and decision
- MeasuredOutcome: post-migration outcome tracking
- PredictionLeadRecord: lead-time tracking for proactive adaptations
- MigrationWindowRecord: local ITL and resource window around a migration
- OrchestrationMetrics: aggregate end-to-end performance and overhead statistics
- TokenRecord: individual token generation event
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.controller.types import ControlDecision
from src.migration.types import MigrationResult
from src.prediction.types import PredictionResult
from src.state.types import RuntimeState


class OrchestrationMode(str, enum.Enum):
    """Operational mode for the closed-loop controller."""
    STATIC = "static"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"


class ExecutionMode(str, enum.Enum):
    """Operational mode for the runtime executor."""
    SIMULATION = "simulation"
    EXECUTION = "execution"
    REPLAY = "replay"


class ErrorCategory(str, enum.Enum):
    """Classified error types for control-plane and execution failures."""
    NONE = "none"
    TELEMETRY_ERROR = "telemetry_error"
    STATE_ERROR = "state_error"
    PREDICTION_ERROR = "prediction_error"
    CANDIDATE_ERROR = "candidate_error"
    COST_ERROR = "cost_error"
    CONTROLLER_ERROR = "controller_error"
    MIGRATION_ERROR = "migration_error"
    VERIFICATION_ERROR = "verification_error"


@dataclass
class MeasuredOutcome:
    """Post-migration or post-cycle physical outcome."""
    active_plan_after: Optional[str] = None
    measured_latency_after_ms: Optional[float] = None
    measured_memory_after_mb: Optional[float] = None
    measured_bandwidth_after_mbps: Optional[float] = None
    migration_pause_ms: Optional[float] = None
    itl_before_ms: Optional[float] = None
    itl_after_ms: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_plan_after": self.active_plan_after,
            "measured_latency_after_ms": self.measured_latency_after_ms,
            "measured_memory_after_mb": self.measured_memory_after_mb,
            "measured_bandwidth_after_mbps": self.measured_bandwidth_after_mbps,
            "migration_pause_ms": self.migration_pause_ms,
            "itl_before_ms": self.itl_before_ms,
            "itl_after_ms": self.itl_after_ms,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MeasuredOutcome:
        return cls(
            active_plan_after=data.get("active_plan_after"),
            measured_latency_after_ms=data.get("measured_latency_after_ms"),
            measured_memory_after_mb=data.get("measured_memory_after_mb"),
            measured_bandwidth_after_mbps=data.get("measured_bandwidth_after_mbps"),
            migration_pause_ms=data.get("migration_pause_ms"),
            itl_before_ms=data.get("itl_before_ms"),
            itl_after_ms=data.get("itl_after_ms"),
            notes=data.get("notes", ""),
        )


@dataclass
class PredictionLeadRecord:
    """Records timestamps and lead-time for proactive partition switching."""
    prediction_time: float
    predicted_risk_time: Optional[float] = None
    decision_time: float = 0.0
    migration_start: Optional[float] = None
    actual_degradation_time: Optional[float] = None
    prediction_lead_time_s: Optional[float] = None
    decision_lead_time_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_time": self.prediction_time,
            "predicted_risk_time": self.predicted_risk_time,
            "decision_time": self.decision_time,
            "migration_start": self.migration_start,
            "actual_degradation_time": self.actual_degradation_time,
            "prediction_lead_time_s": self.prediction_lead_time_s,
            "decision_lead_time_s": self.decision_lead_time_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PredictionLeadRecord:
        return cls(
            prediction_time=data["prediction_time"],
            predicted_risk_time=data.get("predicted_risk_time"),
            decision_time=data.get("decision_time", 0.0),
            migration_start=data.get("migration_start"),
            actual_degradation_time=data.get("actual_degradation_time"),
            prediction_lead_time_s=data.get("prediction_lead_time_s"),
            decision_lead_time_s=data.get("decision_lead_time_s"),
        )


@dataclass
class MigrationWindowRecord:
    """Local before/after context window surrounding a physical migration."""
    token_index: int
    tokens_before: List[int] = field(default_factory=list)
    itl_before_ms: List[float] = field(default_factory=list)
    migration_duration_ms: float = 0.0
    tokens_after: List[int] = field(default_factory=list)
    itl_after_ms: List[float] = field(default_factory=list)
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    network_before_mbps: Optional[float] = None
    network_after_mbps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_index": self.token_index,
            "tokens_before": self.tokens_before,
            "itl_before_ms": self.itl_before_ms,
            "migration_duration_ms": self.migration_duration_ms,
            "tokens_after": self.tokens_after,
            "itl_after_ms": self.itl_after_ms,
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "network_before_mbps": self.network_before_mbps,
            "network_after_mbps": self.network_after_mbps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationWindowRecord:
        return cls(
            token_index=data["token_index"],
            tokens_before=data.get("tokens_before", []),
            itl_before_ms=data.get("itl_before_ms", []),
            migration_duration_ms=data.get("migration_duration_ms", 0.0),
            tokens_after=data.get("tokens_after", []),
            itl_after_ms=data.get("itl_after_ms", []),
            memory_before_mb=data.get("memory_before_mb"),
            memory_after_mb=data.get("memory_after_mb"),
            network_before_mbps=data.get("network_before_mbps"),
            network_after_mbps=data.get("network_after_mbps"),
        )


@dataclass
class TokenRecord:
    """Record of an individual generated token."""
    token_index: int
    token_id: int
    token_str: str
    step_latency_ms: float
    timestamp: float
    active_plan_id: str
    is_ttft: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_index": self.token_index,
            "token_id": self.token_id,
            "token_str": self.token_str,
            "step_latency_ms": self.step_latency_ms,
            "timestamp": self.timestamp,
            "active_plan_id": self.active_plan_id,
            "is_ttft": self.is_ttft,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenRecord:
        return cls(
            token_index=data["token_index"],
            token_id=data["token_id"],
            token_str=data.get("token_str", ""),
            step_latency_ms=data["step_latency_ms"],
            timestamp=data["timestamp"],
            active_plan_id=data["active_plan_id"],
            is_ttft=data.get("is_ttft", False),
        )


@dataclass
class ControlCycle:
    """
    Fundamental unit of closed-loop analysis.

    Captures the complete state, prediction, evaluation, decision,
    action, and measured outcome for one discrete control cycle.
    """
    cycle_id: int
    cycle_timestamp: float
    inference_step: int
    observed_state: Optional[RuntimeState] = None
    prediction_result: Optional[PredictionResult] = None
    candidate_count: int = 0
    candidate_scores: List[Dict[str, Any]] = field(default_factory=list)
    control_decision: Optional[ControlDecision] = None
    migration_result: Optional[MigrationResult] = None
    measured_outcome: Optional[MeasuredOutcome] = None
    cycle_duration_ms: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    error_category: ErrorCategory = ErrorCategory.NONE
    error_message: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "cycle_timestamp": self.cycle_timestamp,
            "inference_step": self.inference_step,
            "observed_state": self.observed_state.to_dict() if self.observed_state else None,
            "prediction_result": self.prediction_result.to_dict() if self.prediction_result else None,
            "candidate_count": self.candidate_count,
            "candidate_scores": self.candidate_scores,
            "control_decision": self.control_decision.to_dict() if self.control_decision else None,
            "migration_result": self.migration_result.to_dict() if self.migration_result else None,
            "measured_outcome": self.measured_outcome.to_dict() if self.measured_outcome else None,
            "cycle_duration_ms": self.cycle_duration_ms,
            "timings": self.timings,
            "error_category": self.error_category.value,
            "error_message": self.error_message,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ControlCycle:
        obs_state = RuntimeState.from_dict(data["observed_state"]) if data.get("observed_state") else None
        pred_res = PredictionResult.from_dict(data["prediction_result"]) if data.get("prediction_result") else None
        ctrl_dec = ControlDecision.from_dict(data["control_decision"]) if data.get("control_decision") else None
        mig_res = MigrationResult.from_dict(data["migration_result"]) if data.get("migration_result") else None
        meas_out = MeasuredOutcome.from_dict(data["measured_outcome"]) if data.get("measured_outcome") else None
        err_cat = ErrorCategory(data.get("error_category", ErrorCategory.NONE.value))

        return cls(
            cycle_id=data["cycle_id"],
            cycle_timestamp=data["cycle_timestamp"],
            inference_step=data["inference_step"],
            observed_state=obs_state,
            prediction_result=pred_res,
            candidate_count=data.get("candidate_count", 0),
            candidate_scores=data.get("candidate_scores", []),
            control_decision=ctrl_dec,
            migration_result=mig_res,
            measured_outcome=meas_out,
            cycle_duration_ms=data.get("cycle_duration_ms", 0.0),
            timings=data.get("timings", {}),
            error_category=err_cat,
            error_message=data.get("error_message"),
            notes=data.get("notes", ""),
        )


@dataclass
class OrchestrationMetrics:
    """Comprehensive performance, stability, and overhead metrics for an experiment."""
    # Inference metrics
    completed_tokens: int = 0
    ttft_ms: float = 0.0
    mean_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    total_inference_latency_ms: float = 0.0
    end_to_end_runtime_ms: float = 0.0

    # Memory & Network metrics
    min_memory_headroom_mb: Optional[float] = None
    max_memory_pressure: Optional[float] = None
    max_kv_cache_bytes: int = 0
    mean_bandwidth_mbps: Optional[float] = None
    min_bandwidth_mbps: Optional[float] = None

    # Control metrics
    total_control_cycles: int = 0
    total_switches: int = 0
    proactive_switches: int = 0
    reactive_switches: int = 0
    safety_overrides: int = 0
    switches_per_hour: float = 0.0
    average_dwell_time_tokens: float = 0.0

    # Migration metrics
    successful_migrations: int = 0
    rolled_back_migrations: int = 0
    failed_migrations: int = 0
    total_migration_time_ms: float = 0.0
    total_migration_bytes: int = 0
    total_model_bytes: int = 0
    total_kv_bytes: int = 0

    # Control plane overhead breakdown (ms)
    telemetry_overhead_ms: float = 0.0
    prediction_overhead_ms: float = 0.0
    candidate_generation_overhead_ms: float = 0.0
    cost_model_overhead_ms: float = 0.0
    controller_overhead_ms: float = 0.0
    migration_execution_overhead_ms: float = 0.0
    total_control_plane_overhead_ms: float = 0.0
    control_overhead_fraction: float = 0.0

    # Violations / errors
    slo_violations: int = 0
    component_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrchestrationMetrics:
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
