"""
Data types and representations for Module 8: Physical Migration and Runtime State Transition.

Defines migration status enums, execution modes, synchronization points,
event lifecycle, phase timings, metrics, and result contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.telemetry.types import DataSource


class MigrationStatus(str, Enum):
    """Lifecycle status of a migration operation."""
    PENDING = "pending"
    VALIDATING = "validating"
    PREPARING = "preparing"
    TRANSFERRING = "transferring"
    VERIFYING = "verifying"
    COMMITTED = "committed"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ALREADY_AT_TARGET = "already_at_target"
    FAILED_PARTIAL = "failed_partial"


class MigrationPoint(str, Enum):
    """Safe runtime synchronization point where migration can occur."""
    TOKEN_BOUNDARY = "token_boundary"
    MANUAL = "manual"


class MigrationMode(str, Enum):
    """Execution mode of the migration manager."""
    EXECUTE = "execute"
    SIMULATE = "simulate"


class VerificationMode(str, Enum):
    """Depth of post-migration correctness verification."""
    FAST = "fast"
    DEEP = "deep"


class MigrationEvent(str, Enum):
    """Lifecycle milestones emitted during migration transactions."""
    MIGRATION_STARTED = "migration_started"
    VALIDATION_COMPLETED = "validation_completed"
    PREPARATION_COMPLETED = "preparation_completed"
    MODEL_TRANSFER_STARTED = "model_transfer_started"
    MODEL_TRANSFER_COMPLETED = "model_transfer_completed"
    KV_TRANSFER_STARTED = "kv_transfer_started"
    KV_TRANSFER_COMPLETED = "kv_transfer_completed"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    COMMIT_STARTED = "commit_started"
    COMMIT_COMPLETED = "commit_completed"
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_COMPLETED = "rollback_completed"
    MIGRATION_FAILED = "migration_failed"
    MIGRATION_COMPLETED = "migration_completed"


@dataclass(frozen=True)
class MigrationPhaseTimings:
    """Detailed phase-level timing breakdown in milliseconds."""
    validation_ms: float = 0.0
    preparation_ms: float = 0.0
    model_transfer_ms: float = 0.0
    kv_transfer_ms: float = 0.0
    verification_ms: float = 0.0
    commit_ms: float = 0.0
    rollback_ms: float = 0.0
    total_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "validation_ms": round(self.validation_ms, 3),
            "preparation_ms": round(self.preparation_ms, 3),
            "model_transfer_ms": round(self.model_transfer_ms, 3),
            "kv_transfer_ms": round(self.kv_transfer_ms, 3),
            "verification_ms": round(self.verification_ms, 3),
            "commit_ms": round(self.commit_ms, 3),
            "rollback_ms": round(self.rollback_ms, 3),
            "total_duration_ms": round(self.total_duration_ms, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationPhaseTimings:
        return cls(
            validation_ms=float(data.get("validation_ms", 0.0)),
            preparation_ms=float(data.get("preparation_ms", 0.0)),
            model_transfer_ms=float(data.get("model_transfer_ms", 0.0)),
            kv_transfer_ms=float(data.get("kv_transfer_ms", 0.0)),
            verification_ms=float(data.get("verification_ms", 0.0)),
            commit_ms=float(data.get("commit_ms", 0.0)),
            rollback_ms=float(data.get("rollback_ms", 0.0)),
            total_duration_ms=float(data.get("total_duration_ms", 0.0)),
        )


@dataclass(frozen=True)
class MigrationMetrics:
    """Detailed physical byte counts and transfer statistics."""
    model_bytes_transferred: int = 0
    kv_cache_bytes_transferred: int = 0
    total_bytes_transferred: int = 0
    transfer_count: int = 0
    affected_layers: List[int] = field(default_factory=list)
    affected_tiers: List[str] = field(default_factory=list)
    per_layer_bytes: Dict[int, int] = field(default_factory=dict)
    per_tier_bytes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_bytes_transferred": self.model_bytes_transferred,
            "kv_cache_bytes_transferred": self.kv_cache_bytes_transferred,
            "total_bytes_transferred": self.total_bytes_transferred,
            "transfer_count": self.transfer_count,
            "affected_layers": self.affected_layers,
            "affected_tiers": self.affected_tiers,
            "per_layer_bytes": self.per_layer_bytes,
            "per_tier_bytes": self.per_tier_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationMetrics:
        return cls(
            model_bytes_transferred=int(data.get("model_bytes_transferred", 0)),
            kv_cache_bytes_transferred=int(data.get("kv_cache_bytes_transferred", 0)),
            total_bytes_transferred=int(data.get("total_bytes_transferred", 0)),
            transfer_count=int(data.get("transfer_count", 0)),
            affected_layers=data.get("affected_layers", []),
            affected_tiers=data.get("affected_tiers", []),
            per_layer_bytes={int(k): int(v) for k, v in data.get("per_layer_bytes", {}).items()},
            per_tier_bytes=data.get("per_tier_bytes", {}),
        )


@dataclass(frozen=True)
class MigrationComparison:
    """Auditable comparison between Module 6 predicted cost and Module 8 measured overhead."""
    predicted_switching_cost: float = 0.0
    actual_migration_duration_ms: float = 0.0
    predicted_kv_bytes: int = 0
    actual_kv_bytes: int = 0
    model_bytes_discrepancy: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_switching_cost": round(self.predicted_switching_cost, 4),
            "actual_migration_duration_ms": round(self.actual_migration_duration_ms, 3),
            "predicted_kv_bytes": self.predicted_kv_bytes,
            "actual_kv_bytes": self.actual_kv_bytes,
            "model_bytes_discrepancy": self.model_bytes_discrepancy,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationComparison:
        return cls(
            predicted_switching_cost=float(data.get("predicted_switching_cost", 0.0)),
            actual_migration_duration_ms=float(data.get("actual_migration_duration_ms", 0.0)),
            predicted_kv_bytes=int(data.get("predicted_kv_bytes", 0)),
            actual_kv_bytes=int(data.get("actual_kv_bytes", 0)),
            model_bytes_discrepancy=int(data.get("model_bytes_discrepancy", 0)),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class MigrationConfig:
    """Configuration parameters governing migration execution."""
    execution_mode: MigrationMode = MigrationMode.EXECUTE
    synchronization_point: MigrationPoint = MigrationPoint.TOKEN_BOUNDARY
    verification_mode: VerificationMode = VerificationMode.FAST
    rollback_enabled: bool = True
    single_flight: bool = True
    emulated_bandwidth_delay: bool = True
    fast_mode: bool = True  # If true, computes emulated network delay analytically without sleeping
    emulated_bandwidth_mbps: float = 50.0
    emulated_latency_ms: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_mode": self.execution_mode.value,
            "synchronization_point": self.synchronization_point.value,
            "verification_mode": self.verification_mode.value,
            "rollback_enabled": self.rollback_enabled,
            "single_flight": self.single_flight,
            "emulated_bandwidth_delay": self.emulated_bandwidth_delay,
            "fast_mode": self.fast_mode,
            "emulated_bandwidth_mbps": self.emulated_bandwidth_mbps,
            "emulated_latency_ms": self.emulated_latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationConfig:
        return cls(
            execution_mode=MigrationMode(data.get("execution_mode", "execute")),
            synchronization_point=MigrationPoint(data.get("synchronization_point", "token_boundary")),
            verification_mode=VerificationMode(data.get("verification_mode", "fast")),
            rollback_enabled=bool(data.get("rollback_enabled", True)),
            single_flight=bool(data.get("single_flight", True)),
            emulated_bandwidth_delay=bool(data.get("emulated_bandwidth_delay", True)),
            fast_mode=bool(data.get("fast_mode", True)),
            emulated_bandwidth_mbps=float(data.get("emulated_bandwidth_mbps", 50.0)),
            emulated_latency_ms=float(data.get("emulated_latency_ms", 10.0)),
        )


@dataclass(frozen=True)
class MigrationResult:
    """
    Comprehensive, immutable outcome of a migration transaction.
    """
    request_id: str
    source_plan: PartitionPlan
    target_plan: PartitionPlan
    source_plan_id: str
    target_plan_id: str
    status: MigrationStatus
    success: bool
    start_timestamp: float
    end_timestamp: float
    timings: MigrationPhaseTimings
    metrics: MigrationMetrics
    comparison: MigrationComparison
    events: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    rollback_performed: bool = False
    provenance: DataSource = DataSource.MEASURED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_plan_id": self.source_plan_id,
            "target_plan_id": self.target_plan_id,
            "status": self.status.value,
            "success": self.success,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "timings": self.timings.to_dict(),
            "metrics": self.metrics.to_dict(),
            "comparison": self.comparison.to_dict(),
            "events": self.events,
            "error_message": self.error_message,
            "rollback_performed": self.rollback_performed,
            "provenance": self.provenance.value,
            "source_plan": self.source_plan.to_dict(),
            "target_plan": self.target_plan.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationResult:
        src = PartitionPlan.from_dict(data["source_plan"]) if "source_plan" in data and isinstance(data["source_plan"], dict) else PartitionPlan.monolithic(12)
        tgt = PartitionPlan.from_dict(data["target_plan"]) if "target_plan" in data and isinstance(data["target_plan"], dict) else PartitionPlan.monolithic(12)
        timings = MigrationPhaseTimings.from_dict(data["timings"]) if data.get("timings") else MigrationPhaseTimings()
        metrics = MigrationMetrics.from_dict(data["metrics"]) if data.get("metrics") else MigrationMetrics()
        comparison = MigrationComparison.from_dict(data["comparison"]) if data.get("comparison") else MigrationComparison()
        prov = DataSource(data.get("provenance", DataSource.MEASURED.value))
        return cls(
            request_id=data["request_id"],
            source_plan=src,
            target_plan=tgt,
            source_plan_id=data.get("source_plan_id", ""),
            target_plan_id=data.get("target_plan_id", ""),
            status=MigrationStatus(data["status"]),
            success=bool(data["success"]),
            start_timestamp=float(data["start_timestamp"]),
            end_timestamp=float(data["end_timestamp"]),
            timings=timings,
            metrics=metrics,
            comparison=comparison,
            events=data.get("events", []),
            error_message=data.get("error_message"),
            rollback_performed=bool(data.get("rollback_performed", False)),
            provenance=prov,
        )
