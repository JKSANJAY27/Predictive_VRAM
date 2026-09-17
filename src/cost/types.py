"""
Data structures and typing definitions for Module 6: Cost Model and Candidate Scoring.

Provides strongly typed configurations, cost component breakdowns, candidate scores,
and calibration models with strict provenance tracking and inspectable components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.partitioning.candidate import CandidatePlan
from src.telemetry.types import DataSource


class ScoreStatus(str, Enum):
    """Feasibility and validity state of a scored candidate plan."""
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"
    INVALID = "invalid"


@dataclass(frozen=True)
class CostWeights:
    """
    Configurable objective function weights and ablation toggles.

    Objective:
        J(a) = alpha * L(a) + beta * C(a) + gamma * M(a) + delta * E(a) + epsilon * P_switch(a)

    Attributes:
        alpha: Weight for normalized latency cost.
        beta: Weight for normalized steady-state communication volume cost.
        gamma: Weight for normalized memory pressure cost.
        delta: Weight for normalized energy proxy cost.
        epsilon: Weight for normalized switching/migration penalty.
        use_latency: Ablation toggle for latency component.
        use_communication: Ablation toggle for communication component.
        use_memory: Ablation toggle for memory pressure component.
        use_energy: Ablation toggle for energy component.
        use_switching: Ablation toggle for switching cost component.
    """
    alpha: float = 1.0
    beta: float = 0.2
    gamma: float = 1.0
    delta: float = 0.1
    epsilon: float = 1.0
    use_latency: bool = True
    use_communication: bool = True
    use_memory: bool = True
    use_energy: bool = True
    use_switching: bool = True

    def __post_init__(self) -> None:
        for name, val in [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("delta", self.delta),
            ("epsilon", self.epsilon),
        ]:
            if val < 0.0:
                raise ValueError(f"Cost weight '{name}' cannot be negative: {val}")

    @classmethod
    def latency_only(cls) -> CostWeights:
        return cls(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, epsilon=0.0)

    @classmethod
    def network_aware(cls) -> CostWeights:
        return cls(alpha=1.0, beta=0.5, gamma=0.0, delta=0.0, epsilon=0.0)

    @classmethod
    def memory_aware(cls) -> CostWeights:
        return cls(alpha=1.0, beta=0.0, gamma=1.0, delta=0.0, epsilon=0.0)

    @classmethod
    def no_switching(cls) -> CostWeights:
        return cls(alpha=1.0, beta=0.2, gamma=1.0, delta=0.1, epsilon=0.0, use_switching=False)

    @classmethod
    def equal_weights(cls) -> CostWeights:
        return cls(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, epsilon=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "use_latency": self.use_latency,
            "use_communication": self.use_communication,
            "use_memory": self.use_memory,
            "use_energy": self.use_energy,
            "use_switching": self.use_switching,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CostWeights:
        return cls(
            alpha=float(data.get("alpha", 1.0)),
            beta=float(data.get("beta", 0.2)),
            gamma=float(data.get("gamma", 1.0)),
            delta=float(data.get("delta", 0.1)),
            epsilon=float(data.get("epsilon", 1.0)),
            use_latency=bool(data.get("use_latency", True)),
            use_communication=bool(data.get("use_communication", True)),
            use_memory=bool(data.get("use_memory", True)),
            use_energy=bool(data.get("use_energy", True)),
            use_switching=bool(data.get("use_switching", True)),
        )


@dataclass(frozen=True)
class CostBreakdown:
    """
    Transparent, inspectable breakdown of raw, normalized, and weighted cost components.
    """
    # Raw un-normalized metrics
    latency_raw_ms: float
    communication_raw_mb: float
    memory_pressure_raw: float
    energy_raw_j: float
    switching_raw_units: float

    # Normalized metrics (unitless [0, inf), scale-divided)
    latency_normalized: float
    communication_normalized: float
    memory_pressure_normalized: float
    energy_normalized: float
    switching_normalized: float

    # Weighted terms: weight * normalized
    latency_cost: float
    communication_cost: float
    memory_pressure_cost: float
    energy_cost: float
    switching_cost: float

    # Total composite score
    total_cost: float

    # Sub-component details
    compute_latency_ms: float = 0.0
    comm_latency_ms: float = 0.0
    queue_latency_ms: float = 0.0
    steady_state_bytes: int = 0
    boundary_count: int = 0
    current_headroom_mb: Optional[float] = None
    predicted_min_headroom_mb: Optional[float] = None
    current_pressure: Optional[float] = None
    predicted_max_pressure: Optional[float] = None
    layer_migration_cost: float = 0.0
    boundary_change_cost: float = 0.0
    kv_transfer_cost: float = 0.0
    estimated_kv_transfer_bytes: int = 0

    # Data source provenance per component
    latency_provenance: DataSource = DataSource.ESTIMATED
    communication_provenance: DataSource = DataSource.ESTIMATED
    memory_provenance: DataSource = DataSource.ESTIMATED
    energy_provenance: DataSource = DataSource.ESTIMATED
    switching_provenance: DataSource = DataSource.ESTIMATED

    # Weights used
    weights: CostWeights = field(default_factory=CostWeights)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": {
                "latency_raw_ms": self.latency_raw_ms,
                "communication_raw_mb": self.communication_raw_mb,
                "memory_pressure_raw": self.memory_pressure_raw,
                "energy_raw_j": self.energy_raw_j,
                "switching_raw_units": self.switching_raw_units,
            },
            "normalized": {
                "latency_normalized": self.latency_normalized,
                "communication_normalized": self.communication_normalized,
                "memory_pressure_normalized": self.memory_pressure_normalized,
                "energy_normalized": self.energy_normalized,
                "switching_normalized": self.switching_normalized,
            },
            "weighted": {
                "latency_cost": self.latency_cost,
                "communication_cost": self.communication_cost,
                "memory_pressure_cost": self.memory_pressure_cost,
                "energy_cost": self.energy_cost,
                "switching_cost": self.switching_cost,
                "total_cost": self.total_cost,
            },
            "subcomponents": {
                "compute_latency_ms": self.compute_latency_ms,
                "comm_latency_ms": self.comm_latency_ms,
                "queue_latency_ms": self.queue_latency_ms,
                "steady_state_bytes": self.steady_state_bytes,
                "boundary_count": self.boundary_count,
                "current_headroom_mb": self.current_headroom_mb,
                "predicted_min_headroom_mb": self.predicted_min_headroom_mb,
                "current_pressure": self.current_pressure,
                "predicted_max_pressure": self.predicted_max_pressure,
                "layer_migration_cost": self.layer_migration_cost,
                "boundary_change_cost": self.boundary_change_cost,
                "kv_transfer_cost": self.kv_transfer_cost,
                "estimated_kv_transfer_bytes": self.estimated_kv_transfer_bytes,
            },
            "provenance": {
                "latency": self.latency_provenance.value,
                "communication": self.communication_provenance.value,
                "memory": self.memory_provenance.value,
                "energy": self.energy_provenance.value,
                "switching": self.switching_provenance.value,
            },
            "weights": self.weights.to_dict(),
        }


@dataclass(frozen=True)
class CandidateScore:
    """
    Evaluated cost score for a single CandidatePlan.

    Attributes:
        candidate_plan: The evaluated CandidatePlan.
        breakdown: Granular CostBreakdown object.
        total_cost: Composite objective score J(a).
        feasible: True if candidate satisfies physical/capacity constraints.
        status: Feasibility / validity status enum.
        confidence: Confidence factor in [0.0, 1.0] reflecting telemetry/forecast quality.
        explanation: Human-readable rationale for the score.
        metadata: Extra contextual annotations.
    """
    candidate_plan: CandidatePlan
    breakdown: CostBreakdown
    total_cost: float
    feasible: bool
    status: ScoreStatus
    confidence: float = 1.0
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.candidate_plan.plan_id,
            "total_cost": self.total_cost,
            "feasible": self.feasible,
            "status": self.status.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "breakdown": self.breakdown.to_dict(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CostModelCalibration:
    """
    Empirical calibration parameters for hardware compute, transfer latency, and power.
    """
    tier_compute_ms_per_layer: Dict[str, float] = field(default_factory=lambda: {
        "user_device": 8.0,
        "edge_a": 2.0,
        "edge_b": 1.0,
    })
    base_rtt_ms: float = 10.0
    power_proxy_watts: Dict[str, float] = field(default_factory=lambda: {
        "user_device": 5.0,
        "edge_a": 45.0,
        "edge_b": 150.0,
    })
    per_layer_switching_cost: float = 0.1
    boundary_shift_switching_cost: float = 0.2
    kv_transfer_cost_per_mb: float = 0.05
    tier_disruption_cost: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_compute_ms_per_layer": self.tier_compute_ms_per_layer,
            "base_rtt_ms": self.base_rtt_ms,
            "power_proxy_watts": self.power_proxy_watts,
            "per_layer_switching_cost": self.per_layer_switching_cost,
            "boundary_shift_switching_cost": self.boundary_shift_switching_cost,
            "kv_transfer_cost_per_mb": self.kv_transfer_cost_per_mb,
            "tier_disruption_cost": self.tier_disruption_cost,
        }
