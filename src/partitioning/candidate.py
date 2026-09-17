"""
CandidatePlan data representation for candidate split partitions.

Encapsulates the underlying PartitionPlan, active tier assignments, transfer boundaries,
and multi-horizon feasibility evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.partitioning.feasibility import FeasibilityStatus
from src.runtime.partition import PartitionPlan, TransferBoundary
from src.runtime.tier import TierId


@dataclass(frozen=True)
class CandidatePlan:
    """
    A proposed model partition candidate evaluated for feasibility.

    Attributes:
        plan_id: Stable, unique identifier (e.g. 'local', 'u0-5_ea6-11').
        partition_plan: The underlying valid PartitionPlan.
        active_tiers: Ordered list of tiers participating in execution.
        layer_assignment: Mapping from tier string to (start_layer, end_layer).
        number_of_boundaries: Number of inter-tier communication cuts.
        boundaries: TransferBoundary descriptors.
        feasibility_now: Feasibility evaluation under current conditions.
        feasibility_predicted: Feasibility evaluation across forecast horizon.
        feasibility_reasons: Human-readable explanations for status.
        estimated_memory_requirements: Mapping of tier string to required MB.
        resource_violations: Specific constraint failure descriptions.
        metadata: Diagnostic and structural tags.
    """
    plan_id: str
    partition_plan: PartitionPlan
    active_tiers: List[TierId]
    layer_assignment: Dict[str, Tuple[int, int]]
    number_of_boundaries: int
    boundaries: List[TransferBoundary]
    feasibility_now: FeasibilityStatus
    feasibility_predicted: FeasibilityStatus
    feasibility_reasons: List[str] = field(default_factory=list)
    estimated_memory_requirements: Dict[str, float] = field(default_factory=dict)
    resource_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_feasible(self) -> bool:
        """True strictly if both current and predicted conditions pass."""
        return (
            self.feasibility_now == FeasibilityStatus.FEASIBLE
            and self.feasibility_predicted == FeasibilityStatus.FEASIBLE
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "partition_plan": self.partition_plan.to_dict(),
            "active_tiers": [t.value for t in self.active_tiers],
            "layer_assignment": self.layer_assignment,
            "number_of_boundaries": self.number_of_boundaries,
            "boundaries": [b.to_dict() for b in self.boundaries],
            "feasibility_now": self.feasibility_now.value,
            "feasibility_predicted": self.feasibility_predicted.value,
            "is_feasible": self.is_feasible,
            "feasibility_reasons": self.feasibility_reasons,
            "estimated_memory_requirements": self.estimated_memory_requirements,
            "resource_violations": self.resource_violations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidatePlan:
        plan_dict = data["partition_plan"]
        p_plan = PartitionPlan(
            total_layers=plan_dict["total_layers"],
            user_device=tuple(plan_dict["user_device"]) if plan_dict.get("user_device") else None,
            edge_a=tuple(plan_dict["edge_a"]) if plan_dict.get("edge_a") else None,
            edge_b=tuple(plan_dict["edge_b"]) if plan_dict.get("edge_b") else None,
        )
        boundaries = [
            TransferBoundary(
                source_tier=TierId(b["source_tier"]),
                destination_tier=TierId(b["destination_tier"]),
                cut_layer=b["cut_layer"],
            )
            for b in data.get("boundaries", [])
        ]

        return cls(
            plan_id=data["plan_id"],
            partition_plan=p_plan,
            active_tiers=[TierId(t) for t in data["active_tiers"]],
            layer_assignment={k: tuple(v) for k, v in data["layer_assignment"].items()},
            number_of_boundaries=data["number_of_boundaries"],
            boundaries=boundaries,
            feasibility_now=FeasibilityStatus(data["feasibility_now"]),
            feasibility_predicted=FeasibilityStatus(data["feasibility_predicted"]),
            feasibility_reasons=data.get("feasibility_reasons", []),
            estimated_memory_requirements=data.get("estimated_memory_requirements", {}),
            resource_violations=data.get("resource_violations", []),
            metadata=data.get("metadata", {}),
        )
