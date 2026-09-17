"""
Structural plan comparison and difference analysis utility.

Quantifies the structural distance between two PartitionPlans, computing:
- Number of layers reassigned
- Affected execution tiers
- Boundary shift displacement
- Added and removed communication boundaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.runtime.partition import PartitionPlan, TransferBoundary
from src.runtime.tier import TierId


@dataclass(frozen=True)
class PlanDifference:
    """
    Structural delta between a current plan and a proposed candidate plan.
    """
    is_identical: bool
    changed_layer_count: int
    affected_tiers: List[TierId]
    boundary_shift_distance: int
    added_boundaries: List[TransferBoundary]
    removed_boundaries: List[TransferBoundary]
    boundary_count_delta: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_identical": self.is_identical,
            "changed_layer_count": self.changed_layer_count,
            "affected_tiers": [t.value for t in self.affected_tiers],
            "boundary_shift_distance": self.boundary_shift_distance,
            "added_boundaries": [b.to_dict() for b in self.added_boundaries],
            "removed_boundaries": [b.to_dict() for b in self.removed_boundaries],
            "boundary_count_delta": self.boundary_count_delta,
        }


def compare_plans(current_plan: PartitionPlan, candidate_plan: PartitionPlan) -> PlanDifference:
    """
    Compute structural difference between two PartitionPlans.

    Args:
        current_plan: Currently active baseline plan.
        candidate_plan: Proposed target candidate plan.
    """
    if current_plan.total_layers != candidate_plan.total_layers:
        raise ValueError(
            f"Cannot compare plans with different total_layers: "
            f"{current_plan.total_layers} vs {candidate_plan.total_layers}"
        )

    total_layers = current_plan.total_layers

    # 1. Count re-assigned layers
    changed_layers = 0
    for l_idx in range(total_layers):
        if current_plan.get_tier_for_layer(l_idx) != candidate_plan.get_tier_for_layer(l_idx):
            changed_layers += 1

    # 2. Identify affected tiers
    affected: List[TierId] = []
    for tier_id in (TierId.USER_DEVICE, TierId.EDGE_A, TierId.EDGE_B):
        if current_plan.get_tier_range(tier_id) != candidate_plan.get_tier_range(tier_id):
            affected.append(tier_id)

    # 3. Analyze transfer boundaries
    curr_b = current_plan.get_transfer_boundaries()
    cand_b = candidate_plan.get_transfer_boundaries()

    curr_b_set = set(curr_b)
    cand_b_set = set(cand_b)

    added = [b for b in cand_b if b not in curr_b_set]
    removed = [b for b in curr_b if b not in cand_b_set]

    # Calculate shift distance between corresponding boundaries
    shift_dist = 0
    min_len = min(len(curr_b), len(cand_b))
    for i in range(min_len):
        shift_dist += abs(cand_b[i].cut_layer - curr_b[i].cut_layer)
    if len(cand_b) > len(curr_b):
        for i in range(min_len, len(cand_b)):
            shift_dist += cand_b[i].cut_layer
    elif len(curr_b) > len(cand_b):
        for i in range(min_len, len(curr_b)):
            shift_dist += curr_b[i].cut_layer

    is_ident = (changed_layers == 0 and len(affected) == 0 and len(added) == 0 and len(removed) == 0)

    return PlanDifference(
        is_identical=is_ident,
        changed_layer_count=changed_layers,
        affected_tiers=affected,
        boundary_shift_distance=shift_dist,
        added_boundaries=added,
        removed_boundaries=removed,
        boundary_count_delta=len(cand_b) - len(curr_b),
    )
