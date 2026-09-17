"""
Migration planning and affected layer delta analysis for Module 8.

Calculates the exact layer reassignments, unchanged layers, and affected tiers
between source and destination PartitionPlans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.partitioning.comparison import compare_plans
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId


@dataclass(frozen=True)
class MigrationPlan:
    """
    Concrete plan of work detailing physical layer and KV transitions.
    """
    source_plan: PartitionPlan
    target_plan: PartitionPlan
    affected_layers: List[int]
    unchanged_layers: List[int]
    layer_transitions: Dict[int, Tuple[TierId, TierId]]  # layer_idx -> (source_tier, target_tier)
    affected_tiers: List[TierId]
    is_noop: bool = False

    def get_source_tier(self, layer_idx: int) -> TierId:
        if layer_idx in self.layer_transitions:
            return self.layer_transitions[layer_idx][0]
        return self._find_tier(self.source_plan, layer_idx)

    def get_target_tier(self, layer_idx: int) -> TierId:
        if layer_idx in self.layer_transitions:
            return self.layer_transitions[layer_idx][1]
        return self._find_tier(self.target_plan, layer_idx)

    @staticmethod
    def _find_tier(plan: PartitionPlan, layer_idx: int) -> TierId:
        for tier_id in (TierId.USER_DEVICE, TierId.EDGE_A, TierId.EDGE_B):
            rng = plan.get_tier_range(tier_id)
            if rng is not None and rng[0] <= layer_idx <= rng[1]:
                return tier_id
        raise ValueError(f"Layer {layer_idx} not assigned to any tier in plan {plan}")


class MigrationPlanner:
    """
    Analyzes PartitionPlan differences and determines exact physical movements.
    """

    @classmethod
    def plan(
        cls,
        source_plan: PartitionPlan,
        target_plan: PartitionPlan,
    ) -> MigrationPlan:
        """
        Produce a MigrationPlan specifying affected layers and transitions.
        """
        if source_plan.total_layers != target_plan.total_layers:
            raise ValueError(
                f"Cannot plan migration across differing layer counts: "
                f"{source_plan.total_layers} vs {target_plan.total_layers}"
            )

        total_layers = source_plan.total_layers

        # Build layer-to-tier mappings for source and target
        def build_mapping(plan: PartitionPlan) -> Dict[int, TierId]:
            mapping: Dict[int, TierId] = {}
            for tier_id in (TierId.USER_DEVICE, TierId.EDGE_A, TierId.EDGE_B):
                rng = plan.get_tier_range(tier_id)
                if rng is not None:
                    for l in range(rng[0], rng[1] + 1):
                        mapping[l] = tier_id
            return mapping

        src_map = build_mapping(source_plan)
        tgt_map = build_mapping(target_plan)

        affected: List[int] = []
        unchanged: List[int] = []
        transitions: Dict[int, Tuple[TierId, TierId]] = {}
        affected_tiers_set = set()

        for l in range(total_layers):
            s_tier = src_map[l]
            t_tier = tgt_map[l]
            if s_tier != t_tier:
                affected.append(l)
                transitions[l] = (s_tier, t_tier)
                affected_tiers_set.add(s_tier)
                affected_tiers_set.add(t_tier)
            else:
                unchanged.append(l)

        # Sort tiers deterministically
        tier_order = [TierId.USER_DEVICE, TierId.EDGE_A, TierId.EDGE_B]
        affected_tiers = [t for t in tier_order if t in affected_tiers_set]

        return MigrationPlan(
            source_plan=source_plan,
            target_plan=target_plan,
            affected_layers=sorted(affected),
            unchanged_layers=sorted(unchanged),
            layer_transitions=transitions,
            affected_tiers=affected_tiers,
            is_noop=len(affected) == 0,
        )
