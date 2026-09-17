"""
Structural enumeration of contiguous layer partition plans.

Generates all valid combinations of sequential layer cut points across enabled execution tiers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set

from src.partitioning.plan_id import generate_plan_id
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId


def enumerate_structural_plans(
    total_layers: int,
    enabled_tiers: Optional[Sequence[TierId]] = None,
    allow_monolithic: bool = True,
    allow_two_tier: bool = True,
    allow_three_tier: bool = True,
) -> List[PartitionPlan]:
    """
    Exhaustively enumerate all structurally valid contiguous PartitionPlan configurations.

    Args:
        total_layers: Total number of transformer layers in model.
        enabled_tiers: Sequence of allowed TierIds (defaults to UserDevice, EdgeA, EdgeB).
        allow_monolithic: Whether to include single-tier local execution.
        allow_two_tier: Whether to include 2-tier splits.
        allow_three_tier: Whether to include 3-tier splits.

    Returns:
        A list of structurally validated PartitionPlan objects in deterministic order.
    """
    if total_layers <= 0:
        raise ValueError(f"total_layers must be positive, got {total_layers}")

    allowed = set(enabled_tiers) if enabled_tiers is not None else {
        TierId.USER_DEVICE,
        TierId.EDGE_A,
        TierId.EDGE_B,
    }

    plans: List[PartitionPlan] = []
    seen_ids: Set[str] = set()

    def _add_plan(p: PartitionPlan) -> None:
        pid = generate_plan_id(p)
        if pid not in seen_ids:
            seen_ids.add(pid)
            plans.append(p)

    # 1. Monolithic Local: UserDevice executes all layers [0, total_layers - 1]
    if allow_monolithic and TierId.USER_DEVICE in allowed:
        _add_plan(PartitionPlan.monolithic(total_layers))

    # 2. Two-tier splits
    if allow_two_tier and total_layers >= 2:
        # 2a. UserDevice -> EdgeA
        if TierId.USER_DEVICE in allowed and TierId.EDGE_A in allowed:
            for cut in range(total_layers - 1):
                _add_plan(PartitionPlan.two_tier(total_layers=total_layers, cut_layer=cut))

        # 2b. UserDevice -> EdgeB
        if TierId.USER_DEVICE in allowed and TierId.EDGE_B in allowed:
            for cut in range(total_layers - 1):
                p = PartitionPlan(
                    total_layers=total_layers,
                    user_device=(0, cut),
                    edge_a=None,
                    edge_b=(cut + 1, total_layers - 1),
                )
                _add_plan(p)

    # 3. Three-tier splits: UserDevice -> EdgeA -> EdgeB
    if allow_three_tier and total_layers >= 3:
        if (
            TierId.USER_DEVICE in allowed
            and TierId.EDGE_A in allowed
            and TierId.EDGE_B in allowed
        ):
            for c1 in range(total_layers - 2):
                for c2 in range(c1 + 1, total_layers - 1):
                    _add_plan(
                        PartitionPlan.three_tier(
                            total_layers=total_layers,
                            cut1=c1,
                            cut2=c2,
                        )
                    )

    return plans
