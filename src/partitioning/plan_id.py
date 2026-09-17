"""
Deterministic, stable identifier generator for PartitionPlan configurations.
"""

from __future__ import annotations

from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId


def generate_plan_id(plan: PartitionPlan) -> str:
    """
    Generate a deterministic, canonical string identifier for a PartitionPlan.

    Examples:
        - Monolithic local (all layers on UserDevice): 'local'
        - Two-tier (UserDevice -> EdgeA): 'u0-5_ea6-11'
        - Two-tier (UserDevice -> EdgeB): 'u0-5_eb6-11'
        - Three-tier: 'u0-3_ea4-7_eb8-11'
    """
    active = plan.get_active_tiers()
    if len(active) == 1 and active[0][0] == TierId.USER_DEVICE:
        return "local"

    tokens: list[str] = []
    prefix_map = {
        TierId.USER_DEVICE: "u",
        TierId.EDGE_A: "ea",
        TierId.EDGE_B: "eb",
    }

    for tier_id, (start, end) in active:
        prefix = prefix_map[tier_id]
        tokens.append(f"{prefix}{start}-{end}")

    return "_".join(tokens)
