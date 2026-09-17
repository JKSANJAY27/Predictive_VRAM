"""
Human-readable table formatting for candidate partition plan audits.
"""

from __future__ import annotations

from typing import Sequence

from src.partitioning.candidate import CandidatePlan
from src.runtime.tier import TierId


def format_candidate_table(candidates: Sequence[CandidatePlan]) -> str:
    """
    Format a sequence of CandidatePlan objects into an auditable text table.

    Example output:
        PLAN                  TIERS    BOUNDS  FEASIBLE_NOW  PREDICTED   PRIMARY REASON
        -----------------------------------------------------------------------------------------
        local                 U        0       FEASIBLE      FEASIBLE    All constraints ok
        u0-5_ea6-11           U-EA     1       FEASIBLE      FEASIBLE    All constraints ok
        u0-3_ea4-7_eb8-11     U-EA-EB  2       FEASIBLE      INFEASIBLE  Predicted VRAM exhaustion
    """
    if not candidates:
        return "No candidate plans in catalog."

    tier_abbr = {
        TierId.USER_DEVICE: "U",
        TierId.EDGE_A: "EA",
        TierId.EDGE_B: "EB",
    }

    headers = f"{'PLAN ID':<22} | {'TIERS':<8} | {'BOUNDS':<6} | {'NOW':<10} | {'PREDICTED':<10} | {'EXPLANATION'}"
    sep = "-" * len(headers) + "-" * 20

    lines = [headers, sep]

    for c in candidates:
        tiers_str = "-".join(tier_abbr.get(t, t.value) for t in c.active_tiers)
        now_str = c.feasibility_now.value.upper()
        pred_str = c.feasibility_predicted.value.upper()
        reason = c.feasibility_reasons[0] if c.feasibility_reasons else "N/A"

        lines.append(
            f"{c.plan_id:<22} | {tiers_str:<8} | {c.number_of_boundaries:<6} | {now_str:<10} | {pred_str:<10} | {reason}"
        )

    return "\n".join(lines)
