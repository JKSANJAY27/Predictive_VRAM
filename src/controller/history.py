"""
ControllerHistory: Chronological decision log and runtime stability metrics tracker.

Exposes research metrics including:
    - Switch counts and switch frequency
    - Proactive vs reactive switch counts
    - Threshold, cooldown, and hysteresis rejection distributions
    - Average dwell times and plan occupancy ratios
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.controller.types import ControlAction, ControlDecision, DecisionReason


class ControllerHistory:
    """
    Maintains historical decision records and computes stability & optimization statistics.
    """

    def __init__(self) -> None:
        self.decisions: List[ControlDecision] = []

    def record(self, decision: ControlDecision) -> None:
        """Append a decision record."""
        self.decisions.append(decision)

    def clear(self) -> None:
        """Reset historical records."""
        self.decisions.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Compute aggregate controller performance and stability metrics."""
        total_cycles = len(self.decisions)
        if total_cycles == 0:
            return {
                "total_cycles": 0,
                "total_switches": 0,
                "proactive_switches": 0,
                "safety_overrides": 0,
                "fallback_events": 0,
                "threshold_rejections": 0,
                "cooldown_rejections": 0,
                "hysteresis_rejections": 0,
                "average_dwell_seconds": 0.0,
                "switch_frequency_per_hour": 0.0,
                "reason_distribution": {},
                "plan_occupancy": {},
            }

        switches = [d for d in self.decisions if d.action == ControlAction.SWITCH]
        total_switches = len(switches)
        proactive_switches = sum(1 for d in switches if d.is_proactive)
        safety_overrides = sum(1 for d in switches if d.is_safety_override)
        fallback_events = sum(1 for d in self.decisions if d.action == ControlAction.SAFE_FALLBACK)

        threshold_rejections = sum(1 for d in self.decisions if d.reason == DecisionReason.BELOW_THRESHOLD)
        cooldown_rejections = sum(
            1 for d in self.decisions if d.reason in (DecisionReason.COOLDOWN_ACTIVE, DecisionReason.DWELL_TIME_ACTIVE)
        )
        hysteresis_rejections = sum(
            1 for d in self.decisions if d.reason == DecisionReason.HYSTERESIS_NOT_SATISFIED
        )

        # Average dwell time between switches
        if len(switches) >= 2:
            intervals = [switches[i].timestamp - switches[i - 1].timestamp for i in range(1, len(switches))]
            avg_dwell = float(sum(intervals) / len(intervals))
        elif total_switches == 1:
            avg_dwell = float(switches[0].timestamp - self.decisions[0].timestamp)
        else:
            avg_dwell = float(self.decisions[-1].timestamp - self.decisions[0].timestamp)

        # Switches per simulated hour
        total_duration = max(1.0, self.decisions[-1].timestamp - self.decisions[0].timestamp)
        switches_per_hour = (total_switches / total_duration) * 3600.0

        # Reason distribution
        reasons: Dict[str, int] = {}
        for d in self.decisions:
            reasons[d.reason.value] = reasons.get(d.reason.value, 0) + 1

        # Plan occupancy (% of cycles on each plan)
        occupancy_counts: Dict[str, int] = {}
        for d in self.decisions:
            occupancy_counts[d.current_plan_id] = occupancy_counts.get(d.current_plan_id, 0) + 1
        occupancy = {k: float(v / total_cycles) for k, v in occupancy_counts.items()}

        return {
            "total_cycles": total_cycles,
            "total_switches": total_switches,
            "proactive_switches": proactive_switches,
            "safety_overrides": safety_overrides,
            "fallback_events": fallback_events,
            "threshold_rejections": threshold_rejections,
            "cooldown_rejections": cooldown_rejections,
            "hysteresis_rejections": hysteresis_rejections,
            "average_dwell_seconds": round(avg_dwell, 2),
            "switch_frequency_per_hour": round(switches_per_hour, 2),
            "reason_distribution": reasons,
            "plan_occupancy": occupancy,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.get_metrics(),
            "decisions": [d.to_dict() for d in self.decisions],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)
