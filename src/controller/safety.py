"""
SafetyPolicy: Evaluates runtime safety, memory exhaustion risks, and emergency recovery.

Prioritizes:
    1. Correctness & Physical Safety
    2. Feasibility
    3. Stability
    4. Optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.controller.types import ControllerConfig, SafetyStatus
from src.cost.types import CandidateScore, ScoreStatus
from src.partitioning.feasibility import FeasibilityStatus
from src.prediction.types import PredictionResult
from src.state.types import RuntimeState


@dataclass(frozen=True)
class SafetyEvaluation:
    """Outcome of safety policy inspection."""
    status: SafetyStatus
    is_emergency: bool
    requires_fallback: bool
    reason: str


class SafetyPolicy:
    """
    Evaluates physical resource constraints and forecast warnings to protect
    the runtime against memory exhaustion or severe channel dropouts.
    """

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()

    def evaluate_current_plan(
        self,
        current_score: CandidateScore,
        state: Optional[RuntimeState] = None,
        forecast: Optional[PredictionResult] = None,
    ) -> SafetyEvaluation:
        """
        Audit the safety of the current partition plan under current and future conditions.

        Args:
            current_score: Evaluated score of the active plan.
            state: Optional current RuntimeState.
            forecast: Optional multi-step resource forecast.

        Returns:
            SafetyEvaluation with status, emergency flag, and explanatory reason.
        """
        c = current_score.candidate_plan
        b = current_score.breakdown

        # 1. Check for outright infeasibility (current or predicted)
        if (
            c.feasibility_now == FeasibilityStatus.INFEASIBLE
            or c.feasibility_predicted == FeasibilityStatus.INFEASIBLE
            or current_score.status == ScoreStatus.INFEASIBLE
        ):
            return SafetyEvaluation(
                status=SafetyStatus.EMERGENCY,
                is_emergency=True,
                requires_fallback=True,
                reason="Current partition is physically infeasible under current or predicted conditions.",
            )

        # 2. Check for dangerous memory pressure escalation
        max_pressure = b.predicted_max_pressure or b.current_pressure or b.memory_pressure_raw
        if max_pressure is not None and max_pressure >= self.config.max_memory_pressure:
            return SafetyEvaluation(
                status=SafetyStatus.EMERGENCY,
                is_emergency=True,
                requires_fallback=False,  # Can switch to a lower-memory candidate
                reason=f"Current partition predicted memory pressure {max_pressure:.2f} >= safety limit {self.config.max_memory_pressure:.2f}.",
            )

        # 3. Check for forecast validity
        if forecast is not None and not forecast.is_valid:
            return SafetyEvaluation(
                status=SafetyStatus.WARNING,
                is_emergency=False,
                requires_fallback=False,
                reason=f"Resource forecast is marked invalid: {forecast.status_message}.",
            )

        # 4. Elevated pressure warning
        if max_pressure is not None and max_pressure >= 0.75:
            return SafetyEvaluation(
                status=SafetyStatus.WARNING,
                is_emergency=False,
                requires_fallback=False,
                reason=f"Memory pressure elevated: {max_pressure:.2f}.",
            )

        # 5. Normal safe operation
        return SafetyEvaluation(
            status=SafetyStatus.SAFE,
            is_emergency=False,
            requires_fallback=False,
            reason="Current partition is operating within safe physical margins.",
        )
