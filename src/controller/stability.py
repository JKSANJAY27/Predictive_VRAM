"""
StabilityChecker: Anti-thrashing, thresholding, cooldown, dwell-time, and hysteresis enforcement.

Prevents rapid, costly partition flapping (A -> B -> A -> B) by enforcing:
    1. Expected Gain Threshold: Gain = J(current) - J(candidate) > theta
    2. Minimum Dwell Time: Must remain on a partition for at least tau_dwell seconds
    3. Cooldown Duration: Minimum interval between non-emergency switches
    4. Confirmation Hysteresis: Candidate must be preferred for N consecutive cycles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.controller.types import ControllerConfig, ControllerState, DecisionReason


@dataclass(frozen=True)
class StabilityResult:
    """Outcome of stability constraint evaluation."""
    switch_allowed: bool
    reason: Optional[DecisionReason]
    explanation: str
    expected_gain: float
    dwell_elapsed_seconds: float
    cooldown_remaining_seconds: float
    consecutive_count: int


class StabilityChecker:
    """
    Evaluates multi-layered stability guards before permitting a partition change.
    """

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()

    def evaluate(
        self,
        current_cost: float,
        candidate_cost: float,
        candidate_plan_id: str,
        ctrl_state: ControllerState,
        current_timestamp: float,
        is_emergency: bool = False,
    ) -> StabilityResult:
        """
        Evaluate whether proposed candidate satisfies all stability conditions.

        Args:
            current_cost: Total composite score of active plan J(P_t).
            candidate_cost: Total composite score of proposed plan J(a*).
            candidate_plan_id: Identifier of proposed plan.
            ctrl_state: Persistent controller state.
            current_timestamp: Current evaluation timestamp.
            is_emergency: True if safety policy mandates emergency recovery.

        Returns:
            StabilityResult with permission flag, reason code, and explanation.
        """
        gain = current_cost - candidate_cost
        theta = self.config.switch_threshold

        # Compute elapsed dwell time and cooldown remaining
        if ctrl_state.last_switch_timestamp > 0.0:
            dwell_elapsed = max(0.0, current_timestamp - ctrl_state.last_switch_timestamp)
        else:
            dwell_elapsed = 1e6  # Infinite if never switched yet

        cooldown_rem = max(0.0, self.config.cooldown_seconds - dwell_elapsed)

        # 1. Update Hysteresis tracking
        if ctrl_state.consecutive_preference_plan_id == candidate_plan_id:
            consecutive_count = ctrl_state.consecutive_preference_count + 1
        else:
            consecutive_count = 1

        # 2. EMERGENCY OVERRIDE: bypass threshold, dwell, cooldown, hysteresis if configured
        if is_emergency and self.config.emergency_override:
            return StabilityResult(
                switch_allowed=True,
                reason=DecisionReason.EMERGENCY_RECOVERY,
                explanation="Emergency override permitted: bypassing threshold and cooldown to resolve unsafe condition.",
                expected_gain=gain,
                dwell_elapsed_seconds=dwell_elapsed,
                cooldown_remaining_seconds=0.0,
                consecutive_count=consecutive_count,
            )

        # 3. Check for identical plan (no action needed)
        if candidate_plan_id == ctrl_state.current_plan_id:
            return StabilityResult(
                switch_allowed=False,
                reason=DecisionReason.BEST_CANDIDATE_IS_CURRENT,
                explanation="Current active plan remains optimal.",
                expected_gain=gain,
                dwell_elapsed_seconds=dwell_elapsed,
                cooldown_remaining_seconds=cooldown_rem,
                consecutive_count=consecutive_count,
            )

        # 4. Check for negative or zero gain
        if gain <= 0.0:
            return StabilityResult(
                switch_allowed=False,
                reason=DecisionReason.NO_IMPROVEMENT,
                explanation=f"Candidate '{candidate_plan_id}' does not improve cost (gain {gain:.3f} <= 0).",
                expected_gain=gain,
                dwell_elapsed_seconds=dwell_elapsed,
                cooldown_remaining_seconds=cooldown_rem,
                consecutive_count=consecutive_count,
            )

        # 5. Check Threshold condition: gain > theta
        if gain <= theta:
            return StabilityResult(
                switch_allowed=False,
                reason=DecisionReason.BELOW_THRESHOLD,
                explanation=f"Candidate gain {gain:.3f} is below required improvement threshold {theta:.2f}.",
                expected_gain=gain,
                dwell_elapsed_seconds=dwell_elapsed,
                cooldown_remaining_seconds=cooldown_rem,
                consecutive_count=consecutive_count,
            )

        # 6. Check Dwell Time / Cooldown condition
        if dwell_elapsed < self.config.minimum_dwell_seconds:
            return StabilityResult(
                switch_allowed=False,
                reason=DecisionReason.DWELL_TIME_ACTIVE,
                explanation=f"Minimum dwell time active: {dwell_elapsed:.1f}s elapsed < {self.config.minimum_dwell_seconds:.1f}s required.",
                expected_gain=gain,
                dwell_elapsed_seconds=dwell_elapsed,
                cooldown_remaining_seconds=cooldown_rem,
                consecutive_count=consecutive_count,
            )

        if cooldown_rem > 0.0:
            return StabilityResult(
                switch_allowed=False,
                reason=DecisionReason.COOLDOWN_ACTIVE,
                explanation=f"Switch cooldown active: {cooldown_rem:.1f}s remaining.",
                expected_gain=gain,
                dwell_elapsed_seconds=dwell_elapsed,
                cooldown_remaining_seconds=cooldown_rem,
                consecutive_count=consecutive_count,
            )

        # 7. Check Hysteresis confirmation count
        if consecutive_count < self.config.hysteresis_cycles:
            return StabilityResult(
                switch_allowed=False,
                reason=DecisionReason.HYSTERESIS_NOT_SATISFIED,
                explanation=f"Hysteresis confirmation pending: cycle {consecutive_count}/{self.config.hysteresis_cycles} for candidate '{candidate_plan_id}'.",
                expected_gain=gain,
                dwell_elapsed_seconds=dwell_elapsed,
                cooldown_remaining_seconds=cooldown_rem,
                consecutive_count=consecutive_count,
            )

        # 8. All stability conditions passed!
        return StabilityResult(
            switch_allowed=True,
            reason=None,  # Will be classified as PREDICTIVE_SWITCH_APPROVED or REACTIVE
            explanation=f"Candidate '{candidate_plan_id}' passed all stability checks (gain {gain:.3f} > {theta:.2f}).",
            expected_gain=gain,
            dwell_elapsed_seconds=dwell_elapsed,
            cooldown_remaining_seconds=0.0,
            consecutive_count=consecutive_count,
        )
