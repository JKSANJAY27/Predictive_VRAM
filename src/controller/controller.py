"""
AdaptivePartitionController: Central decision policy engine for predictive dynamic split inference.

Observes runtime state, forecasts, and candidate partition scores to decide:
    - KEEP_CURRENT
    - SWITCH (produces MigrationRequest for Module 8)
    - SAFE_FALLBACK
    - NO_ACTION

Strict Research Invariant:
    Does NOT physically move model weights, modify module placement, or transfer
    tensors (strictly deferred to Module 8 MigrationManager).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.controller.history import ControllerHistory
from src.controller.safety import SafetyEvaluation, SafetyPolicy
from src.controller.stability import StabilityChecker, StabilityResult
from src.controller.types import (
    ControlAction,
    ControlDecision,
    ControllerConfig,
    ControllerMode,
    ControllerState,
    DecisionReason,
    MigrationRequest,
    SafetyStatus,
)
from src.cost.model import CostModel
from src.cost.types import CandidateScore, ScoreStatus
from src.partitioning.candidate import CandidatePlan
from src.partitioning.comparison import compare_plans
from src.partitioning.feasibility import FeasibilityStatus
from src.partitioning.plan_id import generate_plan_id
from src.prediction.types import PredictionResult
from src.runtime.partition import PartitionPlan
from src.state.types import RuntimeState


class AdaptivePartitionController:
    """
    Adaptive predictive partition controller orchestrating multi-horizon optimization,
    anti-thrashing stability, and emergency safety policies.
    """

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        initial_plan: Optional[PartitionPlan] = None,
        cost_model: Optional[CostModel] = None,
    ) -> None:
        self.config = config or ControllerConfig()
        self.cost_model = cost_model or CostModel()

        # Initial active partition (default: 12-layer monolithic local)
        curr_p = initial_plan or PartitionPlan.monolithic(total_layers=12)
        curr_id = generate_plan_id(curr_p)

        self.state = ControllerState(
            current_plan=curr_p,
            current_plan_id=curr_id,
        )

        # Sub-modules
        self.safety_policy = SafetyPolicy(self.config)
        self.stability_checker = StabilityChecker(self.config)
        self.history = ControllerHistory()

    def decide(
        self,
        state: Optional[RuntimeState] = None,
        candidates: Optional[Sequence[CandidatePlan]] = None,
        forecast: Optional[PredictionResult] = None,
        current_plan: Optional[PartitionPlan] = None,
    ) -> ControlDecision:
        """
        Execute a single controller evaluation cycle and decide active partition action.

        Args:
            state: Current RuntimeState observation.
            candidates: Feasible/structural candidate plans from Module 5.
            forecast: Multi-step prediction result from Module 4.
            current_plan: Optional explicit active plan (overrides internal state if provided).

        Returns:
            ControlDecision with action, rationale, and optional MigrationRequest.
        """
        self.state.controller_cycle += 1
        cycle = self.state.controller_cycle
        ts = state.timestamp if state else float(cycle)

        # Synchronize active plan if passed explicitly
        if current_plan is not None:
            self.state.current_plan = current_plan
            self.state.current_plan_id = generate_plan_id(current_plan)

        curr_p = self.state.current_plan
        curr_id = self.state.current_plan_id

        # ---------------------------------------------------------------------
        # 1. Handle STATIC Controller Mode
        # ---------------------------------------------------------------------
        if self.config.mode == ControllerMode.STATIC:
            decision = ControlDecision(
                action=ControlAction.KEEP_CURRENT,
                current_plan_id=curr_id,
                selected_plan_id=curr_id,
                current_cost=1.0,
                selected_cost=1.0,
                expected_gain=0.0,
                threshold=self.config.switch_threshold,
                switch_allowed=False,
                reason=DecisionReason.STATIC_POLICY,
                explanation="Static controller mode active: partition adaptation disabled.",
                timestamp=ts,
                controller_cycle=cycle,
                prediction_used=False,
            )
            self._finalize_decision(decision)
            return decision

        # ---------------------------------------------------------------------
        # 2. Determine Effective Forecast (Predictive vs Reactive Mode)
        # ---------------------------------------------------------------------
        if self.config.mode == ControllerMode.REACTIVE:
            effective_forecast: Optional[PredictionResult] = None
            prediction_used = False
        else:
            effective_forecast = forecast
            prediction_used = forecast is not None and forecast.is_valid

        # ---------------------------------------------------------------------
        # 3. Score Candidates and Evaluate Current Plan Cost
        # ---------------------------------------------------------------------
        cand_list = list(candidates) if candidates else []

        # Ensure current plan is scored
        scored_candidates = self.cost_model.score_candidates(
            candidates=cand_list,
            state=state,
            forecast=effective_forecast,
            current_plan=curr_p,
        )

        curr_score: Optional[CandidateScore] = None
        for s in scored_candidates:
            if s.candidate_plan.plan_id == curr_id:
                curr_score = s
                break

        if curr_score is None:
            # Score current plan explicitly as CandidatePlan wrapper
            curr_cand = CandidatePlan(
                plan_id=curr_id,
                partition_plan=curr_p,
                active_tiers=[t for t, _ in curr_p.get_active_tiers()],
                layer_assignment={t.value: r for t, r in curr_p.get_active_tiers()},
                number_of_boundaries=len(curr_p.get_transfer_boundaries()),
                boundaries=curr_p.get_transfer_boundaries(),
                feasibility_now=FeasibilityStatus.FEASIBLE,
                feasibility_predicted=FeasibilityStatus.FEASIBLE,
            )
            curr_score = self.cost_model.score(
                candidate=curr_cand,
                state=state,
                forecast=effective_forecast,
                current_plan=curr_p,
            )

        current_cost = curr_score.total_cost

        # ---------------------------------------------------------------------
        # 4. Audit Safety of Current Plan
        # ---------------------------------------------------------------------
        safety_eval: SafetyEvaluation = self.safety_policy.evaluate_current_plan(
            current_score=curr_score,
            state=state,
            forecast=effective_forecast,
        )

        # ---------------------------------------------------------------------
        # 5. Filter Feasible Candidates and Find a* = argmin J(a)
        # ---------------------------------------------------------------------
        eligible_scores: List[CandidateScore] = []
        for s in scored_candidates:
            # Rule 1: Never switch to an explicitly infeasible candidate
            if s.status == ScoreStatus.INFEASIBLE or not s.feasible:
                continue

            # Rule 2: Exclude UNKNOWN candidates unless explicitly configured
            if s.status == ScoreStatus.UNKNOWN and not self.config.allow_unknown_switches:
                # Exception: current plan itself is allowed
                if s.candidate_plan.plan_id != curr_id:
                    continue

            eligible_scores.append(s)

        # If no eligible candidate exists
        if not eligible_scores:
            action = ControlAction.SAFE_FALLBACK if safety_eval.is_emergency else ControlAction.KEEP_CURRENT
            decision = ControlDecision(
                action=action,
                current_plan_id=curr_id,
                selected_plan_id=self.config.fallback_plan_id,
                current_cost=current_cost,
                selected_cost=current_cost,
                expected_gain=0.0,
                threshold=self.config.switch_threshold,
                switch_allowed=False,
                reason=DecisionReason.NO_FEASIBLE_CANDIDATE,
                explanation="No eligible feasible candidate plan found in catalog.",
                timestamp=ts,
                controller_cycle=cycle,
                prediction_used=prediction_used,
                safety_status=safety_eval.status,
            )
            self._finalize_decision(decision)
            return decision

        # Identify lowest-cost candidate: a* = argmin J(a)
        best_score = min(eligible_scores, key=lambda s: s.total_cost)
        best_plan_id = best_score.candidate_plan.plan_id
        best_cost = best_score.total_cost

        # ---------------------------------------------------------------------
        # 6. Evaluate Stability Constraints (Gain, Threshold, Dwell, Hysteresis)
        # ---------------------------------------------------------------------
        stab_result: StabilityResult = self.stability_checker.evaluate(
            current_cost=current_cost,
            candidate_cost=best_cost,
            candidate_plan_id=best_plan_id,
            ctrl_state=self.state,
            current_timestamp=ts,
            is_emergency=safety_eval.is_emergency,
        )

        # Update hysteresis state
        self.state.consecutive_preference_plan_id = best_plan_id
        self.state.consecutive_preference_count = stab_result.consecutive_count

        # ---------------------------------------------------------------------
        # 7. Formulate Control Decision
        # ---------------------------------------------------------------------
        if stab_result.switch_allowed:
            # Classify decision as proactive vs reactive
            is_proactive = False
            if (
                prediction_used
                and not safety_eval.is_emergency
                and curr_score.candidate_plan.feasibility_now == FeasibilityStatus.FEASIBLE
            ):
                # Proactive condition: switched because forecast indicated future cost/resource advantage
                is_proactive = True

            if safety_eval.is_emergency:
                reason = DecisionReason.EMERGENCY_RECOVERY
            elif is_proactive:
                reason = DecisionReason.PREDICTIVE_SWITCH_APPROVED
            else:
                reason = DecisionReason.REACTIVE_SWITCH_APPROVED

            # Create MigrationRequest for Module 8
            diff = compare_plans(curr_p, best_score.candidate_plan.partition_plan)
            migration_req = MigrationRequest(
                source_plan=curr_p,
                target_plan=best_score.candidate_plan.partition_plan,
                source_plan_id=curr_id,
                target_plan_id=best_plan_id,
                expected_gain=stab_result.expected_gain,
                switching_cost=best_score.breakdown.switching_cost,
                controller_cycle=cycle,
                timestamp=ts,
                changed_layers=list(range(diff.changed_layer_count)),
                affected_tiers=[t.value for t in diff.affected_tiers],
                estimated_kv_transfer_bytes=best_score.breakdown.estimated_kv_transfer_bytes,
                reason=stab_result.explanation,
            )

            decision = ControlDecision(
                action=ControlAction.SWITCH,
                current_plan_id=curr_id,
                selected_plan_id=best_plan_id,
                current_cost=current_cost,
                selected_cost=best_cost,
                expected_gain=stab_result.expected_gain,
                threshold=self.config.switch_threshold,
                switch_allowed=True,
                reason=reason,
                explanation=stab_result.explanation,
                timestamp=ts,
                controller_cycle=cycle,
                prediction_used=prediction_used,
                is_proactive=is_proactive,
                is_safety_override=safety_eval.is_emergency,
                safety_status=safety_eval.status,
                stability_status={
                    "dwell_elapsed": stab_result.dwell_elapsed_seconds,
                    "cooldown_remaining": stab_result.cooldown_remaining_seconds,
                    "consecutive_count": stab_result.consecutive_count,
                },
                migration_request=migration_req,
            )

            # Update controller state upon approved switch
            self.state.last_switch_timestamp = ts
            self.state.last_switch_cycle = cycle
            self.state.switch_count += 1
            self.state.current_plan = best_score.candidate_plan.partition_plan
            self.state.current_plan_id = best_plan_id
            self.state.consecutive_preference_count = 0

        else:
            # Switch rejected by stability checker (dwell, threshold, hysteresis)
            action = ControlAction.KEEP_CURRENT
            reason = stab_result.reason or DecisionReason.BELOW_THRESHOLD

            decision = ControlDecision(
                action=action,
                current_plan_id=curr_id,
                selected_plan_id=best_plan_id,
                current_cost=current_cost,
                selected_cost=best_cost,
                expected_gain=stab_result.expected_gain,
                threshold=self.config.switch_threshold,
                switch_allowed=False,
                reason=reason,
                explanation=stab_result.explanation,
                timestamp=ts,
                controller_cycle=cycle,
                prediction_used=prediction_used,
                is_proactive=False,
                is_safety_override=False,
                safety_status=safety_eval.status,
                stability_status={
                    "dwell_elapsed": stab_result.dwell_elapsed_seconds,
                    "cooldown_remaining": stab_result.cooldown_remaining_seconds,
                    "consecutive_count": stab_result.consecutive_count,
                },
            )

        self._finalize_decision(decision)
        return decision

    def _finalize_decision(self, decision: ControlDecision) -> None:
        self.state.last_decision = decision
        self.history.record(decision)

    def replay(
        self,
        trace_steps: Sequence[Dict[str, Any]],
    ) -> List[ControlDecision]:
        """
        Execute controller sequentially across a sequence of recorded trace steps.

        Each step dictionary should contain:
            - 'state': RuntimeState
            - 'candidates': Sequence[CandidatePlan]
            - 'forecast': Optional[PredictionResult]
        """
        decisions: List[ControlDecision] = []
        for step in trace_steps:
            d = self.decide(
                state=step.get("state"),
                candidates=step.get("candidates"),
                forecast=step.get("forecast"),
            )
            decisions.append(d)
        return decisions

    def get_metrics(self) -> Dict[str, Any]:
        """Expose runtime statistics and stability metrics."""
        return self.history.get_metrics()
