"""
CostModel: Central coordinator for candidate plan scoring and trade-off evaluation.

Orchestrates:
    - Decomposed latency estimation
    - Steady-state communication volume
    - Multi-horizon memory pressure and headroom
    - Power-proxy energy estimation
    - Structural switching and migration penalties
    - Deterministic normalization and configurable weighting
    - Human-readable cost breakdown tables and explanation strings

Strict research constraint:
    Scores candidates ONLY. Does NOT implement argmin, keep/migrate decisions,
    threshold comparisons, cooldown, or runtime migration (deferred to Module 7).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from src.cost.communication import CommunicationCostModel
from src.cost.energy import EnergyCostModel
from src.cost.latency import LatencyCostModel
from src.cost.memory import MemoryPressureCostModel
from src.cost.normalization import CostNormalizer, NormalizationConfig
from src.cost.switching import SwitchingCostModel
from src.cost.types import (
    CandidateScore,
    CostBreakdown,
    CostModelCalibration,
    CostWeights,
    ScoreStatus,
)
from src.partitioning.candidate import CandidatePlan
from src.partitioning.feasibility import FeasibilityStatus
from src.partitioning.metadata import TierCapacity
from src.prediction.types import PredictionResult
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.state.types import RuntimeState


class CostModel:
    """
    Evaluates CandidatePlans against the composite objective function:
        J(a) = alpha * L(a) + beta * C(a) + gamma * M(a) + delta * E(a) + epsilon * P_switch(a)
    """

    def __init__(
        self,
        weights: Optional[CostWeights] = None,
        normalization_config: Optional[NormalizationConfig] = None,
        calibration: Optional[CostModelCalibration] = None,
        tier_capacities: Optional[Dict[TierId, TierCapacity]] = None,
    ) -> None:
        if isinstance(weights, dict):
            self.weights = CostWeights.from_dict(weights)
        else:
            self.weights = weights or CostWeights()
        self.normalization_config = normalization_config or NormalizationConfig()
        self.calibration = calibration or CostModelCalibration()
        self.tier_capacities = tier_capacities or {}

        # Component models
        self.normalizer = CostNormalizer(self.normalization_config)
        self.latency_model = LatencyCostModel(self.calibration)
        self.comm_model = CommunicationCostModel()
        self.memory_model = MemoryPressureCostModel(self.tier_capacities)
        self.energy_model = EnergyCostModel(self.calibration)
        self.switching_model = SwitchingCostModel(self.calibration)

    def score(
        self,
        candidate: CandidatePlan,
        state: Optional[RuntimeState] = None,
        forecast: Optional[PredictionResult] = None,
        current_plan: Optional[PartitionPlan] = None,
    ) -> CandidateScore:
        """
        Score a single candidate plan under current and forecast conditions.

        Args:
            candidate: CandidatePlan to evaluate.
            state: Optional current RuntimeState.
            forecast: Optional multi-step prediction result.
            current_plan: Currently active PartitionPlan (for transition penalty).

        Returns:
            CandidateScore containing granular breakdown and total score.
        """
        # 1. Evaluate raw sub-component models
        lat_br = self.latency_model.evaluate(candidate, state=state, forecast=forecast)
        comm_br = self.comm_model.evaluate(candidate, state=state)
        mem_br = self.memory_model.evaluate(candidate, state=state, forecast=forecast)
        energy_br = self.energy_model.evaluate(candidate, state=state)
        switch_br = self.switching_model.evaluate(candidate, current_plan=current_plan, state=state)

        # 2. Normalize components
        lat_norm = self.normalizer.normalize_latency(lat_br.total_latency_ms)
        comm_norm = self.normalizer.normalize_communication(comm_br.steady_state_mb)
        mem_norm = self.normalizer.normalize_memory_pressure(mem_br.pressure_ratio)
        energy_norm = self.normalizer.normalize_energy(energy_br.energy_joules)
        switch_norm = self.normalizer.normalize_switching(switch_br.switching_units)

        # 3. Apply weights and ablation toggles
        lat_cost = (self.weights.alpha * lat_norm) if self.weights.use_latency else 0.0
        comm_cost = (self.weights.beta * comm_norm) if self.weights.use_communication else 0.0
        mem_cost = (self.weights.gamma * mem_norm) if self.weights.use_memory else 0.0
        energy_cost = (self.weights.delta * energy_norm) if self.weights.use_energy else 0.0
        switch_cost = (self.weights.epsilon * switch_norm) if self.weights.use_switching else 0.0

        base_total = lat_cost + comm_cost + mem_cost + energy_cost + switch_cost

        # 4. Feasibility & Uncertainty Auditing
        is_infeasible = (
            candidate.feasibility_now == FeasibilityStatus.INFEASIBLE
            or candidate.feasibility_predicted == FeasibilityStatus.INFEASIBLE
            or mem_br.status == ScoreStatus.INFEASIBLE
        )
        is_unknown = (
            candidate.feasibility_now == FeasibilityStatus.UNKNOWN
            or candidate.feasibility_predicted == FeasibilityStatus.UNKNOWN
            or mem_br.status == ScoreStatus.UNKNOWN
        )

        if is_infeasible:
            status = ScoreStatus.INFEASIBLE
            feasible = False
            # Apply large barrier penalty to ensure infeasible plans never win in comparison
            total_cost = base_total + 1000.0
        elif is_unknown:
            status = ScoreStatus.UNKNOWN
            feasible = True  # Feasible subject to unmeasured capacity
            total_cost = base_total
        else:
            status = ScoreStatus.FEASIBLE
            feasible = True
            total_cost = base_total

        # 5. Build CostBreakdown
        breakdown = CostBreakdown(
            latency_raw_ms=lat_br.total_latency_ms,
            communication_raw_mb=comm_br.steady_state_mb,
            memory_pressure_raw=mem_br.pressure_ratio,
            energy_raw_j=energy_br.energy_joules,
            switching_raw_units=switch_br.switching_units,
            latency_normalized=lat_norm,
            communication_normalized=comm_norm,
            memory_pressure_normalized=mem_norm,
            energy_normalized=energy_norm,
            switching_normalized=switch_norm,
            latency_cost=lat_cost,
            communication_cost=comm_cost,
            memory_pressure_cost=mem_cost,
            energy_cost=energy_cost,
            switching_cost=switch_cost,
            total_cost=total_cost,
            compute_latency_ms=lat_br.compute_latency_ms,
            comm_latency_ms=lat_br.comm_latency_ms,
            queue_latency_ms=lat_br.queue_latency_ms,
            steady_state_bytes=comm_br.steady_state_bytes,
            boundary_count=candidate.number_of_boundaries,
            current_headroom_mb=mem_br.current_headroom_mb,
            predicted_min_headroom_mb=mem_br.predicted_min_headroom_mb,
            current_pressure=mem_br.current_pressure,
            predicted_max_pressure=mem_br.predicted_max_pressure,
            layer_migration_cost=switch_br.layer_migration_cost,
            boundary_change_cost=switch_br.boundary_change_cost,
            kv_transfer_cost=switch_br.kv_transfer_cost,
            estimated_kv_transfer_bytes=switch_br.estimated_kv_transfer_bytes,
            latency_provenance=lat_br.provenance,
            communication_provenance=comm_br.provenance,
            memory_provenance=mem_br.provenance,
            energy_provenance=energy_br.provenance,
            switching_provenance=switch_br.provenance,
            weights=self.weights,
        )

        # 6. Generate human-readable explanation
        explanation = self._build_explanation(candidate, breakdown, status, switch_br.is_identical)

        # Determine confidence based on forecast validity
        confidence = 1.0
        if forecast and not forecast.is_valid:
            confidence = 0.5
        elif is_unknown:
            confidence = 0.7

        return CandidateScore(
            candidate_plan=candidate,
            breakdown=breakdown,
            total_cost=total_cost,
            feasible=feasible,
            status=status,
            confidence=confidence,
            explanation=explanation,
            metadata={
                "is_current_plan": switch_br.is_identical,
                "forecast_used": forecast is not None,
            },
        )

    def score_candidates(
        self,
        candidates: Sequence[CandidatePlan],
        state: Optional[RuntimeState] = None,
        forecast: Optional[PredictionResult] = None,
        current_plan: Optional[PartitionPlan] = None,
    ) -> List[CandidateScore]:
        """
        Score a sequence of candidates, strictly preserving input ordering.

        Note: Does NOT sort or filter candidates. Ordering and ranking are
        explicitly deferred to the controller in Module 7.
        """
        return [
            self.score(
                candidate=c,
                state=state,
                forecast=forecast,
                current_plan=current_plan,
            )
            for c in candidates
        ]

    def _build_explanation(
        self,
        candidate: CandidatePlan,
        b: CostBreakdown,
        status: ScoreStatus,
        is_current: bool,
    ) -> str:
        parts: List[str] = [f"Plan '{candidate.plan_id}' [{status.value.upper()}]:"]

        if status == ScoreStatus.INFEASIBLE:
            parts.append("Infeasible due to resource constraint violations.")
            return " ".join(parts)

        # Latency
        parts.append(f"latency {b.latency_raw_ms:.1f}ms (compute {b.compute_latency_ms:.1f}ms, comm {b.comm_latency_ms:.1f}ms);")

        # Communication
        if b.boundary_count == 0:
            parts.append("zero inter-tier transfer;")
        else:
            parts.append(f"{b.boundary_count} cuts ({b.communication_raw_mb * 1024:.1f} KB/step);")

        # Memory pressure
        if b.memory_pressure_cost > 0:
            parts.append(f"pressure ratio {b.memory_pressure_raw:.2f};")

        # Switching
        if is_current:
            parts.append("current active plan (0 switching penalty).")
        else:
            parts.append(f"switching penalty {b.switching_cost:.2f}.")

        return " ".join(parts)

    def format_cost_table(self, scores: Sequence[CandidateScore]) -> str:
        """
        Render an auditable text table of candidate scores.
        """
        if not scores:
            return "No candidate scores to display."

        headers = (
            f"{'PLAN ID':<22} | {'STATUS':<10} | {'LAT (ms)':<9} | {'COMM (MB)':<10} "
            f"| {'MEM P':<6} | {'ENERGY':<7} | {'SWITCH':<7} | {'TOTAL':<8}"
        )
        sep = "-" * len(headers)
        lines = [headers, sep]

        for s in scores:
            b = s.breakdown
            stat_str = s.status.value.upper()
            lat_str = f"{b.latency_raw_ms:7.1f}"
            comm_str = f"{b.communication_raw_mb:8.4f}"
            mem_str = f"{b.memory_pressure_raw:5.2f}"
            energy_str = f"{b.energy_raw_j:6.2f}"
            switch_str = f"{b.switching_cost:6.2f}"
            tot_str = f"{s.total_cost:7.3f}"

            lines.append(
                f"{s.candidate_plan.plan_id:<22} | {stat_str:<10} | {lat_str} | {comm_str} "
                f"| {mem_str} | {energy_str} | {switch_str} | {tot_str}"
            )

        return "\n".join(lines)

    def verify_monotonicity(self, candidate: CandidatePlan) -> Dict[str, bool]:
        """
        Diagnostic verification: Confirms that increasing each cost dimension
        strictly increases the total score, all else held equal.
        """
        results: Dict[str, bool] = {}

        # 1. Switching monotonicity: changed plan > identical plan
        score_ident = self.score(candidate, current_plan=candidate.partition_plan)
        # Construct a synthetic current plan with different cut
        total_layers = candidate.partition_plan.total_layers
        diff_plan = PartitionPlan.monolithic(total_layers=total_layers)
        if candidate.plan_id != "local":
            score_diff = self.score(candidate, current_plan=diff_plan)
            results["switching_monotonicity"] = score_diff.total_cost > score_ident.total_cost
        else:
            results["switching_monotonicity"] = True

        # 2. Latency monotonicity: higher latency weight -> higher total score
        w_low = CostWeights(alpha=0.5, beta=0.0, gamma=0.0, delta=0.0, epsilon=0.0)
        w_high = CostWeights(alpha=1.5, beta=0.0, gamma=0.0, delta=0.0, epsilon=0.0)
        m_low = CostModel(weights=w_low)
        m_high = CostModel(weights=w_high)
        s_low = m_low.score(candidate)
        s_high = m_high.score(candidate)
        results["latency_monotonicity"] = s_high.total_cost > s_low.total_cost

        # 3. Communication monotonicity
        w_c_low = CostWeights(alpha=0.0, beta=0.1, gamma=0.0, delta=0.0, epsilon=0.0)
        w_c_high = CostWeights(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0, epsilon=0.0)
        if candidate.number_of_boundaries > 0:
            results["communication_monotonicity"] = (
                CostModel(weights=w_c_high).score(candidate).total_cost
                > CostModel(weights=w_c_low).score(candidate).total_cost
            )
        else:
            results["communication_monotonicity"] = True

        # 4. Energy monotonicity
        w_e_low = CostWeights(alpha=0.0, beta=0.0, gamma=0.0, delta=0.1, epsilon=0.0)
        w_e_high = CostWeights(alpha=0.0, beta=0.0, gamma=0.0, delta=1.0, epsilon=0.0)
        results["energy_monotonicity"] = (
            CostModel(weights=w_e_high).score(candidate).total_cost
            > CostModel(weights=w_e_low).score(candidate).total_cost
        )

        return results
