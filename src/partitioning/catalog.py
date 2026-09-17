"""
SplitCatalog: Orchestrates structural partition enumeration, memory requirement estimation,
feasibility evaluation, and candidate plan filtering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.partitioning.candidate import CandidatePlan
from src.partitioning.comparison import PlanDifference, compare_plans
from src.partitioning.feasibility import FeasibilityStatus, evaluate_plan_feasibility
from src.partitioning.formatting import format_candidate_table
from src.partitioning.memory_estimator import estimate_plan_memory
from src.partitioning.metadata import ModelMetadata, TierCapacity
from src.partitioning.plan_id import generate_plan_id
from src.partitioning.structural import enumerate_structural_plans
from src.prediction.types import PredictionResult
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.state.types import RuntimeState


class SplitCatalog:
    """
    Catalog and generator of feasible candidate PartitionPlans.

    Translates model metadata, tier capacities, current state, and predicted resource
    conditions into structurally sound and resource-audited candidate plans.
    """

    def __init__(
        self,
        tier_capacities: Optional[Dict[TierId, TierCapacity]] = None,
        allow_monolithic: bool = True,
        allow_two_tier: bool = True,
        allow_three_tier: bool = True,
        safety_margin_mb: float = 256.0,
        default_context_length: int = 128,
    ) -> None:
        self.tier_capacities = tier_capacities or self._default_capacities()
        self.allow_monolithic = allow_monolithic
        self.allow_two_tier = allow_two_tier
        self.allow_three_tier = allow_three_tier
        self.safety_margin_mb = safety_margin_mb
        self.default_context_length = default_context_length

    @staticmethod
    def _default_capacities() -> Dict[TierId, TierCapacity]:
        return {
            TierId.USER_DEVICE: TierCapacity(tier_id=TierId.USER_DEVICE, device="cpu", is_enabled=True),
            TierId.EDGE_A: TierCapacity(tier_id=TierId.EDGE_A, device="cpu", is_enabled=True),
            TierId.EDGE_B: TierCapacity(tier_id=TierId.EDGE_B, device="cpu", is_enabled=True),
        }

    def generate(
        self,
        model_metadata: ModelMetadata,
        current_state: Optional[RuntimeState] = None,
        forecast: Optional[PredictionResult] = None,
        current_plan: Optional[PartitionPlan] = None,
        mode: str = "exhaustive",
        context_length: Optional[int] = None,
    ) -> List[CandidatePlan]:
        """
        Generate candidate PartitionPlans and evaluate their feasibility.

        Args:
            model_metadata: Model dimension and parameter specs.
            current_state: Optional currently observed RuntimeState.
            forecast: Optional short-horizon PredictionResult.
            current_plan: Optional currently active PartitionPlan for delta analysis.
            mode: 'exhaustive' (returns all structural candidates with feasibility metadata)
                  or 'filtered' (excludes candidates marked INFEASIBLE).
            context_length: Sequence length for KV-cache estimation.
        """
        ctx_len = context_length or self.default_context_length

        # 1. Enumerate structural plans based on enabled tiers
        enabled_tier_ids = [t for t, cap in self.tier_capacities.items() if cap.is_enabled]
        structural_plans = enumerate_structural_plans(
            total_layers=model_metadata.total_layers,
            enabled_tiers=enabled_tier_ids,
            allow_monolithic=self.allow_monolithic,
            allow_two_tier=self.allow_two_tier,
            allow_three_tier=self.allow_three_tier,
        )

        candidates: List[CandidatePlan] = []

        for p in structural_plans:
            pid = generate_plan_id(p)
            active_tiers = [t for t, _ in p.get_active_tiers()]
            layer_assign = {t.value: rng for t, rng in p.get_active_tiers()}
            boundaries = p.get_transfer_boundaries()

            # 2. Estimate memory requirements
            mem_reqs = estimate_plan_memory(
                plan=p,
                model_metadata=model_metadata,
                context_length=ctx_len,
                safety_margin_mb=self.safety_margin_mb,
            )

            # 3. Evaluate feasibility (current and predicted)
            s_now, s_pred, reasons, violations = evaluate_plan_feasibility(
                plan=p,
                tier_capacities=self.tier_capacities,
                memory_requirements=mem_reqs,
                current_state=current_state,
                forecast=forecast,
            )

            # 4. Optional delta comparison against current plan
            meta: Dict[str, Any] = {
                "total_layers": model_metadata.total_layers,
                "context_length": ctx_len,
            }
            if current_plan is not None:
                diff = compare_plans(current_plan, p)
                meta["is_current_plan"] = diff.is_identical
                meta["changed_layers_vs_current"] = diff.changed_layer_count
                meta["boundary_shift_vs_current"] = diff.boundary_shift_distance

            candidate = CandidatePlan(
                plan_id=pid,
                partition_plan=p,
                active_tiers=active_tiers,
                layer_assignment=layer_assign,
                number_of_boundaries=len(boundaries),
                boundaries=boundaries,
                feasibility_now=s_now,
                feasibility_predicted=s_pred,
                feasibility_reasons=reasons,
                estimated_memory_requirements={
                    t.value: req.total_required_mb for t, req in mem_reqs.items()
                },
                resource_violations=violations,
                metadata=meta,
            )

            if mode == "filtered":
                # Keep only plans that are not infeasible either now or in prediction
                if (
                    candidate.feasibility_now != FeasibilityStatus.INFEASIBLE
                    and candidate.feasibility_predicted != FeasibilityStatus.INFEASIBLE
                ):
                    candidates.append(candidate)
            else:
                candidates.append(candidate)

        return candidates

    def filter_feasible(self, candidates: Sequence[CandidatePlan]) -> List[CandidatePlan]:
        """Filter candidates to strictly those evaluated as FeasibilityStatus.FEASIBLE."""
        return [c for c in candidates if c.is_feasible]

    def compare_plans(
        self,
        current_plan: PartitionPlan,
        candidate_plan: PartitionPlan,
    ) -> PlanDifference:
        """Utility method to compare two partition plans."""
        return compare_plans(current_plan, candidate_plan)

    def format_table(self, candidates: Sequence[CandidatePlan]) -> str:
        """Render a formatted ASCII table of candidate plans."""
        return format_candidate_table(candidates)
