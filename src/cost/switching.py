"""
Switching and repartitioning penalty model for candidate partition evaluation.

Computes transition overhead:
    P_switch(a) = w_layer * changed_layers
                + w_bound * (shift_distance + added/removed_cuts)
                + w_kv * kv_transfer_mb
                + w_disrupt * affected_tiers

Key research invariants:
    - Identical plan (candidate == current): P_switch = 0.0
    - Any plan change: P_switch > 0.0
    - Reuses Module 5's compare_plans() and PlanDifference without duplicate logic
    - Quantifies estimated KV-cache transfer volume without performing physical transfer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.cost.types import CostModelCalibration
from src.partitioning.candidate import CandidatePlan
from src.partitioning.comparison import PlanDifference, compare_plans
from src.runtime.partition import PartitionPlan
from src.state.types import RuntimeState
from src.telemetry.types import DataSource


@dataclass(frozen=True)
class SwitchingBreakdown:
    """Decomposed switching cost components."""
    switching_units: float
    layer_migration_cost: float
    boundary_change_cost: float
    kv_transfer_cost: float
    disruption_cost: float
    estimated_kv_transfer_bytes: int
    is_identical: bool
    provenance: DataSource


class SwitchingCostModel:
    """
    Computes structural repartitioning and migration penalties using PlanDifference.
    """

    def __init__(self, calibration: Optional[CostModelCalibration] = None) -> None:
        self.calibration = calibration or CostModelCalibration()

    def evaluate(
        self,
        candidate: CandidatePlan,
        current_plan: Optional[PartitionPlan] = None,
        state: Optional[RuntimeState] = None,
    ) -> SwitchingBreakdown:
        """
        Evaluate transition penalty from current active plan to proposed candidate.

        Args:
            candidate: Proposed CandidatePlan.
            current_plan: Currently active PartitionPlan (None if initial deployment).
            state: Optional current RuntimeState with KV-cache measurements.

        Returns:
            SwitchingBreakdown with granular penalty components.
        """
        # If no current plan is active, deployment is considered baseline zero penalty
        if current_plan is None:
            return SwitchingBreakdown(
                switching_units=0.0,
                layer_migration_cost=0.0,
                boundary_change_cost=0.0,
                kv_transfer_cost=0.0,
                disruption_cost=0.0,
                estimated_kv_transfer_bytes=0,
                is_identical=True,
                provenance=DataSource.ESTIMATED,
            )

        # Compute structural difference via Module 5 utility
        diff: PlanDifference = compare_plans(current_plan, candidate.partition_plan)

        # Invariant 1: Identical plans have strictly 0.0 switching cost
        if diff.is_identical:
            return SwitchingBreakdown(
                switching_units=0.0,
                layer_migration_cost=0.0,
                boundary_change_cost=0.0,
                kv_transfer_cost=0.0,
                disruption_cost=0.0,
                estimated_kv_transfer_bytes=0,
                is_identical=True,
                provenance=DataSource.ESTIMATED,
            )

        # Invariant 2: Non-identical plans incur positive switching cost
        w_layer = self.calibration.per_layer_switching_cost
        w_bound = self.calibration.boundary_shift_switching_cost
        w_kv = self.calibration.kv_transfer_cost_per_mb
        w_disrupt = self.calibration.tier_disruption_cost

        # 1. Layer migration cost
        layer_cost = diff.changed_layer_count * w_layer

        # 2. Boundary reconfiguration cost
        cut_edits = len(diff.added_boundaries) + len(diff.removed_boundaries)
        boundary_cost = (diff.boundary_shift_distance + cut_edits * 0.5) * w_bound

        # 3. Disruption cost (number of affected tiers)
        disruption_cost = len(diff.affected_tiers) * w_disrupt

        # 4. KV-cache migration volume estimation
        total_layers = current_plan.total_layers
        kv_bytes_total = 0
        if state and state.inference and state.inference.kv_cache_bytes:
            val = state.inference.kv_cache_bytes.value
            if val is not None and val > 0:
                kv_bytes_total = int(val)

        # Fraction of KV-cache associated with migrated layers
        migrated_ratio = float(diff.changed_layer_count / max(1, total_layers))
        estimated_kv_bytes = int(kv_bytes_total * migrated_ratio)
        estimated_kv_mb = float(estimated_kv_bytes / (1024.0 * 1024.0))
        kv_cost = estimated_kv_mb * w_kv

        total_units = layer_cost + boundary_cost + disruption_cost + kv_cost

        # Ensure strictly positive for non-identical plans
        if total_units <= 0.0:
            total_units = 0.01

        return SwitchingBreakdown(
            switching_units=float(total_units),
            layer_migration_cost=float(layer_cost),
            boundary_change_cost=float(boundary_cost),
            kv_transfer_cost=float(kv_cost),
            disruption_cost=float(disruption_cost),
            estimated_kv_transfer_bytes=estimated_kv_bytes,
            is_identical=False,
            provenance=DataSource.ESTIMATED,
        )
