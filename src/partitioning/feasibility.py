"""
Feasibility evaluation engine for candidate partition plans.

Evaluates structural validity, current resource limits, and predicted horizon conditions,
explicitly preserving UNKNOWN when telemetry or hardware metrics are absent.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.partitioning.memory_estimator import TierMemoryRequirement
from src.partitioning.metadata import TierCapacity
from src.prediction.types import PredictionResult
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.state.types import RuntimeState


class FeasibilityStatus(str, Enum):
    """
    Feasibility assessment of a candidate partition plan.

    FEASIBLE   — All required known constraints pass with sufficient margin.
    INFEASIBLE — Violates structural limits or exceeds available memory.
    UNKNOWN    — Required resource telemetry is unavailable on host (e.g. no GPU).
    """
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


def evaluate_plan_feasibility(
    plan: PartitionPlan,
    tier_capacities: Dict[TierId, TierCapacity],
    memory_requirements: Dict[TierId, TierMemoryRequirement],
    current_state: Optional[RuntimeState] = None,
    forecast: Optional[PredictionResult] = None,
) -> Tuple[FeasibilityStatus, FeasibilityStatus, List[str], List[str]]:
    """
    Evaluate candidate plan feasibility now and over the prediction horizon.

    Args:
        plan: Candidate PartitionPlan.
        tier_capacities: Static capacity and activity state per tier.
        memory_requirements: Pre-estimated memory demands per tier.
        current_state: Optional current observed RuntimeState.
        forecast: Optional short-horizon PredictionResult from Module 4.

    Returns:
        (status_now, status_predicted, reasons, violations)
    """
    reasons: List[str] = []
    violations: List[str] = []

    active_tiers = plan.get_active_tiers()

    # 1. Structural & Tier Activity Check
    for tier_id, (start, end) in active_tiers:
        cap = tier_capacities.get(tier_id)
        if cap is None or not cap.is_enabled:
            violations.append(f"Tier '{tier_id.value}' is inactive or disabled in configuration")
            reasons.append(f"Tier '{tier_id.value}' is disabled")
            return FeasibilityStatus.INFEASIBLE, FeasibilityStatus.INFEASIBLE, reasons, violations

        layer_count = end - start + 1
        if cap.max_layers is not None and layer_count > cap.max_layers:
            violations.append(
                f"Tier '{tier_id.value}' assigned {layer_count} layers, exceeding max_layers limit {cap.max_layers}"
            )
            reasons.append(f"Tier '{tier_id.value}' max_layers exceeded")
            return FeasibilityStatus.INFEASIBLE, FeasibilityStatus.INFEASIBLE, reasons, violations

    # 2. Current Memory Feasibility Evaluation
    status_now = FeasibilityStatus.FEASIBLE

    for tier_id, req in memory_requirements.items():
        cap = tier_capacities.get(tier_id)
        if cap is None:
            continue

        # Determine effective current available memory
        avail_mb: Optional[float] = None

        if tier_id == TierId.USER_DEVICE and current_state is not None:
            if cap.device == "cuda":
                # Check VRAM if available
                vram_val = current_state.memory.vram_free_mb.value
                if vram_val is not None:
                    avail_mb = float(vram_val)
            else:
                # CPU device -> RAM available
                ram_val = current_state.memory.ram_available_mb.value
                if ram_val is not None:
                    avail_mb = float(ram_val)

        # Fallback to configured tier capacity if state did not provide measurement
        if avail_mb is None:
            avail_mb = cap.available_memory_mb

        if avail_mb is None:
            # Memory capacity is unknown (e.g. VRAM on CPU-only host)
            status_now = FeasibilityStatus.UNKNOWN
            reasons.append(f"Tier '{tier_id.value}' memory capacity/telemetry is unavailable")
        elif avail_mb < req.total_required_mb:
            status_now = FeasibilityStatus.INFEASIBLE
            violations.append(
                f"Tier '{tier_id.value}' current available memory {avail_mb:.1f} MB < required {req.total_required_mb:.1f} MB"
            )
            reasons.append(f"Insufficient current memory on '{tier_id.value}'")

    if status_now == FeasibilityStatus.FEASIBLE:
        reasons.append("All current structural and capacity constraints satisfied")

    # 3. Predicted Memory Feasibility Evaluation
    # If currently infeasible, predicted is also infeasible
    if status_now == FeasibilityStatus.INFEASIBLE:
        return status_now, FeasibilityStatus.INFEASIBLE, reasons, violations

    if forecast is None or not forecast.is_valid:
        # Without a valid forecast, predicted status reflects current status
        return status_now, status_now, reasons, violations

    status_pred = FeasibilityStatus.FEASIBLE

    for tier_id, req in memory_requirements.items():
        cap = tier_capacities.get(tier_id)
        if cap is None:
            continue

        if cap.device == "cuda":
            vram_fc = forecast.get_target("vram_free_mb")
            if vram_fc is None or not vram_fc.is_available or not vram_fc.values:
                status_pred = FeasibilityStatus.UNKNOWN
                reasons.append(f"Tier '{tier_id.value}' predicted VRAM is unavailable")
            else:
                # Check if any step across horizon drops below requirement
                valid_preds = [v for v in vram_fc.values if v is not None]
                if not valid_preds:
                    status_pred = FeasibilityStatus.UNKNOWN
                elif min(valid_preds) < req.total_required_mb:
                    status_pred = FeasibilityStatus.INFEASIBLE
                    violations.append(
                        f"Tier '{tier_id.value}' predicted minimum VRAM ({min(valid_preds):.1f} MB) < required ({req.total_required_mb:.1f} MB)"
                    )
                    reasons.append(f"Predicted VRAM exhaustion on '{tier_id.value}' over horizon")
        else:
            # CPU device memory (RAM) check if available in forecast
            ram_fc = forecast.get_target("ram_available_mb")
            if ram_fc is not None and ram_fc.is_available and ram_fc.values:
                valid_preds = [v for v in ram_fc.values if v is not None]
                if valid_preds and min(valid_preds) < req.total_required_mb:
                    status_pred = FeasibilityStatus.INFEASIBLE
                    violations.append(
                        f"Tier '{tier_id.value}' predicted minimum RAM ({min(valid_preds):.1f} MB) < required ({req.total_required_mb:.1f} MB)"
                    )
                    reasons.append(f"Predicted RAM exhaustion on '{tier_id.value}' over horizon")

    if status_pred == FeasibilityStatus.FEASIBLE and status_now == FeasibilityStatus.FEASIBLE:
        reasons.append("All predicted horizon resource constraints satisfied")

    return status_now, status_pred, reasons, violations
