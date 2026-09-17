"""
Memory pressure cost model for candidate partition evaluation.

Quantifies risk and headroom saturation across active execution tiers:
    M_pressure(a) = max_{t in active_tiers} [ required_memory(t) / usable_capacity(t) ]

Supports:
    - Multi-horizon evaluation: current state and forecast horizon trajectory
    - Preservation of UNKNOWN status when physical VRAM is unavailable (never fabricate 0)
    - Emulated memory evaluation with strict DataSource.EMULATED provenance tracking
    - Non-linear penalty when approaching threshold headroom
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.cost.types import ScoreStatus
from src.partitioning.candidate import CandidatePlan
from src.partitioning.metadata import TierCapacity
from src.prediction.types import PredictionResult
from src.runtime.tier import TierId
from src.state.types import RuntimeState
from src.telemetry.types import DataSource


@dataclass(frozen=True)
class MemoryPressureBreakdown:
    """Detailed memory pressure and headroom metrics."""
    pressure_ratio: float
    current_pressure: Optional[float]
    predicted_max_pressure: Optional[float]
    current_headroom_mb: Optional[float]
    predicted_min_headroom_mb: Optional[float]
    status: ScoreStatus
    provenance: DataSource


class MemoryPressureCostModel:
    """
    Computes provenance-aware memory pressure scores across current and forecast conditions.
    """

    def __init__(
        self,
        tier_capacities: Optional[Dict[TierId, TierCapacity]] = None,
        safety_margin_mb: float = 256.0,
        risk_threshold: float = 0.85,
    ) -> None:
        self.tier_capacities = tier_capacities or {}
        self.safety_margin_mb = safety_margin_mb
        self.risk_threshold = risk_threshold

    def evaluate(
        self,
        candidate: CandidatePlan,
        state: Optional[RuntimeState] = None,
        forecast: Optional[PredictionResult] = None,
    ) -> MemoryPressureBreakdown:
        """
        Evaluate memory pressure for a candidate plan.

        Args:
            candidate: CandidatePlan with estimated_memory_requirements per tier.
            state: Optional current RuntimeState.
            forecast: Optional multi-step resource forecast.

        Returns:
            MemoryPressureBreakdown with pressure ratios, headroom, status, and provenance.
        """
        # Collect required memory per active tier
        reqs = candidate.estimated_memory_requirements  # tier_key -> required_mb

        if not reqs:
            # Fallback if requirements not precomputed
            reqs = {t.value: 256.0 for t in candidate.active_tiers}

        current_pressures: List[float] = []
        current_headrooms: List[float] = []
        has_unknown = False
        is_emulated = False

        for tier in candidate.active_tiers:
            tier_key = tier.value
            required_mb = reqs.get(tier_key, 0.0)

            # Look up capacity from instance config or candidate metadata
            cap = self.tier_capacities.get(tier)
            if cap is None and "tier_capacities" in candidate.metadata:
                cap = candidate.metadata["tier_capacities"].get(tier)

            if cap is None or cap.available_memory_mb is None or cap.memory_provenance == DataSource.UNAVAILABLE:
                # VRAM or memory is unavailable on host
                has_unknown = True
                continue

            if cap.memory_provenance == DataSource.EMULATED:
                is_emulated = True

            available_mb = cap.available_memory_mb
            headroom = available_mb - required_mb
            current_headrooms.append(headroom)

            if available_mb > 0:
                current_pressures.append(required_mb / available_mb)
            else:
                current_pressures.append(10.0)  # Complete exhaustion

        # Handle forecast adjustments if available
        predicted_pressures: List[float] = []
        predicted_headrooms: List[float] = []

        if forecast:
            # Check for memory forecast targets (e.g. vram_free_mb or kv_cache_bytes)
            if "vram_free_mb" in forecast.targets:
                vram_fc = forecast.targets["vram_free_mb"]
                if vram_fc.is_available:
                    valid_free = [v for v in vram_fc.values if v is not None]
                    if valid_free:
                        min_free = min(valid_free)
                        # User device or primary edge pressure under minimum predicted free VRAM
                        for tier in candidate.active_tiers:
                            req_mb = reqs.get(tier.value, 0.0)
                            if min_free > 0:
                                predicted_pressures.append(req_mb / min_free)
                            predicted_headrooms.append(min_free - req_mb)
                elif vram_fc.status == "unavailable":
                    has_unknown = True

        # Aggregate metrics
        curr_p = max(current_pressures) if current_pressures else None
        curr_h = min(current_headrooms) if current_headrooms else None
        pred_p = max(predicted_pressures) if predicted_pressures else curr_p
        pred_h = min(predicted_headrooms) if predicted_headrooms else curr_h

        # Determine overall raw pressure ratio
        if has_unknown and not current_pressures:
            # All tiers have unavailable memory (CPU-only host without emulation)
            return MemoryPressureBreakdown(
                pressure_ratio=0.5,  # Neutral baseline for auditability
                current_pressure=None,
                predicted_max_pressure=None,
                current_headroom_mb=None,
                predicted_min_headroom_mb=None,
                status=ScoreStatus.UNKNOWN,
                provenance=DataSource.UNAVAILABLE,
            )

        effective_pressure = max(p for p in [curr_p, pred_p] if p is not None) if (curr_p or pred_p) else 0.5

        # Non-linear penalty if approaching risk threshold (> 0.85)
        if effective_pressure > self.risk_threshold:
            # Escalates rapidly: e.g. at 0.90 -> +0.25; at 1.0 -> +1.0
            excess = effective_pressure - self.risk_threshold
            effective_pressure += (excess / (1.0 - self.risk_threshold + 0.01)) ** 2

        provenance = DataSource.EMULATED if is_emulated else DataSource.ESTIMATED
        status = ScoreStatus.INFEASIBLE if effective_pressure >= 2.0 else ScoreStatus.FEASIBLE

        return MemoryPressureBreakdown(
            pressure_ratio=float(effective_pressure),
            current_pressure=curr_p,
            predicted_max_pressure=pred_p,
            current_headroom_mb=curr_h,
            predicted_min_headroom_mb=pred_h,
            status=status,
            provenance=provenance,
        )
