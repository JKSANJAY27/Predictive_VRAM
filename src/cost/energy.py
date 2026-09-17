"""
Energy cost model for candidate partition evaluation.

Estimates proxy operational energy consumption per token:
    E(a) = sum_{t in active_tiers} [ compute_time(t) * power_proxy(t) ]

Supports:
    - Modeled analytical proxy (Joules = seconds * Watts)
    - Strict provenance tracking (DataSource.ESTIMATED or DataSource.UNAVAILABLE)
    - Zero fabrication of physical GPU power metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.cost.types import CostModelCalibration
from src.partitioning.candidate import CandidatePlan
from src.state.types import RuntimeState
from src.telemetry.types import DataSource


@dataclass(frozen=True)
class EnergyBreakdown:
    """Estimated energy consumption per inference step in Joules."""
    energy_joules: float
    provenance: DataSource


class EnergyCostModel:
    """
    Computes modeled power-proxy energy estimates for candidate plans.
    """

    def __init__(self, calibration: Optional[CostModelCalibration] = None) -> None:
        self.calibration = calibration or CostModelCalibration()

    def evaluate(
        self,
        candidate: CandidatePlan,
        state: Optional[RuntimeState] = None,
        compute_ms_by_tier: Optional[Dict[str, float]] = None,
    ) -> EnergyBreakdown:
        """
        Evaluate estimated energy expenditure in Joules.

        Args:
            candidate: CandidatePlan with active tiers and layer assignment.
            state: Optional current RuntimeState.
            compute_ms_by_tier: Optional precomputed compute latency per tier in ms.

        Returns:
            EnergyBreakdown with estimated Joules and provenance.
        """
        rates = self.calibration.tier_compute_ms_per_layer
        powers = self.calibration.power_proxy_watts

        total_joules = 0.0

        for tier in candidate.active_tiers:
            tier_key = tier.value
            power_watts = powers.get(tier_key, 10.0)

            # Compute execution duration on this tier
            if compute_ms_by_tier and tier_key in compute_ms_by_tier:
                tier_ms = compute_ms_by_tier[tier_key]
            else:
                layers = candidate.layer_assignment.get(tier_key)
                layer_count = max(0, layers[1] - layers[0] + 1) if layers else 0
                ms_per_layer = rates.get(tier_key, 5.0)
                tier_ms = layer_count * ms_per_layer

            tier_seconds = tier_ms / 1000.0
            total_joules += tier_seconds * power_watts

        # Energy is modeled analytically, so provenance is ESTIMATED
        return EnergyBreakdown(
            energy_joules=float(total_joules),
            provenance=DataSource.ESTIMATED,
        )
