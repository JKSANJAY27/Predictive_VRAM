"""
Latency cost model for candidate partition evaluation.

Decomposes end-to-end token latency into:
    L(a) = L_compute(a) + L_comm(a) + L_queue(a)

Where:
    - L_compute: Sum of per-tier execution latencies based on layer assignments
    - L_comm: Transfer time across cut boundaries (activation bytes / bandwidth + RTT)
    - L_queue: Execution queueing delay influenced by host system utilization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.cost.types import CostModelCalibration
from src.partitioning.candidate import CandidatePlan
from src.prediction.types import PredictionResult
from src.runtime.tier import TierId
from src.state.types import RuntimeState
from src.telemetry.types import DataSource


@dataclass(frozen=True)
class LatencyBreakdown:
    """Decomposed latency components in milliseconds."""
    compute_latency_ms: float
    comm_latency_ms: float
    queue_latency_ms: float
    total_latency_ms: float
    provenance: DataSource


class LatencyCostModel:
    """
    Computes transparent, predictable latency estimates for candidate plans.
    """

    def __init__(self, calibration: Optional[CostModelCalibration] = None) -> None:
        self.calibration = calibration or CostModelCalibration()

    def evaluate(
        self,
        candidate: CandidatePlan,
        state: Optional[RuntimeState] = None,
        forecast: Optional[PredictionResult] = None,
    ) -> LatencyBreakdown:
        """
        Evaluate predicted token latency for a candidate plan.

        Args:
            candidate: Proposed CandidatePlan with active tiers and boundaries.
            state: Optional current RuntimeState.
            forecast: Optional multi-step resource forecast.

        Returns:
            LatencyBreakdown with compute, comm, queue, and total latency in ms.
        """
        # 1. Compute Latency: per-tier layers * compute rate
        compute_ms = 0.0
        rates = self.calibration.tier_compute_ms_per_layer

        for tier in candidate.active_tiers:
            tier_key = tier.value
            ms_per_layer = rates.get(tier_key, 5.0)

            # Determine layers assigned to this tier
            layers = candidate.layer_assignment.get(tier_key)
            if layers is not None:
                start_l, end_l = layers
                layer_count = max(0, end_l - start_l + 1)
            else:
                layer_count = 0

            compute_ms += layer_count * ms_per_layer

        # 2. Communication Latency across boundaries
        comm_ms = 0.0
        provenance = DataSource.ESTIMATED

        if candidate.number_of_boundaries > 0:
            # Determine effective bandwidth (Mbps)
            bw_mbps = 50.0  # default baseline
            if forecast and "bandwidth_mbps" in forecast.targets:
                target_fc = forecast.targets["bandwidth_mbps"]
                valid_vals = [v for v in target_fc.values if v is not None and v > 0]
                if valid_vals:
                    bw_mbps = float(sum(valid_vals) / len(valid_vals))
            elif state and state.network.bandwidth_mbps.value is not None:
                if state.network.bandwidth_mbps.value > 0:
                    bw_mbps = float(state.network.bandwidth_mbps.value)
                    if state.network.bandwidth_mbps.source == DataSource.EMULATED:
                        provenance = DataSource.EMULATED

            # Determine RTT latency (ms)
            rtt_ms = self.calibration.base_rtt_ms
            if forecast and "latency_ms" in forecast.targets:
                target_lat = forecast.targets["latency_ms"]
                valid_lats = [v for v in target_lat.values if v is not None and v >= 0]
                if valid_lats:
                    rtt_ms = float(sum(valid_lats) / len(valid_lats))
            elif state and state.network.latency_ms.value is not None:
                if state.network.latency_ms.value >= 0:
                    rtt_ms = float(state.network.latency_ms.value)
                    if state.network.latency_ms.source == DataSource.EMULATED:
                        provenance = DataSource.EMULATED

            # Activation tensor size (bytes)
            act_bytes = 4096.0  # default token activation: e.g. 1 x 1 x 768 * 4 bytes
            if state and state.activation and state.activation.latest_activation_bytes:
                val = state.activation.latest_activation_bytes.value
                if val is not None and val > 0:
                    act_bytes = float(val)

            # Transfer time per boundary = transmission time + one-way propagation / RTT
            # (act_bytes * 8 bits) / (bw_mbps * 1e6 bits/sec) * 1000 ms/sec
            transfer_time_ms = (act_bytes * 8.0) / (bw_mbps * 1e6) * 1000.0
            total_per_boundary_ms = transfer_time_ms + rtt_ms

            comm_ms = total_per_boundary_ms * candidate.number_of_boundaries

        # 3. Queueing Latency
        queue_ms = 0.0
        if state and state.compute and state.compute.cpu_utilization.value is not None:
            cpu_util = state.compute.cpu_utilization.value
            # Gentle queueing penalty based on host utilization
            if cpu_util > 50.0:
                queue_ms = (compute_ms * 0.1) * ((cpu_util - 50.0) / 50.0)

        total_ms = compute_ms + comm_ms + queue_ms

        return LatencyBreakdown(
            compute_latency_ms=compute_ms,
            comm_latency_ms=comm_ms,
            queue_latency_ms=queue_ms,
            total_latency_ms=total_ms,
            provenance=provenance,
        )
