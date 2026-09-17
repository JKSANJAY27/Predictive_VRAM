"""
Steady-state communication cost model for candidate partition evaluation.

Estimates recurring per-token inter-tier network transfer volume:
    C(a) = num_boundaries * activation_tensor_size_mb

Crucial research invariant:
    Distinguishes steady-state per-token communication volume from
    one-time partition switching/migration traffic (P_switch) to prevent
    double-counting network overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.partitioning.candidate import CandidatePlan
from src.state.types import RuntimeState
from src.telemetry.types import DataSource


@dataclass(frozen=True)
class CommunicationBreakdown:
    """Steady-state communication volume and boundary count."""
    steady_state_bytes: int
    steady_state_mb: float
    transfer_count: int
    provenance: DataSource


class CommunicationCostModel:
    """
    Computes transparent steady-state data transfer costs for candidate plans.
    """

    def __init__(self, default_activation_bytes: int = 4096) -> None:
        self.default_activation_bytes = default_activation_bytes

    def evaluate(
        self,
        candidate: CandidatePlan,
        state: Optional[RuntimeState] = None,
    ) -> CommunicationBreakdown:
        """
        Evaluate steady-state transfer volume for a single forward inference step.

        Args:
            candidate: Proposed CandidatePlan with boundary count.
            state: Optional current RuntimeState with measured activation sizes.

        Returns:
            CommunicationBreakdown with byte count, MB, and transfer count.
        """
        if candidate.number_of_boundaries == 0:
            return CommunicationBreakdown(
                steady_state_bytes=0,
                steady_state_mb=0.0,
                transfer_count=0,
                provenance=DataSource.ESTIMATED,
            )

        # Determine activation tensor byte size
        act_bytes = self.default_activation_bytes
        provenance = DataSource.ESTIMATED

        if state and state.activation and state.activation.latest_activation_bytes:
            tagged_val = state.activation.latest_activation_bytes
            if tagged_val.value is not None and tagged_val.value > 0:
                act_bytes = int(tagged_val.value)
                provenance = tagged_val.source

        total_bytes = act_bytes * candidate.number_of_boundaries
        total_mb = float(total_bytes / (1024.0 * 1024.0))

        return CommunicationBreakdown(
            steady_state_bytes=total_bytes,
            steady_state_mb=total_mb,
            transfer_count=candidate.number_of_boundaries,
            provenance=provenance,
        )
