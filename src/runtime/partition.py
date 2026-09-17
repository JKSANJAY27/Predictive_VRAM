"""
PartitionPlan abstraction and validation for distributed split inference.

Ensures mathematically and structurally sound layer assignments across execution tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.runtime.tier import TierId


@dataclass(frozen=True)
class TransferBoundary:
    """Represents a transfer boundary where activation tensors move between tiers."""
    source_tier: TierId
    destination_tier: TierId
    cut_layer: int  # Layer index after which transfer occurs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tier": self.source_tier.value,
            "destination_tier": self.destination_tier.value,
            "cut_layer": self.cut_layer,
        }


@dataclass(frozen=True)
class PartitionPlan:
    """
    Defines the partition plan of an autoregressive Transformer across tiers.

    Attributes:
        total_layers: Total number of transformer blocks in the target model.
        user_device: Optional (start_layer, end_layer) inclusive for UserDevice.
        edge_a: Optional (start_layer, end_layer) inclusive for EdgeA.
        edge_b: Optional (start_layer, end_layer) inclusive for EdgeB.
    """
    total_layers: int
    user_device: Optional[Tuple[int, int]] = None
    edge_a: Optional[Tuple[int, int]] = None
    edge_b: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        self.validate()

    def get_tier_range(self, tier_id: TierId) -> Optional[Tuple[int, int]]:
        """Retrieve layer range assigned to a specific tier."""
        if tier_id == TierId.USER_DEVICE:
            return self.user_device
        elif tier_id == TierId.EDGE_A:
            return self.edge_a
        elif tier_id == TierId.EDGE_B:
            return self.edge_b
        else:
            raise ValueError(f"Unknown tier_id: {tier_id}")

    def get_active_tiers(self) -> List[Tuple[TierId, Tuple[int, int]]]:
        """Return list of active (tier_id, layer_range) ordered by layer sequence."""
        active = []
        for tier_id in (TierId.USER_DEVICE, TierId.EDGE_A, TierId.EDGE_B):
            rng = self.get_tier_range(tier_id)
            if rng is not None:
                active.append((tier_id, rng))
        return active

    def validate(self) -> None:
        """
        Strict validation of the partition plan:
        1. total_layers must be positive.
        2. Tier ranges must be valid (0 <= start <= end < total_layers).
        3. All layers 0 to total_layers - 1 must be covered without gaps.
        4. No layer may be assigned more than once (no overlap).
        5. Sequential execution order must be strictly preserved:
           UserDevice -> EdgeA -> EdgeB.
        """
        if self.total_layers <= 0:
            raise ValueError(f"total_layers must be positive, got {self.total_layers}")

        active_tiers = self.get_active_tiers()
        if not active_tiers:
            raise ValueError("PartitionPlan must have at least one active tier assigned.")

        # Check individual ranges
        covered_layers: List[int] = []
        for tier_id, (start, end) in active_tiers:
            if start < 0 or end >= self.total_layers:
                raise ValueError(
                    f"Tier '{tier_id.value}' layer range ({start}, {end}) is outside "
                    f"valid model layer boundaries [0, {self.total_layers - 1}]."
                )
            if end < start:
                raise ValueError(
                    f"Tier '{tier_id.value}' has invalid range ({start}, {end}): "
                    f"start layer must be <= end layer."
                )

        # Check ordering and continuity
        current_expected_layer = 0
        for tier_id, (start, end) in active_tiers:
            if start != current_expected_layer:
                if start > current_expected_layer:
                    missing = list(range(current_expected_layer, start))
                    raise ValueError(
                        f"Missing layers {missing} before tier '{tier_id.value}'. "
                        f"Expected start layer {current_expected_layer}, got {start}."
                    )
                else:
                    overlap = list(range(start, current_expected_layer))
                    raise ValueError(
                        f"Duplicated/overlapping layers {overlap} assigned to tier '{tier_id.value}'."
                    )
            current_expected_layer = end + 1

        if current_expected_layer != self.total_layers:
            missing = list(range(current_expected_layer, self.total_layers))
            raise ValueError(
                f"Missing final layers {missing}. Plan only covers up to layer "
                f"{current_expected_layer - 1} out of {self.total_layers} total layers."
            )

    def get_tier_for_layer(self, layer_idx: int) -> TierId:
        """Return the TierId responsible for executing the specified layer."""
        if layer_idx < 0 or layer_idx >= self.total_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range for model with {self.total_layers} layers."
            )
        for tier_id, (start, end) in self.get_active_tiers():
            if start <= layer_idx <= end:
                return tier_id
        raise RuntimeError(f"Unassigned layer index {layer_idx} (validation failed).")

    def get_transfer_boundaries(self) -> List[TransferBoundary]:
        """
        Identify all inter-tier transfer boundaries in this plan.
        Returns a list of TransferBoundary objects.
        """
        active = self.get_active_tiers()
        boundaries: List[TransferBoundary] = []
        for i in range(len(active) - 1):
            src_tier, (_, src_end) = active[i]
            dst_tier, _ = active[i + 1]
            boundaries.append(
                TransferBoundary(
                    source_tier=src_tier,
                    destination_tier=dst_tier,
                    cut_layer=src_end,
                )
            )
        return boundaries

    def to_dict(self) -> Dict[str, Any]:
        """Convert partition plan to structured dictionary."""
        return {
            "total_layers": self.total_layers,
            "user_device": list(self.user_device) if self.user_device else None,
            "edge_a": list(self.edge_a) if self.edge_a else None,
            "edge_b": list(self.edge_b) if self.edge_b else None,
            "boundaries": [b.to_dict() for b in self.get_transfer_boundaries()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PartitionPlan:
        return cls(
            total_layers=int(data["total_layers"]),
            user_device=tuple(data["user_device"]) if data.get("user_device") is not None else None,
            edge_a=tuple(data["edge_a"]) if data.get("edge_a") is not None else None,
            edge_b=tuple(data["edge_b"]) if data.get("edge_b") is not None else None,
        )

    @classmethod
    def monolithic(cls, total_layers: int) -> PartitionPlan:
        """Create a monolithic plan with all layers executing on User Device."""
        return cls(
            total_layers=total_layers,
            user_device=(0, total_layers - 1),
            edge_a=None,
            edge_b=None,
        )

    @classmethod
    def two_tier(cls, total_layers: int, cut_layer: int) -> PartitionPlan:
        """
        Create a 2-tier plan split between User Device and Edge A.
        cut_layer: Last layer on User Device (0-indexed).
        """
        if cut_layer < 0 or cut_layer >= total_layers - 1:
            raise ValueError(
                f"cut_layer {cut_layer} must be between 0 and {total_layers - 2} for a 2-tier split."
            )
        return cls(
            total_layers=total_layers,
            user_device=(0, cut_layer),
            edge_a=(cut_layer + 1, total_layers - 1),
            edge_b=None,
        )

    @classmethod
    def three_tier(cls, total_layers: int, cut1: int, cut2: int) -> PartitionPlan:
        """
        Create a 3-tier plan:
        - User Device: [0, cut1]
        - Edge A: [cut1 + 1, cut2]
        - Edge B: [cut2 + 1, total_layers - 1]
        """
        if cut1 < 0 or cut2 <= cut1 or cut2 >= total_layers - 1:
            raise ValueError(
                f"Invalid cuts ({cut1}, {cut2}) for 3-tier split with {total_layers} layers. "
                f"Must satisfy 0 <= cut1 < cut2 < {total_layers - 1}."
            )
        return cls(
            total_layers=total_layers,
            user_device=(0, cut1),
            edge_a=(cut1 + 1, cut2),
            edge_b=(cut2 + 1, total_layers - 1),
        )
