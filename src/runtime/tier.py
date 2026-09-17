"""
Abstractions for execution tiers in distributed split inference.

Represents the physical or logical tiers (e.g. User Device, Edge Node A, Edge Node B)
participating in autoregressive model partition execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class TierId(str, Enum):
    """Unique identifier for system tiers."""
    USER_DEVICE = "user_device"
    EDGE_A = "edge_a"
    EDGE_B = "edge_b"

    def __str__(self) -> str:
        return self.value


@dataclass
class Tier:
    """
    Represents an execution tier in the distributed hierarchy.

    Attributes:
        tier_id: Enum identifier for the tier.
        name: Human-readable name.
        device: PyTorch device string (e.g., 'cpu', 'cuda:0').
        layer_range: Optional tuple of (start_layer, end_layer) inclusive.
        metadata: Generic tier runtime attributes (e.g., node IP, hardware specs).
    """
    tier_id: TierId
    name: str
    device: str = "cpu"
    layer_range: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer_range is not None:
            start, end = self.layer_range
            if start < 0 or end < start:
                raise ValueError(
                    f"Invalid layer_range {self.layer_range} for tier '{self.tier_id}'. "
                    f"Start index must be >= 0 and end index must be >= start."
                )

    @property
    def is_active(self) -> bool:
        """Returns True if the tier has layers assigned to execute."""
        return self.layer_range is not None

    @property
    def layer_count(self) -> int:
        """Returns the number of assigned layers (0 if inactive)."""
        if self.layer_range is None:
            return 0
        return self.layer_range[1] - self.layer_range[0] + 1

    @property
    def start_layer(self) -> Optional[int]:
        """Inclusive starting layer index."""
        return self.layer_range[0] if self.layer_range else None

    @property
    def end_layer(self) -> Optional[int]:
        """Inclusive ending layer index."""
        return self.layer_range[1] if self.layer_range else None

    def contains_layer(self, layer_idx: int) -> bool:
        """Check if this tier is responsible for executing the specified layer index."""
        if self.layer_range is None:
            return False
        return self.layer_range[0] <= layer_idx <= self.layer_range[1]

    def to_dict(self) -> Dict[str, Any]:
        """Export tier attributes to dictionary."""
        return {
            "tier_id": self.tier_id.value,
            "name": self.name,
            "device": self.device,
            "layer_range": list(self.layer_range) if self.layer_range else None,
            "layer_count": self.layer_count,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }
