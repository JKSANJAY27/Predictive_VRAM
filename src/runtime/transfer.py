"""
Transfer management for activation tensors crossing tier boundaries.

Provides an explicit, observable boundary for inter-tier tensor communication.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.runtime.tier import TierId


@dataclass(frozen=True)
class TransferRecord:
    """Detailed metadata for a single tensor transfer across tier boundaries."""
    source_tier: TierId
    destination_tier: TierId
    tensor_shape: Tuple[int, ...]
    dtype: str
    num_elements: int
    byte_size: int
    timestamp: float
    step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tier": self.source_tier.value,
            "destination_tier": self.destination_tier.value,
            "tensor_shape": list(self.tensor_shape),
            "dtype": self.dtype,
            "num_elements": self.num_elements,
            "byte_size": self.byte_size,
            "timestamp": self.timestamp,
            "step": self.step,
        }


class TransferManager:
    """
    Manages activation and state transfers between execution tiers.

    In Module 1, performs direct memory transfer or cross-device movement
    while recording full transfer telemetry (shape, bytes, timing).
    In future modules, this interface will integrate network emulation and latency models.
    """

    def __init__(self) -> None:
        self._history: List[TransferRecord] = []

    def transfer(
        self,
        tensor: torch.Tensor,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Transfer a tensor from source_tier to destination_tier.

        Args:
            tensor: PyTorch tensor to transfer.
            source_tier: Originating tier.
            destination_tier: Receiving tier.
            target_device: PyTorch device target for the receiving tier.
            step: Optional token generation step index for telemetry.

        Returns:
            The transferred tensor located on target_device.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

        num_elements = tensor.numel()
        element_size = tensor.element_size()
        byte_size = num_elements * element_size

        record = TransferRecord(
            source_tier=source_tier,
            destination_tier=destination_tier,
            tensor_shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            num_elements=num_elements,
            byte_size=byte_size,
            timestamp=time.time(),
            step=step,
        )
        self._history.append(record)

        # Move to target device if different
        if str(tensor.device) != target_device:
            return tensor.to(target_device)
        return tensor

    @property
    def total_transfers(self) -> int:
        """Total number of transfers recorded."""
        return len(self._history)

    @property
    def total_bytes(self) -> int:
        """Cumulative bytes transferred across all boundaries."""
        return sum(r.byte_size for r in self._history)

    def get_history(self) -> List[TransferRecord]:
        """Return shallow copy of recorded transfers."""
        return list(self._history)

    def clear_history(self) -> None:
        """Reset transfer telemetry records."""
        self._history.clear()
