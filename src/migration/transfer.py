"""
Transfer providers for physical and emulated migration in Module 8.

Separates physical tensor movement from steady-state token activations,
providing pluggable local and network-emulated transports.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.runtime.tier import TierId
from src.telemetry.types import DataSource


class MigrationTransferProvider(ABC):
    """
    Abstract interface for migrating state and parameter tensors between tiers.
    """

    @abstractmethod
    def transfer_tensor(
        self,
        tensor: torch.Tensor,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
    ) -> Tuple[torch.Tensor, int, float, DataSource]:
        """
        Transfer a single tensor to the target device and return (tensor, bytes, duration_ms, provenance).
        """
        pass

    @abstractmethod
    def transfer_module(
        self,
        module: nn.Module,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
    ) -> Tuple[nn.Module, int, float, DataSource]:
        """
        Transfer an entire neural network module (all parameters and buffers) to target device.
        """
        pass


class LocalTensorTransfer(MigrationTransferProvider):
    """
    Direct in-process device transfer (e.g. CPU <-> GPU or local memory clone).
    """

    def transfer_tensor(
        self,
        tensor: torch.Tensor,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
    ) -> Tuple[torch.Tensor, int, float, DataSource]:
        start = time.perf_counter()
        bytes_transferred = tensor.numel() * tensor.element_size()
        if str(tensor.device) != target_device:
            moved = tensor.to(target_device)
        else:
            moved = tensor.clone()
        duration_ms = (time.perf_counter() - start) * 1000.0
        return moved, bytes_transferred, duration_ms, DataSource.MEASURED

    def transfer_module(
        self,
        module: nn.Module,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
    ) -> Tuple[nn.Module, int, float, DataSource]:
        start = time.perf_counter()
        total_bytes = 0
        for p in module.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in module.buffers():
            total_bytes += b.numel() * b.element_size()

        module.to(target_device)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return module, total_bytes, duration_ms, DataSource.MEASURED


class EmulatedNetworkTransfer(MigrationTransferProvider):
    """
    Network-aware transfer provider calculating simulated transmission duration
    from configured bandwidth and latency profiles.
    """

    def __init__(
        self,
        bandwidth_mbps: float = 50.0,
        latency_ms: float = 10.0,
        fast_mode: bool = True,
    ) -> None:
        self.bandwidth_mbps = max(0.1, bandwidth_mbps)
        self.latency_ms = max(0.0, latency_ms)
        self.fast_mode = fast_mode
        self._local = LocalTensorTransfer()

    def _calc_emulated_duration_ms(self, bytes_transferred: int) -> float:
        bits = bytes_transferred * 8
        transfer_sec = bits / (self.bandwidth_mbps * 1e6)
        transfer_ms = (transfer_sec * 1000.0) + self.latency_ms
        return transfer_ms

    def transfer_tensor(
        self,
        tensor: torch.Tensor,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
    ) -> Tuple[torch.Tensor, int, float, DataSource]:
        moved, bytes_count, _, _ = self._local.transfer_tensor(
            tensor, source_tier, destination_tier, target_device
        )
        sim_ms = self._calc_emulated_duration_ms(bytes_count)
        if not self.fast_mode and sim_ms > 0:
            time.sleep(min(0.05, sim_ms / 1000.0))  # Cap physical sleep in testing
        return moved, bytes_count, sim_ms, DataSource.EMULATED

    def transfer_module(
        self,
        module: nn.Module,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
    ) -> Tuple[nn.Module, int, float, DataSource]:
        moved, bytes_count, _, _ = self._local.transfer_module(
            module, source_tier, destination_tier, target_device
        )
        sim_ms = self._calc_emulated_duration_ms(bytes_count)
        if not self.fast_mode and sim_ms > 0:
            time.sleep(min(0.05, sim_ms / 1000.0))
        return moved, bytes_count, sim_ms, DataSource.EMULATED
