"""
Pluggable Hardware Telemetry Provider interface for Module 11.

Defines a common, hardware-agnostic contract for physical resource inspection
across CPU-only, CUDA-accelerated, and simulated execution environments.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.hardware.types import (
    DeviceDescriptor,
    DeviceType,
    HardwareComputeTelemetry,
    HardwareMemoryTelemetry,
    HardwareSnapshot,
    TelemetryCapabilities,
)


class HardwareTelemetryProvider(ABC):
    """
    Abstract interface for physical hardware resource discovery and monitoring.

    Design constraints:
    - Never assume CUDA exists.
    - Never substitute Host RAM for VRAM.
    - Unsupported GPU metrics must return MeasurementRecord.unavailable(), never zero.
    - Hot-path calls must avoid expensive synchronization (e.g. torch.cuda.synchronize()).
    """

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """Primary device architecture handled by this provider."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the underlying hardware/driver is present and accessible."""
        pass

    @abstractmethod
    def discover_devices(self) -> List[DeviceDescriptor]:
        """Enumerate and describe all available execution devices of this type."""
        pass

    @abstractmethod
    def get_memory_telemetry(self, device_id: Optional[int] = None) -> HardwareMemoryTelemetry:
        """
        Sample current physical memory allocations.

        Args:
            device_id: Optional zero-based device index.

        Returns:
            HardwareMemoryTelemetry with explicit provenance on every field.
        """
        pass

    @abstractmethod
    def get_compute_telemetry(self, device_id: Optional[int] = None) -> HardwareComputeTelemetry:
        """
        Sample current compute utilization and processor frequency.

        Args:
            device_id: Optional zero-based device index.

        Returns:
            HardwareComputeTelemetry with explicit provenance on every field.
        """
        pass

    @abstractmethod
    def get_telemetry_capabilities(self, device_id: Optional[int] = None) -> TelemetryCapabilities:
        """Return the set of metric capabilities reliably supported on this hardware."""
        pass

    def collect_snapshot(self, device_id: Optional[int] = None) -> HardwareSnapshot:
        """
        Convenience method to collect unified memory and compute telemetry snapshot.
        """
        start = time.perf_counter()
        mem = self.get_memory_telemetry(device_id)
        comp = self.get_compute_telemetry(device_id)
        duration_ms = (time.perf_counter() - start) * 1000.0

        devices = self.discover_devices()
        dev_name = "Unknown"
        dev_idx = device_id
        if devices:
            matched = next((d for d in devices if d.device_index == (device_id or 0)), devices[0])
            dev_name = matched.name
            dev_idx = matched.device_index

        return HardwareSnapshot(
            device_type=self.device_type,
            device_index=dev_idx,
            device_name=dev_name,
            memory=mem,
            compute=comp,
            timestamp=time.monotonic(),
            collection_duration_ms=duration_ms,
        )
