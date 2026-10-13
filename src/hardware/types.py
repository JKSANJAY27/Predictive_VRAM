"""
Hardware and Measurement types for Module 11: Real-Hardware Integration & 6G-Oriented Validation.

Preserves strict data provenance and separates measured, estimated, emulated,
and unavailable observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.runtime.tier import TierId
from src.telemetry.types import DataSource, TaggedValue


# ---------------------------------------------------------------------------
# Device and Telemetry Enums
# ---------------------------------------------------------------------------

class DeviceType(str, Enum):
    """Execution hardware architecture."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    MOCK_CUDA = "mock_cuda"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Measurement Record with Rich Provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasurementRecord:
    """
    Measurement record with explicit value, provenance, method, and uncertainty.

    Never silently converts unavailable or estimated metrics to zero or measured.
    """
    value: Optional[float]
    source: DataSource
    unit: str = ""
    timestamp: float = field(default_factory=time.monotonic)
    method: str = "direct_query"
    valid: bool = True
    error: Optional[str] = None
    uncertainty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "method": self.method,
            "valid": self.valid,
            "error": self.error,
            "uncertainty": self.uncertainty,
            "metadata": self.metadata,
        }

    def to_tagged_value(self) -> TaggedValue:
        """Compatibility bridge to existing Module 2/3 TaggedValue."""
        return TaggedValue(
            value=self.value,
            source=self.source,
            unit=self.unit,
        )

    @classmethod
    def measured(
        cls,
        value: float,
        unit: str = "",
        method: str = "direct_query",
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MeasurementRecord:
        return cls(
            value=float(value),
            source=DataSource.MEASURED,
            unit=unit,
            timestamp=timestamp if timestamp is not None else time.monotonic(),
            method=method,
            valid=True,
            metadata=metadata or {},
        )

    @classmethod
    def estimated(
        cls,
        value: float,
        unit: str = "",
        method: str = "analytical_model",
        timestamp: Optional[float] = None,
        uncertainty: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MeasurementRecord:
        return cls(
            value=float(value),
            source=DataSource.ESTIMATED,
            unit=unit,
            timestamp=timestamp if timestamp is not None else time.monotonic(),
            method=method,
            valid=True,
            uncertainty=uncertainty,
            metadata=metadata or {},
        )

    @classmethod
    def emulated(
        cls,
        value: float,
        unit: str = "",
        method: str = "synthetic_profile",
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MeasurementRecord:
        return cls(
            value=float(value),
            source=DataSource.EMULATED,
            unit=unit,
            timestamp=timestamp if timestamp is not None else time.monotonic(),
            method=method,
            valid=True,
            metadata=metadata or {},
        )

    @classmethod
    def unavailable(
        cls,
        unit: str = "",
        error: str = "Metric unavailable on this hardware platform",
        timestamp: Optional[float] = None,
        method: str = "unavailable",
    ) -> MeasurementRecord:
        return cls(
            value=None,
            source=DataSource.UNAVAILABLE,
            unit=unit,
            timestamp=timestamp if timestamp is not None else time.monotonic(),
            method=method,
            valid=False,
            error=error,
        )


# ---------------------------------------------------------------------------
# Hardware Telemetry Snapshots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareMemoryTelemetry:
    """
    Physical and device memory observations.

    Guarantees:
    - Host RAM is never substituted for VRAM.
    - If CUDA is absent, VRAM fields are marked UNAVAILABLE.
    - Explicit separation between total, allocated, reserved, and free memory.
    """
    total_vram_bytes: MeasurementRecord
    allocated_vram_bytes: MeasurementRecord
    reserved_vram_bytes: MeasurementRecord
    free_vram_bytes: MeasurementRecord

    host_ram_total_bytes: MeasurementRecord
    host_ram_used_bytes: MeasurementRecord
    host_ram_available_bytes: MeasurementRecord
    process_rss_bytes: MeasurementRecord

    timestamp: float = field(default_factory=time.monotonic)
    collection_duration_ms: float = 0.0
    errors: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_vram_bytes": self.total_vram_bytes.to_dict(),
            "allocated_vram_bytes": self.allocated_vram_bytes.to_dict(),
            "reserved_vram_bytes": self.reserved_vram_bytes.to_dict(),
            "free_vram_bytes": self.free_vram_bytes.to_dict(),
            "host_ram_total_bytes": self.host_ram_total_bytes.to_dict(),
            "host_ram_used_bytes": self.host_ram_used_bytes.to_dict(),
            "host_ram_available_bytes": self.host_ram_available_bytes.to_dict(),
            "process_rss_bytes": self.process_rss_bytes.to_dict(),
            "timestamp": self.timestamp,
            "collection_duration_ms": self.collection_duration_ms,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class HardwareComputeTelemetry:
    """
    Physical processor compute metrics.

    Guarantees:
    - Unsupported GPU utilization is tagged UNAVAILABLE, never reported as 0.0.
    """
    cpu_utilization_pct: MeasurementRecord
    gpu_utilization_pct: MeasurementRecord
    gpu_memory_utilization_pct: MeasurementRecord
    cpu_freq_mhz: MeasurementRecord
    cpu_count_logical: MeasurementRecord
    cpu_count_physical: MeasurementRecord

    timestamp: float = field(default_factory=time.monotonic)
    collection_duration_ms: float = 0.0
    errors: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_utilization_pct": self.cpu_utilization_pct.to_dict(),
            "gpu_utilization_pct": self.gpu_utilization_pct.to_dict(),
            "gpu_memory_utilization_pct": self.gpu_memory_utilization_pct.to_dict(),
            "cpu_freq_mhz": self.cpu_freq_mhz.to_dict(),
            "cpu_count_logical": self.cpu_count_logical.to_dict(),
            "cpu_count_physical": self.cpu_count_physical.to_dict(),
            "timestamp": self.timestamp,
            "collection_duration_ms": self.collection_duration_ms,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class HardwareSnapshot:
    """Unified hardware snapshot combining memory, compute, and device metadata."""
    device_type: DeviceType
    device_index: Optional[int]
    device_name: str
    memory: HardwareMemoryTelemetry
    compute: HardwareComputeTelemetry
    timestamp: float = field(default_factory=time.monotonic)
    collection_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type.value,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "memory": self.memory.to_dict(),
            "compute": self.compute.to_dict(),
            "timestamp": self.timestamp,
            "collection_duration_ms": self.collection_duration_ms,
        }


# ---------------------------------------------------------------------------
# Device Descriptors and Endpoint Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceDescriptor:
    """Descriptor for a discovered execution device."""
    device_type: DeviceType
    device_index: int
    name: str
    total_memory_bytes: Optional[int]
    compute_capability: Optional[Tuple[int, int]] = None
    is_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type.value,
            "device_index": self.device_index,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
            "compute_capability": list(self.compute_capability) if self.compute_capability else None,
            "is_available": self.is_available,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TelemetryCapabilities:
    """Flags indicating which telemetry capabilities are reliably supported."""
    has_host_ram: bool = True
    has_cpu_util: bool = True
    has_vram_total: bool = False
    has_vram_allocated: bool = False
    has_vram_reserved: bool = False
    has_vram_free: bool = False
    has_gpu_util: bool = False
    has_nvml: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "has_host_ram": self.has_host_ram,
            "has_cpu_util": self.has_cpu_util,
            "has_vram_total": self.has_vram_total,
            "has_vram_allocated": self.has_vram_allocated,
            "has_vram_reserved": self.has_vram_reserved,
            "has_vram_free": self.has_vram_free,
            "has_gpu_util": self.has_gpu_util,
            "has_nvml": self.has_nvml,
        }


@dataclass(frozen=True)
class EndpointMetadata:
    """Network connection metadata for an execution tier."""
    host: str = "127.0.0.1"
    port: int = 8000
    interface_name: Optional[str] = None
    auth_enabled: bool = False
    max_payload_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB
    timeout_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "interface_name": self.interface_name,
            "auth_enabled": self.auth_enabled,
            "max_payload_bytes": self.max_payload_bytes,
            "timeout_seconds": self.timeout_seconds,
        }
