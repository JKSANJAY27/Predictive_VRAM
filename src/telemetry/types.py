"""
Telemetry data structures for the Predictive VRAM split inference system.

Design principles:
- Every numeric/string field that represents a system measurement carries an explicit
  DataSource tag so consumers (e.g., the prediction engine) know the provenance and
  reliability of each datum.
- All snapshot dataclasses are frozen (immutable) after creation.
- No GPU/CUDA calls appear anywhere in this module; VRAM fields are always tagged
  DataSource.UNAVAILABLE on this development machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Source Tag
# ---------------------------------------------------------------------------

class DataSource(str, Enum):
    """
    Provenance tag for every telemetry field.

    MEASURED   — Directly observed from the OS/runtime (psutil, torch, etc.).
    ESTIMATED  — Derived analytically from model config or tensor shapes.
    UNAVAILABLE — Hardware or software required for measurement is absent
                  (e.g., no CUDA-capable GPU on this development machine).
    EMULATED   — Externally injected for experiment / simulation purposes;
                 does NOT reflect real physical conditions.
    """
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    EMULATED = "emulated"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Tagged field helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaggedValue:
    """
    A scalar value paired with its DataSource provenance tag.

    Attributes:
        value: The numeric or string measurement (None if unavailable).
        source: DataSource indicating provenance.
        unit: Optional human-readable unit string (e.g., 'MB', '%', 'ms').
    """
    value: Optional[float]
    source: DataSource
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source.value,
            "unit": self.unit,
        }

    @classmethod
    def unavailable(cls, unit: str = "") -> "TaggedValue":
        """Convenience constructor for absent measurements."""
        return cls(value=None, source=DataSource.UNAVAILABLE, unit=unit)

    @classmethod
    def measured(cls, value: float, unit: str = "") -> "TaggedValue":
        """Convenience constructor for directly measured values."""
        return cls(value=value, source=DataSource.MEASURED, unit=unit)

    @classmethod
    def estimated(cls, value: float, unit: str = "") -> "TaggedValue":
        """Convenience constructor for analytically derived values."""
        return cls(value=value, source=DataSource.ESTIMATED, unit=unit)

    @classmethod
    def emulated(cls, value: float, unit: str = "") -> "TaggedValue":
        """Convenience constructor for externally injected values."""
        return cls(value=value, source=DataSource.EMULATED, unit=unit)


# ---------------------------------------------------------------------------
# Memory Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemorySnapshot:
    """
    System memory state at a single point in time.

    RAM fields are MEASURED via psutil on supported platforms.
    VRAM fields are UNAVAILABLE on this development machine (no CUDA GPU).
    """
    # --- RAM (system memory) ---
    ram_total_mb: TaggedValue          # Total installed RAM
    ram_used_mb: TaggedValue           # Currently used RAM
    ram_available_mb: TaggedValue      # Available (free + reclaimable) RAM
    ram_percent: TaggedValue           # Used RAM as percent of total

    # --- Swap ---
    swap_used_mb: TaggedValue          # Swap / page-file used
    swap_total_mb: TaggedValue         # Total swap

    # --- VRAM (GPU memory) ---
    # Always UNAVAILABLE on CPU-only machines.
    vram_total_mb: TaggedValue         # GPU memory total
    vram_used_mb: TaggedValue          # GPU memory in use
    vram_free_mb: TaggedValue          # GPU memory free

    # --- Metadata ---
    timestamp: float = field(default_factory=time.monotonic)
    process_ram_mb: TaggedValue = field(
        default_factory=lambda: TaggedValue.unavailable("MB")
    )  # This process's RSS; populated if psutil available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ram_total_mb": self.ram_total_mb.to_dict(),
            "ram_used_mb": self.ram_used_mb.to_dict(),
            "ram_available_mb": self.ram_available_mb.to_dict(),
            "ram_percent": self.ram_percent.to_dict(),
            "swap_used_mb": self.swap_used_mb.to_dict(),
            "swap_total_mb": self.swap_total_mb.to_dict(),
            "vram_total_mb": self.vram_total_mb.to_dict(),
            "vram_used_mb": self.vram_used_mb.to_dict(),
            "vram_free_mb": self.vram_free_mb.to_dict(),
            "process_ram_mb": self.process_ram_mb.to_dict(),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# CPU Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPUSnapshot:
    """
    CPU utilisation and topology at a single point in time.

    All fields are MEASURED or ESTIMATED from OS APIs.
    """
    cpu_percent_overall: TaggedValue   # System-wide CPU utilisation (%)
    cpu_count_logical: TaggedValue     # Logical core count
    cpu_count_physical: TaggedValue    # Physical core count
    cpu_freq_mhz: TaggedValue          # Current CPU frequency (MHz)
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent_overall": self.cpu_percent_overall.to_dict(),
            "cpu_count_logical": self.cpu_count_logical.to_dict(),
            "cpu_count_physical": self.cpu_count_physical.to_dict(),
            "cpu_freq_mhz": self.cpu_freq_mhz.to_dict(),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Activation Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActivationSnapshot:
    """
    Metadata about an inter-tier activation tensor transfer.

    Shape and byte size are MEASURED directly from the tensor.
    Transfer cost estimates (latency) are ESTIMATED from network emulation state.
    """
    tensor_shape: Tuple[int, ...]       # e.g., (1, seq_len, hidden_size)
    dtype: str                          # e.g., 'torch.float32'
    byte_size: TaggedValue              # Total bytes in the tensor
    num_elements: TaggedValue           # Total element count
    estimated_transfer_ms: TaggedValue  # Estimated transfer latency (EMULATED/ESTIMATED)
    source_tier: str                    # e.g., 'user_device'
    destination_tier: str              # e.g., 'edge_a'
    step: Optional[int] = None          # Autoregressive step index
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_shape": list(self.tensor_shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size.to_dict(),
            "num_elements": self.num_elements.to_dict(),
            "estimated_transfer_ms": self.estimated_transfer_ms.to_dict(),
            "source_tier": self.source_tier,
            "destination_tier": self.destination_tier,
            "step": self.step,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# KV-Cache Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KVCacheSnapshot:
    """
    KV-cache memory state after processing a token step.

    All byte counts are MEASURED from DynamicCache tensor sizes.
    Growth delta is ESTIMATED by comparison with the previous snapshot.
    """
    num_cached_layers: TaggedValue          # Number of layers with cached KV
    total_kv_bytes: TaggedValue             # Total bytes across all layers
    total_kv_mb: TaggedValue                # Same, in MB
    kv_growth_bytes_since_last: TaggedValue # Bytes added since last snapshot (ESTIMATED)
    per_layer_bytes: List[int]              # Per-layer byte count list (MEASURED)
    sample_layer_shape: Tuple[int, ...]     # Shape of key tensor from layer 0 (if any)
    step: Optional[int] = None
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_cached_layers": self.num_cached_layers.to_dict(),
            "total_kv_bytes": self.total_kv_bytes.to_dict(),
            "total_kv_mb": self.total_kv_mb.to_dict(),
            "kv_growth_bytes_since_last": self.kv_growth_bytes_since_last.to_dict(),
            "per_layer_bytes": self.per_layer_bytes,
            "sample_layer_shape": list(self.sample_layer_shape),
            "step": self.step,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Network Condition Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkConditionSnapshot:
    """
    Network conditions between tiers at a single point in time.

    ALL fields are EMULATED on this development machine — there are no real
    edge nodes, no real network probes, and no real RTT measurements.
    The emulation state is injected externally via NetworkEmulationState.
    """
    rtt_ms: TaggedValue                 # Round-trip time (always EMULATED)
    bandwidth_mbps: TaggedValue         # Uplink bandwidth (always EMULATED)
    packet_loss_rate: TaggedValue       # Loss rate 0.0–1.0 (always EMULATED)
    jitter_ms: TaggedValue              # RTT jitter std-dev (always EMULATED)
    scenario_label: str = "baseline"    # Human-readable scenario name
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rtt_ms": self.rtt_ms.to_dict(),
            "bandwidth_mbps": self.bandwidth_mbps.to_dict(),
            "packet_loss_rate": self.packet_loss_rate.to_dict(),
            "jitter_ms": self.jitter_ms.to_dict(),
            "scenario_label": self.scenario_label,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Composite Telemetry Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelemetrySnapshot:
    """
    A composite telemetry record capturing all system conditions at one
    autoregressive generation step.

    This is the unit of storage in TelemetryBuffer and the unit of
    consumption by the prediction engine (Module 3).

    Fields may be None if collection was skipped or the relevant subsystem
    was not active during that step.
    """
    step: int                                        # Autoregressive step index
    timestamp: float                                 # Monotonic clock at capture time
    memory: Optional[MemorySnapshot] = None
    cpu: Optional[CPUSnapshot] = None
    kv_cache: Optional[KVCacheSnapshot] = None
    network: Optional[NetworkConditionSnapshot] = None
    activations: List[ActivationSnapshot] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "memory": self.memory.to_dict() if self.memory else None,
            "cpu": self.cpu.to_dict() if self.cpu else None,
            "kv_cache": self.kv_cache.to_dict() if self.kv_cache else None,
            "network": self.network.to_dict() if self.network else None,
            "activations": [a.to_dict() for a in self.activations],
            "metadata": self.metadata,
        }
