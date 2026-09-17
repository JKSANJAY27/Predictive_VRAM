"""
telemetry — Runtime Telemetry Layer for Predictive VRAM split inference.

Provides honest, source-tagged system snapshots for:
- Memory (RAM/VRAM)
- CPU utilisation
- Activation tensor sizes (inter-tier transfers)
- KV-cache growth
- Emulated network conditions

Every metric carries a DataSource tag indicating whether it was
MEASURED, ESTIMATED, UNAVAILABLE, or EMULATED.
"""

from src.telemetry.types import (
    DataSource,
    MemorySnapshot,
    CPUSnapshot,
    ActivationSnapshot,
    KVCacheSnapshot,
    NetworkConditionSnapshot,
    TelemetrySnapshot,
)
from src.telemetry.collectors import (
    MemoryCollector,
    CPUCollector,
    ActivationCollector,
    KVCacheCollector,
    NetworkConditionCollector,
)
from src.telemetry.buffer import TelemetryBuffer
from src.telemetry.network_emulation import NetworkEmulationState

__all__ = [
    # Types
    "DataSource",
    "MemorySnapshot",
    "CPUSnapshot",
    "ActivationSnapshot",
    "KVCacheSnapshot",
    "NetworkConditionSnapshot",
    "TelemetrySnapshot",
    # Collectors
    "MemoryCollector",
    "CPUCollector",
    "ActivationCollector",
    "KVCacheCollector",
    "NetworkConditionCollector",
    # Buffer
    "TelemetryBuffer",
    # Network emulation
    "NetworkEmulationState",
]
