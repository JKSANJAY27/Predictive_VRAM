"""
TelemetryAgent and runtime adapters for Module 9.

Implements the unified TelemetryAgent that orchestrates all Module 2 collectors
to generate consistent, source-tagged TelemetrySnapshots at every control opportunity.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.telemetry.collectors import (
    ActivationCollector,
    CPUCollector,
    KVCacheCollector,
    MemoryCollector,
    NetworkConditionCollector,
)
from src.telemetry.network_emulation import NetworkEmulationState
from src.telemetry.types import (
    ActivationSnapshot,
    DataSource,
    NetworkConditionSnapshot,
    TaggedValue,
    TelemetrySnapshot,
)


class TelemetryAgent:
    """
    Unified agent responsible for orchestrating telemetry collection across all subsystems:
    - RAM and system memory
    - CPU utilization
    - KV-cache state and memory footprint
    - Inter-tier network conditions (emulated)
    - Activation tensor transfer metadata

    Supports synthetic fault injection and simulated scenario streams for testing.
    """

    def __init__(
        self,
        network_state: Optional[NetworkEmulationState] = None,
        memory_collector: Optional[MemoryCollector] = None,
        cpu_collector: Optional[CPUCollector] = None,
        kv_collector: Optional[KVCacheCollector] = None,
        network_collector: Optional[NetworkConditionCollector] = None,
        activation_collector: Optional[ActivationCollector] = None,
    ) -> None:
        self.network_state = network_state or NetworkEmulationState()
        self.memory_collector = memory_collector or MemoryCollector()
        self.cpu_collector = cpu_collector or CPUCollector()
        self.kv_collector = kv_collector or KVCacheCollector()
        self.network_collector = network_collector or NetworkConditionCollector(self.network_state)
        self.activation_collector = activation_collector or ActivationCollector()

        # Fault injection toggles for robustness testing
        self.inject_failure: bool = False
        self.inject_network_dropout: bool = False
        self.inject_memory_dropout: bool = False

    def collect(
        self,
        step: int,
        kv_cache: Optional[Any] = None,
        activations: Optional[List[ActivationSnapshot]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetrySnapshot:
        """
        Sample all active collectors and assemble a single TelemetrySnapshot.

        Raises:
            RuntimeError: If inject_failure is True.
        """
        if self.inject_failure:
            raise RuntimeError("Injected telemetry agent failure triggered.")

        now = time.monotonic()

        # Memory snapshot
        if self.inject_memory_dropout:
            mem_snap = None
        else:
            mem_snap = self.memory_collector.collect()

        # CPU snapshot
        cpu_snap = self.cpu_collector.collect()

        # KV-cache snapshot
        kv_snap = None
        if kv_cache is not None:
            kv_snap = self.kv_collector.collect(kv_cache, step=step)

        # Network snapshot
        if self.inject_network_dropout:
            net_snap = NetworkConditionSnapshot(
                rtt_ms=TaggedValue.unavailable("ms"),
                bandwidth_mbps=TaggedValue.unavailable("Mbps"),
                packet_loss_rate=TaggedValue.unavailable("ratio"),
                jitter_ms=TaggedValue.unavailable("ms"),
                scenario_label="dropout",
                timestamp=now,
            )
        else:
            net_snap = self.network_collector.collect()

        act_list = list(activations) if activations else []
        meta = dict(metadata) if metadata else {}

        return TelemetrySnapshot(
            step=step,
            timestamp=now,
            memory=mem_snap,
            cpu=cpu_snap,
            kv_cache=kv_snap,
            network=net_snap,
            activations=act_list,
            metadata=meta,
        )

    def set_network_conditions(
        self,
        bandwidth_mbps: float,
        rtt_ms: float = 5.0,
        packet_loss: float = 0.0,
        jitter_ms: float = 0.0,
    ) -> None:
        """Dynamically update emulated network parameters."""
        self.network_state.bandwidth_mbps = float(bandwidth_mbps)
        self.network_state.rtt_ms = float(rtt_ms)
        self.network_state.packet_loss_rate = float(packet_loss)
        self.network_state.jitter_ms = float(jitter_ms)
