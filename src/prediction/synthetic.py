"""
Controlled synthetic trace generator for time-series evaluation.

Generates explicit, repeatable test sequences across 9 canonical system profiles.
ALL generated quantities are explicitly tagged DataSource.EMULATED.
"""

from __future__ import annotations

import math
from typing import List, Optional

from src.state.types import (
    ActivationState,
    ComputeState,
    EnergyState,
    InferenceState,
    MemoryState,
    NetworkState,
    RuntimeState,
)
from src.telemetry.types import DataSource, TaggedValue

SCENARIO_NAMES = (
    "stable_memory",
    "linear_memory_growth",
    "kv_cache_driven_memory_growth",
    "stable_network",
    "bandwidth_degradation",
    "bandwidth_recovery",
    "oscillating_bandwidth",
    "latency_spike",
    "combined_degradation",
)


class SyntheticTraceGenerator:
    """
    Generates synthetic RuntimeState traces with explicit EMULATED provenance.
    """

    @classmethod
    def generate_trace(
        cls,
        scenario: str = "stable_network",
        num_steps: int = 30,
        dt: float = 0.5,
        start_timestamp: float = 100.0,
        emulate_vram: bool = False,
    ) -> List[RuntimeState]:
        """
        Generate a list of sequential RuntimeState objects following a target scenario profile.

        Args:
            scenario: Name of profile (one of SCENARIO_NAMES).
            num_steps: Number of discrete steps.
            dt: Time increment per step in seconds.
            start_timestamp: Starting clock time.
            emulate_vram: If True, populates VRAM with EMULATED values for test purposes.
                          If False (default), VRAM remains strictly UNAVAILABLE.
        """
        if scenario not in SCENARIO_NAMES:
            raise ValueError(f"Unknown scenario '{scenario}'. Supported: {SCENARIO_NAMES}")

        states: List[RuntimeState] = []

        # Baseline baseline state values
        base_bw = 100.0
        base_lat = 20.0
        base_loss = 0.005
        base_jitter = 2.0
        base_ram = 2048.0
        base_vram_free = 4096.0
        base_vram_alloc = 2048.0
        base_kv = 16384.0

        for i in range(num_steps):
            t = start_timestamp + i * dt
            progress = i / max(1, num_steps - 1)

            bw = base_bw
            lat = base_lat
            loss = base_loss
            jitter = base_jitter
            ram = base_ram
            vram_free = base_vram_free
            vram_alloc = base_vram_alloc
            kv = base_kv + i * 2048.0

            # Apply scenario dynamics
            if scenario == "stable_memory":
                ram = base_ram
                vram_free = base_vram_free
            elif scenario == "linear_memory_growth":
                ram = base_ram + i * 50.0  # +50 MB per step
                vram_free = max(100.0, base_vram_free - i * 50.0)
            elif scenario == "kv_cache_driven_memory_growth":
                # KV cache expands with context length
                kv = base_kv + (i ** 1.2) * 4096.0
                ram = base_ram + (kv / 1024.0 / 1024.0) * 10.0
                vram_free = max(100.0, base_vram_free - (kv / 1024.0 / 1024.0) * 10.0)
            elif scenario == "stable_network":
                bw = base_bw
                lat = base_lat
            elif scenario == "bandwidth_degradation":
                # Drops from 100 Mbps down to 10 Mbps
                bw = max(5.0, base_bw * (1.0 - 0.9 * progress))
                lat = base_lat + 40.0 * progress
            elif scenario == "bandwidth_recovery":
                # Recovers from 10 Mbps up to 100 Mbps
                bw = 10.0 + 90.0 * progress
                lat = max(15.0, 60.0 - 40.0 * progress)
            elif scenario == "oscillating_bandwidth":
                # Sinusoidal oscillation between 20 and 80 Mbps
                bw = 50.0 + 30.0 * math.sin(2.0 * math.pi * progress * 3.0)
                lat = 25.0 + 10.0 * math.cos(2.0 * math.pi * progress * 3.0)
            elif scenario == "latency_spike":
                # Sudden jump at 50% through the trace
                if 0.4 <= progress <= 0.7:
                    lat = 180.0
                    loss = 0.05
                else:
                    lat = base_lat
                    loss = base_loss
            elif scenario == "combined_degradation":
                bw = max(5.0, base_bw * (1.0 - 0.85 * progress))
                lat = base_lat + 60.0 * progress
                ram = base_ram + i * 80.0
                vram_free = max(50.0, base_vram_free - i * 80.0)

            # Build memory state
            if emulate_vram:
                mem_state = MemoryState(
                    vram_allocated_mb=TaggedValue.emulated(vram_alloc, "MB"),
                    vram_free_mb=TaggedValue.emulated(vram_free, "MB"),
                    ram_used_mb=TaggedValue.emulated(ram, "MB"),
                    ram_available_mb=TaggedValue.emulated(max(100.0, 8192.0 - ram), "MB"),
                )
            else:
                mem_state = MemoryState(
                    vram_allocated_mb=TaggedValue.unavailable("MB"),
                    vram_free_mb=TaggedValue.unavailable("MB"),
                    ram_used_mb=TaggedValue.emulated(ram, "MB"),
                    ram_available_mb=TaggedValue.emulated(max(100.0, 8192.0 - ram), "MB"),
                )

            net_state = NetworkState(
                bandwidth_mbps=TaggedValue.emulated(bw, "Mbps"),
                latency_ms=TaggedValue.emulated(lat, "ms"),
                packet_loss=TaggedValue.emulated(loss, "ratio"),
                jitter_ms=TaggedValue.emulated(jitter, "ms"),
            )

            cpu_val = 30.0 + 20.0 * progress
            comp_state = ComputeState(
                gpu_utilization=TaggedValue.unavailable("%"),
                cpu_utilization=TaggedValue.emulated(cpu_val, "%"),
            )

            inf_state = InferenceState(
                kv_cache_bytes=TaggedValue.emulated(kv, "bytes"),
                kv_cache_growth_bytes=TaggedValue.emulated(2048.0, "bytes"),
                generated_tokens=i,
                context_length=10 + i,
                generation_rate=1.0 / dt,
                request_state="decode",
                request_id=f"synthetic-{scenario}",
            )

            act_state = ActivationState(
                latest_activation_bytes=TaggedValue.emulated(4096.0, "bytes"),
                latest_transfer_latency_ms=TaggedValue.emulated(1.5, "ms"),
            )

            energy_state = EnergyState(
                energy_proxy=TaggedValue.emulated(cpu_val * dt, "proxy_units"),
            )

            state = RuntimeState(
                timestamp=t,
                wall_clock=1700000000.0 + t,
                step_index=i,
                network=net_state,
                memory=mem_state,
                compute=comp_state,
                inference=inf_state,
                activation=act_state,
                energy=energy_state,
                metadata={"scenario": scenario, "synthetic": True},
            )
            states.append(state)

        return states
