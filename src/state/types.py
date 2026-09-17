"""
Unified Runtime State representations for the Predictive VRAM split inference system.

Module 3 converts raw TelemetrySnapshot observations into clean, normalized,
timestamp-ordered runtime state suitable for future prediction and control.

Design principles:
- Provenance tracking (DataSource: MEASURED, ESTIMATED, UNAVAILABLE, EMULATED)
  is preserved on all numeric values via TaggedValue.
- All state dataclasses are frozen (immutable) after creation.
- Values of None indicate absence / unavailability; missing values are NEVER imputed as 0.
- Strict physical domain validation: negative latencies, bandwidths, memory,
  packet loss outside [0, 1], and utilization outside [0, 100] are rejected.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.telemetry.types import DataSource, TaggedValue, TelemetrySnapshot


# ---------------------------------------------------------------------------
# Component State Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkState:
    """Network conditions relevant to inter-tier communication."""
    bandwidth_mbps: TaggedValue
    latency_ms: TaggedValue
    packet_loss: TaggedValue
    jitter_ms: TaggedValue

    def __post_init__(self) -> None:
        if self.bandwidth_mbps.value is not None and self.bandwidth_mbps.source != DataSource.UNAVAILABLE:
            if self.bandwidth_mbps.value < 0:
                raise ValueError(f"bandwidth_mbps cannot be negative: {self.bandwidth_mbps.value}")
        if self.latency_ms.value is not None and self.latency_ms.source != DataSource.UNAVAILABLE:
            if self.latency_ms.value < 0:
                raise ValueError(f"latency_ms cannot be negative: {self.latency_ms.value}")
        if self.packet_loss.value is not None and self.packet_loss.source != DataSource.UNAVAILABLE:
            if not (0.0 <= self.packet_loss.value <= 1.0):
                raise ValueError(f"packet_loss must be in [0.0, 1.0]: {self.packet_loss.value}")
        if self.jitter_ms.value is not None and self.jitter_ms.source != DataSource.UNAVAILABLE:
            if self.jitter_ms.value < 0:
                raise ValueError(f"jitter_ms cannot be negative: {self.jitter_ms.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bandwidth_mbps": self.bandwidth_mbps.to_dict(),
            "latency_ms": self.latency_ms.to_dict(),
            "packet_loss": self.packet_loss.to_dict(),
            "jitter_ms": self.jitter_ms.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NetworkState:
        return cls(
            bandwidth_mbps=TaggedValue(
                value=data["bandwidth_mbps"]["value"],
                source=DataSource(data["bandwidth_mbps"]["source"]),
                unit=data["bandwidth_mbps"].get("unit", "Mbps"),
            ),
            latency_ms=TaggedValue(
                value=data["latency_ms"]["value"],
                source=DataSource(data["latency_ms"]["source"]),
                unit=data["latency_ms"].get("unit", "ms"),
            ),
            packet_loss=TaggedValue(
                value=data["packet_loss"]["value"],
                source=DataSource(data["packet_loss"]["source"]),
                unit=data["packet_loss"].get("unit", "ratio"),
            ),
            jitter_ms=TaggedValue(
                value=data["jitter_ms"]["value"],
                source=DataSource(data["jitter_ms"]["source"]),
                unit=data["jitter_ms"].get("unit", "ms"),
            ),
        )


@dataclass(frozen=True)
class MemoryState:
    """RAM and VRAM memory state."""
    vram_allocated_mb: TaggedValue
    vram_free_mb: TaggedValue
    ram_used_mb: TaggedValue
    ram_available_mb: TaggedValue

    def __post_init__(self) -> None:
        if self.vram_allocated_mb.value is not None and self.vram_allocated_mb.source != DataSource.UNAVAILABLE:
            if self.vram_allocated_mb.value < 0:
                raise ValueError(f"vram_allocated_mb cannot be negative: {self.vram_allocated_mb.value}")
        if self.vram_free_mb.value is not None and self.vram_free_mb.source != DataSource.UNAVAILABLE:
            if self.vram_free_mb.value < 0:
                raise ValueError(f"vram_free_mb cannot be negative: {self.vram_free_mb.value}")
        if self.ram_used_mb.value is not None and self.ram_used_mb.source != DataSource.UNAVAILABLE:
            if self.ram_used_mb.value < 0:
                raise ValueError(f"ram_used_mb cannot be negative: {self.ram_used_mb.value}")
        if self.ram_available_mb.value is not None and self.ram_available_mb.source != DataSource.UNAVAILABLE:
            if self.ram_available_mb.value < 0:
                raise ValueError(f"ram_available_mb cannot be negative: {self.ram_available_mb.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vram_allocated_mb": self.vram_allocated_mb.to_dict(),
            "vram_free_mb": self.vram_free_mb.to_dict(),
            "ram_used_mb": self.ram_used_mb.to_dict(),
            "ram_available_mb": self.ram_available_mb.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryState:
        return cls(
            vram_allocated_mb=TaggedValue(
                value=data["vram_allocated_mb"]["value"],
                source=DataSource(data["vram_allocated_mb"]["source"]),
                unit=data["vram_allocated_mb"].get("unit", "MB"),
            ),
            vram_free_mb=TaggedValue(
                value=data["vram_free_mb"]["value"],
                source=DataSource(data["vram_free_mb"]["source"]),
                unit=data["vram_free_mb"].get("unit", "MB"),
            ),
            ram_used_mb=TaggedValue(
                value=data["ram_used_mb"]["value"],
                source=DataSource(data["ram_used_mb"]["source"]),
                unit=data["ram_used_mb"].get("unit", "MB"),
            ),
            ram_available_mb=TaggedValue(
                value=data["ram_available_mb"]["value"],
                source=DataSource(data["ram_available_mb"]["source"]),
                unit=data["ram_available_mb"].get("unit", "MB"),
            ),
        )


@dataclass(frozen=True)
class ComputeState:
    """CPU and GPU compute utilization."""
    gpu_utilization: TaggedValue
    cpu_utilization: TaggedValue

    def __post_init__(self) -> None:
        if self.gpu_utilization.value is not None and self.gpu_utilization.source != DataSource.UNAVAILABLE:
            if not (0.0 <= self.gpu_utilization.value <= 100.0):
                raise ValueError(f"gpu_utilization must be in [0.0, 100.0]: {self.gpu_utilization.value}")
        if self.cpu_utilization.value is not None and self.cpu_utilization.source != DataSource.UNAVAILABLE:
            if not (0.0 <= self.cpu_utilization.value <= 100.0):
                raise ValueError(f"cpu_utilization must be in [0.0, 100.0]: {self.cpu_utilization.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_utilization": self.gpu_utilization.to_dict(),
            "cpu_utilization": self.cpu_utilization.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComputeState:
        return cls(
            gpu_utilization=TaggedValue(
                value=data["gpu_utilization"]["value"],
                source=DataSource(data["gpu_utilization"]["source"]),
                unit=data["gpu_utilization"].get("unit", "%"),
            ),
            cpu_utilization=TaggedValue(
                value=data["cpu_utilization"]["value"],
                source=DataSource(data["cpu_utilization"]["source"]),
                unit=data["cpu_utilization"].get("unit", "%"),
            ),
        )


@dataclass(frozen=True)
class InferenceState:
    """Autoregressive generation and KV cache state."""
    kv_cache_bytes: TaggedValue
    kv_cache_growth_bytes: TaggedValue
    generated_tokens: int
    context_length: int
    generation_rate: Optional[float] = None
    request_state: str = "decode"
    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kv_cache_bytes.value is not None and self.kv_cache_bytes.source != DataSource.UNAVAILABLE:
            if self.kv_cache_bytes.value < 0:
                raise ValueError(f"kv_cache_bytes cannot be negative: {self.kv_cache_bytes.value}")
        if self.generated_tokens < 0:
            raise ValueError(f"generated_tokens cannot be negative: {self.generated_tokens}")
        if self.context_length < 0:
            raise ValueError(f"context_length cannot be negative: {self.context_length}")
        if self.generation_rate is not None and self.generation_rate < 0:
            raise ValueError(f"generation_rate cannot be negative: {self.generation_rate}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kv_cache_bytes": self.kv_cache_bytes.to_dict(),
            "kv_cache_growth_bytes": self.kv_cache_growth_bytes.to_dict(),
            "generated_tokens": self.generated_tokens,
            "context_length": self.context_length,
            "generation_rate": self.generation_rate,
            "request_state": self.request_state,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InferenceState:
        return cls(
            kv_cache_bytes=TaggedValue(
                value=data["kv_cache_bytes"]["value"],
                source=DataSource(data["kv_cache_bytes"]["source"]),
                unit=data["kv_cache_bytes"].get("unit", "bytes"),
            ),
            kv_cache_growth_bytes=TaggedValue(
                value=data["kv_cache_growth_bytes"]["value"],
                source=DataSource(data["kv_cache_growth_bytes"]["source"]),
                unit=data["kv_cache_growth_bytes"].get("unit", "bytes"),
            ),
            generated_tokens=data["generated_tokens"],
            context_length=data["context_length"],
            generation_rate=data.get("generation_rate"),
            request_state=data.get("request_state", "decode"),
            request_id=data.get("request_id"),
        )


@dataclass(frozen=True)
class ActivationState:
    """Inter-tier tensor activation transfer state."""
    latest_activation_bytes: Optional[TaggedValue] = None
    latest_transfer_latency_ms: Optional[TaggedValue] = None

    def __post_init__(self) -> None:
        if self.latest_activation_bytes is not None and self.latest_activation_bytes.value is not None:
            if self.latest_activation_bytes.source != DataSource.UNAVAILABLE and self.latest_activation_bytes.value < 0:
                raise ValueError(f"latest_activation_bytes cannot be negative: {self.latest_activation_bytes.value}")
        if self.latest_transfer_latency_ms is not None and self.latest_transfer_latency_ms.value is not None:
            if self.latest_transfer_latency_ms.source != DataSource.UNAVAILABLE and self.latest_transfer_latency_ms.value < 0:
                raise ValueError(f"latest_transfer_latency_ms cannot be negative: {self.latest_transfer_latency_ms.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latest_activation_bytes": self.latest_activation_bytes.to_dict() if self.latest_activation_bytes else None,
            "latest_transfer_latency_ms": self.latest_transfer_latency_ms.to_dict() if self.latest_transfer_latency_ms else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActivationState:
        act_bytes = data.get("latest_activation_bytes")
        act_lat = data.get("latest_transfer_latency_ms")
        return cls(
            latest_activation_bytes=TaggedValue(
                value=act_bytes["value"],
                source=DataSource(act_bytes["source"]),
                unit=act_bytes.get("unit", "bytes"),
            ) if act_bytes else None,
            latest_transfer_latency_ms=TaggedValue(
                value=act_lat["value"],
                source=DataSource(act_lat["source"]),
                unit=act_lat.get("unit", "ms"),
            ) if act_lat else None,
        )


@dataclass(frozen=True)
class EnergyState:
    """Energy proxy state (derived analytically or estimated from compute/time)."""
    energy_proxy: TaggedValue

    def __post_init__(self) -> None:
        if self.energy_proxy.value is not None and self.energy_proxy.source != DataSource.UNAVAILABLE:
            if self.energy_proxy.value < 0:
                raise ValueError(f"energy_proxy cannot be negative: {self.energy_proxy.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy_proxy": self.energy_proxy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnergyState:
        return cls(
            energy_proxy=TaggedValue(
                value=data["energy_proxy"]["value"],
                source=DataSource(data["energy_proxy"]["source"]),
                unit=data["energy_proxy"].get("unit", "proxy_units"),
            )
        )


# ---------------------------------------------------------------------------
# Composite Runtime State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeState:
    """
    Unified canonical runtime state representation at a single discrete time step t.

    Integrates:
    - B_t, L_t, P_t, J_t: Network condition (bandwidth, latency, loss, jitter)
    - V_t, F_t, R_t: Memory state (VRAM allocated/free, RAM used/available)
    - G_t, C_t: Compute utilization (GPU, CPU)
    - K_t, Delta K_t, T_t: Inference progress & KV-cache state
    - A_t: Inter-tier activation transfer metadata
    - E_t: Energy proxy metric

    Preserves explicit DataSource provenance on all numeric components.
    """
    timestamp: float
    wall_clock: float
    step_index: int
    network: NetworkState
    memory: MemoryState
    compute: ComputeState
    inference: InferenceState
    activation: ActivationState
    energy: EnergyState
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "wall_clock": self.wall_clock,
            "step_index": self.step_index,
            "network": self.network.to_dict(),
            "memory": self.memory.to_dict(),
            "compute": self.compute.to_dict(),
            "inference": self.inference.to_dict(),
            "activation": self.activation.to_dict(),
            "energy": self.energy.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuntimeState:
        return cls(
            timestamp=data["timestamp"],
            wall_clock=data["wall_clock"],
            step_index=data["step_index"],
            network=NetworkState.from_dict(data["network"]),
            memory=MemoryState.from_dict(data["memory"]),
            compute=ComputeState.from_dict(data["compute"]),
            inference=InferenceState.from_dict(data["inference"]),
            activation=ActivationState.from_dict(data["activation"]),
            energy=EnergyState.from_dict(data["energy"]),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> RuntimeState:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_telemetry(
        cls,
        snapshot: TelemetrySnapshot,
        prev_state: Optional[RuntimeState] = None,
        request_state: str = "decode",
        request_id: Optional[str] = None,
        context_length: Optional[int] = None,
        wall_clock: Optional[float] = None,
    ) -> RuntimeState:
        """
        Construct a canonical RuntimeState from a raw TelemetrySnapshot.

        Preserves provenance for all fields. Missing sub-snapshots yield
        TaggedValue.unavailable fields (never imputed as 0).
        """
        cur_time = snapshot.timestamp
        cur_wall = wall_clock if wall_clock is not None else time.time()

        # --- Network ---
        if snapshot.network is not None:
            net_state = NetworkState(
                bandwidth_mbps=snapshot.network.bandwidth_mbps,
                latency_ms=snapshot.network.rtt_ms,
                packet_loss=snapshot.network.packet_loss_rate,
                jitter_ms=snapshot.network.jitter_ms,
            )
        else:
            net_state = NetworkState(
                bandwidth_mbps=TaggedValue.unavailable("Mbps"),
                latency_ms=TaggedValue.unavailable("ms"),
                packet_loss=TaggedValue.unavailable("ratio"),
                jitter_ms=TaggedValue.unavailable("ms"),
            )

        # --- Memory ---
        if snapshot.memory is not None:
            mem_state = MemoryState(
                vram_allocated_mb=snapshot.memory.vram_used_mb,
                vram_free_mb=snapshot.memory.vram_free_mb,
                ram_used_mb=snapshot.memory.ram_used_mb,
                ram_available_mb=snapshot.memory.ram_available_mb,
            )
        else:
            mem_state = MemoryState(
                vram_allocated_mb=TaggedValue.unavailable("MB"),
                vram_free_mb=TaggedValue.unavailable("MB"),
                ram_used_mb=TaggedValue.unavailable("MB"),
                ram_available_mb=TaggedValue.unavailable("MB"),
            )

        # --- Compute ---
        if snapshot.cpu is not None:
            cpu_val = snapshot.cpu.cpu_percent_overall
        else:
            cpu_val = TaggedValue.unavailable("%")
        gpu_val = TaggedValue.unavailable("%")  # No GPU on this dev environment

        comp_state = ComputeState(
            gpu_utilization=gpu_val,
            cpu_utilization=cpu_val,
        )

        # --- Inference ---
        if snapshot.kv_cache is not None:
            kv_bytes = snapshot.kv_cache.total_kv_bytes
            kv_growth = snapshot.kv_cache.kv_growth_bytes_since_last
        else:
            kv_bytes = TaggedValue.unavailable("bytes")
            kv_growth = TaggedValue.unavailable("bytes")

        gen_tokens = snapshot.step
        ctx_len = context_length if context_length is not None else gen_tokens

        # Generation rate (tokens / sec) derived if prev_state is available
        gen_rate: Optional[float] = None
        if prev_state is not None:
            dt = cur_time - prev_state.timestamp
            if dt > 0:
                d_tokens = gen_tokens - prev_state.inference.generated_tokens
                if d_tokens >= 0:
                    gen_rate = float(d_tokens / dt)

        inf_state = InferenceState(
            kv_cache_bytes=kv_bytes,
            kv_cache_growth_bytes=kv_growth,
            generated_tokens=gen_tokens,
            context_length=ctx_len,
            generation_rate=gen_rate,
            request_state=request_state,
            request_id=request_id,
        )

        # --- Activation ---
        latest_act_bytes: Optional[TaggedValue] = None
        latest_act_lat: Optional[TaggedValue] = None
        if snapshot.activations:
            # take the most recent activation transfer
            last_act = snapshot.activations[-1]
            latest_act_bytes = last_act.byte_size
            latest_act_lat = last_act.estimated_transfer_ms

        act_state = ActivationState(
            latest_activation_bytes=latest_act_bytes,
            latest_transfer_latency_ms=latest_act_lat,
        )

        # --- Energy Proxy ---
        # Analytical proxy = CPU utilization * delta_t (or step proxy)
        if cpu_val.value is not None and cpu_val.source != DataSource.UNAVAILABLE:
            dt_step = (cur_time - prev_state.timestamp) if (prev_state is not None and cur_time > prev_state.timestamp) else 0.05
            proxy_val = float(cpu_val.value * dt_step)
            energy_state = EnergyState(
                energy_proxy=TaggedValue.estimated(proxy_val, unit="proxy_units")
            )
        else:
            energy_state = EnergyState(
                energy_proxy=TaggedValue.unavailable("proxy_units")
            )

        return cls(
            timestamp=cur_time,
            wall_clock=cur_wall,
            step_index=snapshot.step,
            network=net_state,
            memory=mem_state,
            compute=comp_state,
            inference=inf_state,
            activation=act_state,
            energy=energy_state,
            metadata=dict(snapshot.metadata),
        )
