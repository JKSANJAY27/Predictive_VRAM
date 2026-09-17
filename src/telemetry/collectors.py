"""
Telemetry collectors for the Predictive VRAM split inference system.

Each collector is a thin, stateless class that samples exactly one aspect
of the system and returns an immutable snapshot dataclass.

DataSource tags on every field guarantee that no metric is silently fabricated:
- MEASURED  — Direct OS/runtime observation (psutil, torch tensor attributes).
- ESTIMATED — Analytically derived (e.g., transfer latency from bandwidth).
- UNAVAILABLE — Required hardware/software is absent (no CUDA GPU here).
- EMULATED  — Externally injected network state.

Dependency notes:
- psutil is required for MemoryCollector and CPUCollector.
  If psutil is not installed, both collectors fall back gracefully,
  returning UNAVAILABLE tags instead of raising ImportError at call time.
- No torch.cuda.* calls appear anywhere in this module.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from src.telemetry.types import (
    ActivationSnapshot,
    CPUSnapshot,
    DataSource,
    KVCacheSnapshot,
    MemorySnapshot,
    NetworkConditionSnapshot,
    TaggedValue,
)
from src.telemetry.network_emulation import NetworkEmulationState

# ---------------------------------------------------------------------------
# psutil availability guard — import once at module level
# ---------------------------------------------------------------------------
try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Memory Collector
# ---------------------------------------------------------------------------

class MemoryCollector:
    """
    Collects system RAM statistics via psutil.

    VRAM fields are always tagged UNAVAILABLE because this development machine
    has no CUDA-capable GPU. The collector never calls any torch.cuda.* API.

    Usage:
        collector = MemoryCollector()
        snapshot = collector.collect()
    """

    def __init__(self) -> None:
        self._psutil_ok = _PSUTIL_AVAILABLE

    def collect(self) -> MemorySnapshot:
        """
        Sample current memory state.

        Returns:
            MemorySnapshot with MEASURED RAM fields and UNAVAILABLE VRAM fields.
        """
        if self._psutil_ok:
            vm = _psutil.virtual_memory()
            sw = _psutil.swap_memory()
            to_mb = lambda b: round(b / (1024 * 1024), 3)

            ram_total   = TaggedValue.measured(to_mb(vm.total),     "MB")
            ram_used    = TaggedValue.measured(to_mb(vm.used),      "MB")
            ram_avail   = TaggedValue.measured(to_mb(vm.available), "MB")
            ram_pct     = TaggedValue.measured(vm.percent,          "%")
            swap_used   = TaggedValue.measured(to_mb(sw.used),      "MB")
            swap_total  = TaggedValue.measured(to_mb(sw.total),     "MB")

            # Per-process RSS
            try:
                proc = _psutil.Process()
                proc_mb = TaggedValue.measured(
                    to_mb(proc.memory_info().rss), "MB"
                )
            except Exception:
                proc_mb = TaggedValue.unavailable("MB")
        else:
            # psutil unavailable — all RAM fields degrade gracefully
            ram_total  = TaggedValue.unavailable("MB")
            ram_used   = TaggedValue.unavailable("MB")
            ram_avail  = TaggedValue.unavailable("MB")
            ram_pct    = TaggedValue.unavailable("%")
            swap_used  = TaggedValue.unavailable("MB")
            swap_total = TaggedValue.unavailable("MB")
            proc_mb    = TaggedValue.unavailable("MB")

        # VRAM: always UNAVAILABLE on this machine — no GPU, no fabrication
        vram_unavailable = TaggedValue.unavailable("MB")

        return MemorySnapshot(
            ram_total_mb=ram_total,
            ram_used_mb=ram_used,
            ram_available_mb=ram_avail,
            ram_percent=ram_pct,
            swap_used_mb=swap_used,
            swap_total_mb=swap_total,
            vram_total_mb=vram_unavailable,
            vram_used_mb=vram_unavailable,
            vram_free_mb=vram_unavailable,
            process_ram_mb=proc_mb,
            timestamp=time.monotonic(),
        )


# ---------------------------------------------------------------------------
# CPU Collector
# ---------------------------------------------------------------------------

class CPUCollector:
    """
    Collects CPU utilisation and topology via psutil.

    `cpu_percent(interval=None)` returns the non-blocking measurement since
    the last call (or 0.0 on the very first call). This is intentional:
    blocking here would stall the generation loop.

    Usage:
        collector = CPUCollector()
        snapshot = collector.collect()
    """

    def __init__(self) -> None:
        self._psutil_ok = _PSUTIL_AVAILABLE
        if self._psutil_ok:
            # Prime the non-blocking percent counter
            _psutil.cpu_percent(interval=None)

    def collect(self) -> CPUSnapshot:
        """
        Sample current CPU state.

        Returns:
            CPUSnapshot with MEASURED fields (or UNAVAILABLE if psutil absent).
        """
        if self._psutil_ok:
            pct = _psutil.cpu_percent(interval=None)
            logical  = _psutil.cpu_count(logical=True) or 1
            physical = _psutil.cpu_count(logical=False) or 1

            try:
                freq = _psutil.cpu_freq()
                freq_mhz = TaggedValue.measured(freq.current if freq else 0.0, "MHz")
            except Exception:
                freq_mhz = TaggedValue.unavailable("MHz")

            return CPUSnapshot(
                cpu_percent_overall=TaggedValue.measured(pct, "%"),
                cpu_count_logical=TaggedValue.measured(float(logical), "cores"),
                cpu_count_physical=TaggedValue.measured(float(physical), "cores"),
                cpu_freq_mhz=freq_mhz,
                timestamp=time.monotonic(),
            )
        else:
            un = TaggedValue.unavailable
            return CPUSnapshot(
                cpu_percent_overall=un("%"),
                cpu_count_logical=un("cores"),
                cpu_count_physical=un("cores"),
                cpu_freq_mhz=un("MHz"),
                timestamp=time.monotonic(),
            )


# ---------------------------------------------------------------------------
# Activation Collector
# ---------------------------------------------------------------------------

class ActivationCollector:
    """
    Derives metadata from an inter-tier activation tensor.

    Shape, element count, and byte size are MEASURED directly from the tensor.
    Estimated transfer latency is EMULATED using the NetworkEmulationState.

    Usage:
        collector = ActivationCollector(network_state=my_emulation_state)
        snapshot = collector.collect(tensor, source_tier="user_device",
                                     destination_tier="edge_a", step=0)
    """

    def __init__(
        self,
        network_state: Optional[NetworkEmulationState] = None,
    ) -> None:
        self._network_state = network_state or NetworkEmulationState()

    def collect(
        self,
        tensor: torch.Tensor,
        source_tier: str,
        destination_tier: str,
        step: Optional[int] = None,
    ) -> ActivationSnapshot:
        """
        Measure a tensor and return its activation snapshot.

        Args:
            tensor: The inter-tier activation tensor.
            source_tier: String name of the originating tier.
            destination_tier: String name of the receiving tier.
            step: Optional autoregressive step index.

        Returns:
            ActivationSnapshot with MEASURED size and EMULATED latency estimate.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"ActivationCollector.collect() expects a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )

        num_elements = tensor.numel()
        byte_size    = num_elements * tensor.element_size()

        estimated_ms = self._network_state.estimate_transfer_ms(byte_size)

        return ActivationSnapshot(
            tensor_shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            byte_size=TaggedValue.measured(float(byte_size), "bytes"),
            num_elements=TaggedValue.measured(float(num_elements), "elements"),
            estimated_transfer_ms=TaggedValue.emulated(estimated_ms, "ms"),
            source_tier=source_tier,
            destination_tier=destination_tier,
            step=step,
            timestamp=time.monotonic(),
        )


# ---------------------------------------------------------------------------
# KV-Cache Collector
# ---------------------------------------------------------------------------

class KVCacheCollector:
    """
    Derives KV-cache memory statistics from a DynamicCache instance.

    Byte counts are MEASURED by iterating the cache's internal tensors.
    Growth delta is ESTIMATED by comparison with the previous call's total.

    Usage:
        collector = KVCacheCollector()
        snapshot = collector.collect(kv_cache, step=0)
    """

    def __init__(self) -> None:
        self._previous_total_bytes: int = 0

    def collect(
        self,
        kv_cache: object,
        step: Optional[int] = None,
    ) -> KVCacheSnapshot:
        """
        Measure current KV-cache state.

        Args:
            kv_cache: A transformers DynamicCache instance (or compatible).
            step: Optional autoregressive step index.

        Returns:
            KVCacheSnapshot with MEASURED byte counts.
        """
        per_layer_bytes: list[int] = []
        sample_shape: tuple[int, ...] = ()
        total_bytes = 0

        cache_layers = getattr(kv_cache, "layers", [])
        for layer_cache in cache_layers:
            k = getattr(layer_cache, "keys", None)
            v = getattr(layer_cache, "values", None)
            layer_bytes = 0
            if k is not None and isinstance(k, torch.Tensor):
                layer_bytes += k.numel() * k.element_size()
                if not sample_shape:
                    sample_shape = tuple(k.shape)
            if v is not None and isinstance(v, torch.Tensor):
                layer_bytes += v.numel() * v.element_size()
            per_layer_bytes.append(layer_bytes)
            total_bytes += layer_bytes

        growth = total_bytes - self._previous_total_bytes
        self._previous_total_bytes = total_bytes
        total_mb = round(total_bytes / (1024 * 1024), 6)

        return KVCacheSnapshot(
            num_cached_layers=TaggedValue.measured(
                float(len(cache_layers)), "layers"
            ),
            total_kv_bytes=TaggedValue.measured(float(total_bytes), "bytes"),
            total_kv_mb=TaggedValue.measured(total_mb, "MB"),
            kv_growth_bytes_since_last=TaggedValue.estimated(float(growth), "bytes"),
            per_layer_bytes=per_layer_bytes,
            sample_layer_shape=sample_shape,
            step=step,
            timestamp=time.monotonic(),
        )


# ---------------------------------------------------------------------------
# Network Condition Collector
# ---------------------------------------------------------------------------

class NetworkConditionCollector:
    """
    Reads current network conditions from an injectable NetworkEmulationState.

    ALL returned fields are tagged EMULATED. There are no real network probes,
    no ping commands, no bandwidth tests. This is purely a read of the current
    emulation state, which is controlled externally for experiments.

    Usage:
        state = NetworkEmulationState()
        collector = NetworkConditionCollector(state=state)
        snapshot = collector.collect()

        # Change scenario:
        state.update(rtt_ms=80.0, bandwidth_mbps=10.0, scenario_label="degraded")
        snapshot2 = collector.collect()
    """

    def __init__(
        self,
        state: Optional[NetworkEmulationState] = None,
    ) -> None:
        self._state = state or NetworkEmulationState()

    @property
    def state(self) -> NetworkEmulationState:
        """Access the underlying emulation state for external mutation."""
        return self._state

    def collect(self) -> NetworkConditionSnapshot:
        """
        Capture current network emulation state as an immutable snapshot.

        Returns:
            NetworkConditionSnapshot with all fields tagged EMULATED.
        """
        s = self._state
        return NetworkConditionSnapshot(
            rtt_ms=TaggedValue.emulated(s.rtt_ms, "ms"),
            bandwidth_mbps=TaggedValue.emulated(s.bandwidth_mbps, "Mbps"),
            packet_loss_rate=TaggedValue.emulated(s.packet_loss_rate, ""),
            jitter_ms=TaggedValue.emulated(s.jitter_ms, "ms"),
            scenario_label=s.scenario_label,
            timestamp=time.monotonic(),
        )
