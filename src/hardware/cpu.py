"""
CPU Hardware Telemetry Provider for Module 11.

Provides physical host memory and CPU utilization telemetry.
VRAM and GPU compute fields are strictly tagged DataSource.UNAVAILABLE.
Host RAM is NEVER substituted for VRAM.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict, List, Optional

from src.hardware.provider import HardwareTelemetryProvider
from src.hardware.types import (
    DeviceDescriptor,
    DeviceType,
    HardwareComputeTelemetry,
    HardwareMemoryTelemetry,
    MeasurementRecord,
    TelemetryCapabilities,
)

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


class CPUHardwareProvider(HardwareTelemetryProvider):
    """
    Hardware provider for CPU-only execution and host platform telemetry.

    Invariants:
    - VRAM metrics are always DataSource.UNAVAILABLE with None value.
    - GPU utilization is always DataSource.UNAVAILABLE with None value (never 0.0).
    - Host RAM and CPU metrics are MEASURED when psutil is available.
    """

    def __init__(self) -> None:
        self._psutil_ok = _PSUTIL_AVAILABLE

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    @property
    def is_available(self) -> bool:
        return True

    def discover_devices(self) -> List[DeviceDescriptor]:
        """Discover host CPU execution context."""
        total_ram: Optional[int] = None
        if self._psutil_ok and psutil is not None:
            try:
                total_ram = int(psutil.virtual_memory().total)
            except Exception:
                total_ram = None

        cpu_name = platform.processor() or f"CPU ({platform.machine()})"
        return [
            DeviceDescriptor(
                device_type=DeviceType.CPU,
                device_index=0,
                name=cpu_name,
                total_memory_bytes=total_ram,
                compute_capability=None,
                is_available=True,
                metadata={
                    "platform": platform.platform(),
                    "architecture": platform.machine(),
                    "python_version": platform.python_version(),
                },
            )
        ]

    def get_memory_telemetry(self, device_id: Optional[int] = None) -> HardwareMemoryTelemetry:
        """
        Sample host RAM. VRAM is strictly marked UNAVAILABLE.
        """
        start = time.perf_counter()
        errors: List[str] = []

        if self._psutil_ok and psutil is not None:
            try:
                vm = psutil.virtual_memory()
                ram_total = MeasurementRecord.measured(
                    float(vm.total), unit="bytes", method="psutil.virtual_memory"
                )
                ram_used = MeasurementRecord.measured(
                    float(vm.used), unit="bytes", method="psutil.virtual_memory"
                )
                ram_avail = MeasurementRecord.measured(
                    float(vm.available), unit="bytes", method="psutil.virtual_memory"
                )
            except Exception as e:
                errors.append(f"virtual_memory_error: {e}")
                ram_total = MeasurementRecord.unavailable("bytes", error=str(e))
                ram_used = MeasurementRecord.unavailable("bytes", error=str(e))
                ram_avail = MeasurementRecord.unavailable("bytes", error=str(e))

            try:
                proc = psutil.Process()
                rss = MeasurementRecord.measured(
                    float(proc.memory_info().rss), unit="bytes", method="psutil.Process.rss"
                )
            except Exception as e:
                errors.append(f"process_rss_error: {e}")
                rss = MeasurementRecord.unavailable("bytes", error=str(e))
        else:
            errors.append("psutil_not_installed")
            ram_total = MeasurementRecord.unavailable("bytes", error="psutil not installed")
            ram_used = MeasurementRecord.unavailable("bytes", error="psutil not installed")
            ram_avail = MeasurementRecord.unavailable("bytes", error="psutil not installed")
            rss = MeasurementRecord.unavailable("bytes", error="psutil not installed")

        # VRAM is strictly UNAVAILABLE on CPU provider
        vram_unavail = MeasurementRecord.unavailable(
            unit="bytes",
            error="CUDA GPU not available on CPU-only provider; host RAM not substituted",
        )

        duration_ms = (time.perf_counter() - start) * 1000.0
        return HardwareMemoryTelemetry(
            total_vram_bytes=vram_unavail,
            allocated_vram_bytes=vram_unavail,
            reserved_vram_bytes=vram_unavail,
            free_vram_bytes=vram_unavail,
            host_ram_total_bytes=ram_total,
            host_ram_used_bytes=ram_used,
            host_ram_available_bytes=ram_avail,
            process_rss_bytes=rss,
            timestamp=time.monotonic(),
            collection_duration_ms=duration_ms,
            errors=tuple(errors),
        )

    def get_compute_telemetry(self, device_id: Optional[int] = None) -> HardwareComputeTelemetry:
        """
        Sample CPU compute metrics. GPU utilization is strictly marked UNAVAILABLE.
        """
        start = time.perf_counter()
        errors: List[str] = []

        if self._psutil_ok and psutil is not None:
            try:
                # Non-blocking CPU sample (interval=None) avoids blocking the inference loop
                cpu_pct = MeasurementRecord.measured(
                    float(psutil.cpu_percent(interval=None)), unit="%", method="psutil.cpu_percent"
                )
            except Exception as e:
                errors.append(f"cpu_percent_error: {e}")
                cpu_pct = MeasurementRecord.unavailable("%", error=str(e))

            try:
                freq = psutil.cpu_freq()
                if freq and freq.current is not None:
                    cpu_freq = MeasurementRecord.measured(
                        float(freq.current), unit="MHz", method="psutil.cpu_freq"
                    )
                else:
                    cpu_freq = MeasurementRecord.unavailable("MHz", error="CPU frequency unavailable")
            except Exception as e:
                errors.append(f"cpu_freq_error: {e}")
                cpu_freq = MeasurementRecord.unavailable("MHz", error=str(e))

            try:
                log_count = psutil.cpu_count(logical=True)
                cpu_log = MeasurementRecord.measured(
                    float(log_count) if log_count else 1.0, unit="cores", method="psutil.cpu_count"
                )
            except Exception as e:
                errors.append(f"logical_cpu_error: {e}")
                cpu_log = MeasurementRecord.unavailable("cores", error=str(e))

            try:
                phys_count = psutil.cpu_count(logical=False)
                if phys_count:
                    cpu_phys = MeasurementRecord.measured(
                        float(phys_count), unit="cores", method="psutil.cpu_count"
                    )
                else:
                    cpu_phys = MeasurementRecord.unavailable("cores", error="Physical count not determined")
            except Exception as e:
                errors.append(f"physical_cpu_error: {e}")
                cpu_phys = MeasurementRecord.unavailable("cores", error=str(e))
        else:
            errors.append("psutil_not_installed")
            cpu_pct = MeasurementRecord.unavailable("%", error="psutil not installed")
            cpu_freq = MeasurementRecord.unavailable("MHz", error="psutil not installed")
            cpu_log = MeasurementRecord.unavailable("cores", error="psutil not installed")
            cpu_phys = MeasurementRecord.unavailable("cores", error="psutil not installed")

        # GPU utilization is strictly UNAVAILABLE - never reported as 0.0
        gpu_unavail = MeasurementRecord.unavailable(
            unit="%",
            error="GPU utilization unavailable: No CUDA accelerator on CPU-only provider",
        )

        duration_ms = (time.perf_counter() - start) * 1000.0
        return HardwareComputeTelemetry(
            cpu_utilization_pct=cpu_pct,
            gpu_utilization_pct=gpu_unavail,
            gpu_memory_utilization_pct=gpu_unavail,
            cpu_freq_mhz=cpu_freq,
            cpu_count_logical=cpu_log,
            cpu_count_physical=cpu_phys,
            timestamp=time.monotonic(),
            collection_duration_ms=duration_ms,
            errors=tuple(errors),
        )

    def get_telemetry_capabilities(self, device_id: Optional[int] = None) -> TelemetryCapabilities:
        return TelemetryCapabilities(
            has_host_ram=self._psutil_ok,
            has_cpu_util=self._psutil_ok,
            has_vram_total=False,
            has_vram_allocated=False,
            has_vram_reserved=False,
            has_vram_free=False,
            has_gpu_util=False,
            has_nvml=False,
        )
