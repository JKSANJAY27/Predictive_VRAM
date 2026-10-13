"""
CUDA Hardware Telemetry Provider and Mock CUDA Provider for Module 11.

Uses PyTorch CUDA APIs for memory queries and optional NVML for hardware utilization.
Provides MockCUDAHardwareProvider for deterministic hardware-independent testing.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.hardware.cpu import CPUHardwareProvider
from src.hardware.provider import HardwareTelemetryProvider
from src.hardware.types import (
    DeviceDescriptor,
    DeviceType,
    HardwareComputeTelemetry,
    HardwareMemoryTelemetry,
    MeasurementRecord,
    TelemetryCapabilities,
)

# Optional NVML dynamic import guard
_NVML_AVAILABLE = False
_pynvml = None
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    _pynvml = pynvml
except Exception:
    _NVML_AVAILABLE = False
    _pynvml = None


class CUDAHardwareProvider(HardwareTelemetryProvider):
    """
    Hardware telemetry provider querying real NVIDIA CUDA devices via PyTorch & NVML.

    Guarantees:
    - Never blocks the inference hot path with torch.cuda.synchronize().
    - When NVML is unavailable, GPU compute utilization is tagged UNAVAILABLE (never 0.0).
    - Distinguishes device capacity, allocated memory, reserved memory, and free memory.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self._cpu_fallback = CPUHardwareProvider()
        self._cuda_ok = torch.cuda.is_available()
        if self.strict and not self._cuda_ok:
            raise RuntimeError("CUDAHardwareProvider initialized with strict=True but CUDA is not available.")

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CUDA

    @property
    def is_available(self) -> bool:
        return self._cuda_ok

    def discover_devices(self) -> List[DeviceDescriptor]:
        """Discover available CUDA execution devices."""
        if not self._cuda_ok:
            return []

        devices: List[DeviceDescriptor] = []
        count = torch.cuda.device_count()
        for idx in range(count):
            try:
                props = torch.cuda.get_device_properties(idx)
                dev = DeviceDescriptor(
                    device_type=DeviceType.CUDA,
                    device_index=idx,
                    name=props.name,
                    total_memory_bytes=props.total_memory,
                    compute_capability=(props.major, props.minor),
                    is_available=True,
                    metadata={
                        "multi_processor_count": getattr(props, "multi_processor_count", None),
                        "major": props.major,
                        "minor": props.minor,
                    },
                )
                devices.append(dev)
            except Exception as e:
                devices.append(
                    DeviceDescriptor(
                        device_type=DeviceType.CUDA,
                        device_index=idx,
                        name=f"CUDA:{idx}",
                        total_memory_bytes=None,
                        is_available=False,
                        metadata={"error": str(e)},
                    )
                )
        return devices

    def get_memory_telemetry(self, device_id: Optional[int] = None) -> HardwareMemoryTelemetry:
        """
        Sample VRAM and host RAM.
        """
        start = time.perf_counter()
        errors: List[str] = []
        idx = device_id if device_id is not None else 0

        # Sample host RAM via CPU provider
        cpu_mem = self._cpu_fallback.get_memory_telemetry()

        if not self._cuda_ok:
            errors.append("cuda_unavailable")
            vram_unavail = MeasurementRecord.unavailable(
                "bytes", error="CUDA not available on this platform"
            )
            return HardwareMemoryTelemetry(
                total_vram_bytes=vram_unavail,
                allocated_vram_bytes=vram_unavail,
                reserved_vram_bytes=vram_unavail,
                free_vram_bytes=vram_unavail,
                host_ram_total_bytes=cpu_mem.host_ram_total_bytes,
                host_ram_used_bytes=cpu_mem.host_ram_used_bytes,
                host_ram_available_bytes=cpu_mem.host_ram_available_bytes,
                process_rss_bytes=cpu_mem.process_rss_bytes,
                timestamp=time.monotonic(),
                collection_duration_ms=(time.perf_counter() - start) * 1000.0,
                errors=tuple(errors),
            )

        try:
            # torch.cuda.mem_get_info returns (free_bytes, total_bytes)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            total_vram = MeasurementRecord.measured(
                float(total_b), unit="bytes", method="torch.cuda.mem_get_info"
            )
            free_vram = MeasurementRecord.measured(
                float(free_b), unit="bytes", method="torch.cuda.mem_get_info"
            )
        except Exception as e:
            errors.append(f"mem_get_info_error: {e}")
            total_vram = MeasurementRecord.unavailable("bytes", error=str(e))
            free_vram = MeasurementRecord.unavailable("bytes", error=str(e))

        try:
            alloc_b = torch.cuda.memory_allocated(idx)
            alloc_vram = MeasurementRecord.measured(
                float(alloc_b), unit="bytes", method="torch.cuda.memory_allocated"
            )
        except Exception as e:
            errors.append(f"memory_allocated_error: {e}")
            alloc_vram = MeasurementRecord.unavailable("bytes", error=str(e))

        try:
            res_b = torch.cuda.memory_reserved(idx)
            res_vram = MeasurementRecord.measured(
                float(res_b), unit="bytes", method="torch.cuda.memory_reserved"
            )
        except Exception as e:
            errors.append(f"memory_reserved_error: {e}")
            res_vram = MeasurementRecord.unavailable("bytes", error=str(e))

        duration_ms = (time.perf_counter() - start) * 1000.0
        return HardwareMemoryTelemetry(
            total_vram_bytes=total_vram,
            allocated_vram_bytes=alloc_vram,
            reserved_vram_bytes=res_vram,
            free_vram_bytes=free_vram,
            host_ram_total_bytes=cpu_mem.host_ram_total_bytes,
            host_ram_used_bytes=cpu_mem.host_ram_used_bytes,
            host_ram_available_bytes=cpu_mem.host_ram_available_bytes,
            process_rss_bytes=cpu_mem.process_rss_bytes,
            timestamp=time.monotonic(),
            collection_duration_ms=duration_ms,
            errors=tuple(errors),
        )

    def get_compute_telemetry(self, device_id: Optional[int] = None) -> HardwareComputeTelemetry:
        """
        Sample GPU and CPU compute metrics.
        GPU utilization requires NVML; if absent, marked UNAVAILABLE (never 0.0).
        """
        start = time.perf_counter()
        errors: List[str] = []
        idx = device_id if device_id is not None else 0

        cpu_comp = self._cpu_fallback.get_compute_telemetry()

        if not self._cuda_ok:
            errors.append("cuda_unavailable")
            gpu_unavail = MeasurementRecord.unavailable(
                "%", error="CUDA not available on this platform"
            )
            return HardwareComputeTelemetry(
                cpu_utilization_pct=cpu_comp.cpu_utilization_pct,
                gpu_utilization_pct=gpu_unavail,
                gpu_memory_utilization_pct=gpu_unavail,
                cpu_freq_mhz=cpu_comp.cpu_freq_mhz,
                cpu_count_logical=cpu_comp.cpu_count_logical,
                cpu_count_physical=cpu_comp.cpu_count_physical,
                timestamp=time.monotonic(),
                collection_duration_ms=(time.perf_counter() - start) * 1000.0,
                errors=tuple(errors),
            )

        gpu_util = MeasurementRecord.unavailable(
            "%", error="NVML not available or unsupported for GPU utilization"
        )
        gpu_mem_util = MeasurementRecord.unavailable(
            "%", error="NVML not available or unsupported for GPU memory utilization"
        )

        if _NVML_AVAILABLE and _pynvml is not None:
            try:
                handle = _pynvml.nvmlDeviceGetHandleByIndex(idx)
                rates = _pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = MeasurementRecord.measured(
                    float(rates.gpu), unit="%", method="nvmlDeviceGetUtilizationRates"
                )
                gpu_mem_util = MeasurementRecord.measured(
                    float(rates.memory), unit="%", method="nvmlDeviceGetUtilizationRates"
                )
            except Exception as e:
                errors.append(f"nvml_query_error: {e}")
                gpu_util = MeasurementRecord.unavailable("%", error=str(e))
                gpu_mem_util = MeasurementRecord.unavailable("%", error=str(e))

        duration_ms = (time.perf_counter() - start) * 1000.0
        return HardwareComputeTelemetry(
            cpu_utilization_pct=cpu_comp.cpu_utilization_pct,
            gpu_utilization_pct=gpu_util,
            gpu_memory_utilization_pct=gpu_mem_util,
            cpu_freq_mhz=cpu_comp.cpu_freq_mhz,
            cpu_count_logical=cpu_comp.cpu_count_logical,
            cpu_count_physical=cpu_comp.cpu_count_physical,
            timestamp=time.monotonic(),
            collection_duration_ms=duration_ms,
            errors=tuple(errors),
        )

    def get_telemetry_capabilities(self, device_id: Optional[int] = None) -> TelemetryCapabilities:
        has_cuda = self._cuda_ok
        return TelemetryCapabilities(
            has_host_ram=True,
            has_cpu_util=True,
            has_vram_total=has_cuda,
            has_vram_allocated=has_cuda,
            has_vram_reserved=has_cuda,
            has_vram_free=has_cuda,
            has_gpu_util=has_cuda and _NVML_AVAILABLE,
            has_nvml=_NVML_AVAILABLE,
        )


class MockCUDAHardwareProvider(HardwareTelemetryProvider):
    """
    Configurable Mock CUDA provider for deterministic unit testing.

    Simulates CUDA GPU memory allocation growth, reserved pools, and GPU utilization
    without requiring physical NVIDIA hardware.
    """

    def __init__(
        self,
        device_name: str = "NVIDIA RTX 4090 (Mocked)",
        total_vram_bytes: int = 24 * 1024 * 1024 * 1024,  # 24 GB
        initial_allocated_bytes: int = 2 * 1024 * 1024 * 1024,  # 2 GB
        initial_reserved_bytes: int = 3 * 1024 * 1024 * 1024,   # 3 GB
        simulated_gpu_utilization_pct: Optional[float] = 45.0,
        compute_capability: Tuple[int, int] = (8, 9),
    ) -> None:
        self.mock_name = device_name
        self.mock_total = total_vram_bytes
        self.mock_allocated = initial_allocated_bytes
        self.mock_reserved = initial_reserved_bytes
        self.mock_gpu_util = simulated_gpu_utilization_pct
        self.mock_compute_cap = compute_capability
        self._cpu_fallback = CPUHardwareProvider()

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.MOCK_CUDA

    @property
    def is_available(self) -> bool:
        return True

    def set_allocated_bytes(self, bytes_val: int) -> None:
        """Update simulated allocated bytes."""
        self.mock_allocated = bytes_val
        if self.mock_allocated > self.mock_reserved:
            self.mock_reserved = self.mock_allocated + 512 * 1024 * 1024  # Add 512 MB reserved headroom

    def discover_devices(self) -> List[DeviceDescriptor]:
        return [
            DeviceDescriptor(
                device_type=DeviceType.MOCK_CUDA,
                device_index=0,
                name=self.mock_name,
                total_memory_bytes=self.mock_total,
                compute_capability=self.mock_compute_cap,
                is_available=True,
                metadata={"mock": True},
            )
        ]

    def get_memory_telemetry(self, device_id: Optional[int] = None) -> HardwareMemoryTelemetry:
        start = time.perf_counter()
        cpu_mem = self._cpu_fallback.get_memory_telemetry()

        free_b = max(0, self.mock_total - self.mock_reserved)
        # Note: Mock uses DataSource.EMULATED to preserve strict research provenance!
        total_rec = MeasurementRecord.emulated(
            float(self.mock_total), unit="bytes", method="mock_cuda.total"
        )
        alloc_rec = MeasurementRecord.emulated(
            float(self.mock_allocated), unit="bytes", method="mock_cuda.allocated"
        )
        res_rec = MeasurementRecord.emulated(
            float(self.mock_reserved), unit="bytes", method="mock_cuda.reserved"
        )
        free_rec = MeasurementRecord.emulated(
            float(free_b), unit="bytes", method="mock_cuda.free"
        )

        return HardwareMemoryTelemetry(
            total_vram_bytes=total_rec,
            allocated_vram_bytes=alloc_rec,
            reserved_vram_bytes=res_rec,
            free_vram_bytes=free_rec,
            host_ram_total_bytes=cpu_mem.host_ram_total_bytes,
            host_ram_used_bytes=cpu_mem.host_ram_used_bytes,
            host_ram_available_bytes=cpu_mem.host_ram_available_bytes,
            process_rss_bytes=cpu_mem.process_rss_bytes,
            timestamp=time.monotonic(),
            collection_duration_ms=(time.perf_counter() - start) * 1000.0,
            errors=(),
        )

    def get_compute_telemetry(self, device_id: Optional[int] = None) -> HardwareComputeTelemetry:
        start = time.perf_counter()
        cpu_comp = self._cpu_fallback.get_compute_telemetry()

        if self.mock_gpu_util is not None:
            gpu_util = MeasurementRecord.emulated(
                float(self.mock_gpu_util), unit="%", method="mock_cuda.util"
            )
            mem_util = MeasurementRecord.emulated(
                float(round((self.mock_allocated / max(1, self.mock_total)) * 100.0, 1)),
                unit="%",
                method="mock_cuda.mem_util",
            )
        else:
            gpu_util = MeasurementRecord.unavailable("%", error="Mock GPU utilization configured as unavailable")
            mem_util = MeasurementRecord.unavailable("%", error="Mock GPU memory util unavailable")

        return HardwareComputeTelemetry(
            cpu_utilization_pct=cpu_comp.cpu_utilization_pct,
            gpu_utilization_pct=gpu_util,
            gpu_memory_utilization_pct=mem_util,
            cpu_freq_mhz=cpu_comp.cpu_freq_mhz,
            cpu_count_logical=cpu_comp.cpu_count_logical,
            cpu_count_physical=cpu_comp.cpu_count_physical,
            timestamp=time.monotonic(),
            collection_duration_ms=(time.perf_counter() - start) * 1000.0,
            errors=(),
        )

    def get_telemetry_capabilities(self, device_id: Optional[int] = None) -> TelemetryCapabilities:
        return TelemetryCapabilities(
            has_host_ram=True,
            has_cpu_util=True,
            has_vram_total=True,
            has_vram_allocated=True,
            has_vram_reserved=True,
            has_vram_free=True,
            has_gpu_util=self.mock_gpu_util is not None,
            has_nvml=False,
        )
