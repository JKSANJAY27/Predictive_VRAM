"""
Hardware-Independent Test Suite for Module 11.

Covers:
- CPU provider behavior and integrity (no VRAM substitution)
- CUDA-unavailable behavior and Mock CUDA provider
- Measurement provenance retention and timestamp validity
- Device capability discovery and unknown capacity validation
- Real network measurement interface, probe metrics, and failure handling
- Safe framed TCP transport, HMAC authentication, and SHA-256 integrity checks
- Transfer timeout, payload limit, cancellation, and connection failure
- Network profile catalog (all 10 profiles + 6G exploratory)
- Platform-guarded Linux netem safety (no accidental OS changes)
- Memory validation engine
- Integration with ClosedLoopRuntime and MigrationManager
"""

import hashlib
import socket
import threading
import time
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from src.hardware.capabilities import DeviceCapabilityRegistry, TierCapabilities
from src.hardware.cpu import CPUHardwareProvider
from src.hardware.cuda import CUDAHardwareProvider, MockCUDAHardwareProvider
from src.hardware.evaluation_scenarios import SIX_G_SCENARIOS
from src.hardware.factory import HardwareProviderFactory
from src.hardware.linux_netem import LinuxNetemAdapter
from src.hardware.memory_validation import MemoryValidationEngine
from src.hardware.network_measurement import MockNetworkProbe, SocketNetworkProbe
from src.hardware.network_profiles import NetworkProfileCatalog
from src.hardware.transport import (
    AuthenticationError,
    IntegrityError,
    PayloadTooLargeError,
    TCPNetworkTransferProvider,
    TransferError,
    TransferServer,
    deserialize_module_safe,
    deserialize_tensor_safe,
    pack_frame,
    serialize_module_safe,
    serialize_tensor_safe,
)
from src.hardware.types import (
    DeviceDescriptor,
    DeviceType,
    EndpointMetadata,
    MeasurementRecord,
    TelemetryCapabilities,
)
from src.migration.manager import MigrationManager
from src.orchestration.config import OrchestrationConfig
from src.orchestration.runtime import ClosedLoopRuntime
from src.orchestration.scenarios import ScenarioRegistry
from src.orchestration.types import OrchestrationMode
from src.runtime.tier import TierId
from src.telemetry.types import DataSource


# ===========================================================================
# 1. CPU Hardware Provider Tests
# ===========================================================================

class TestCPUHardwareProvider:
    def test_cpu_discovery(self):
        provider = CPUHardwareProvider()
        assert provider.device_type == DeviceType.CPU
        assert provider.is_available is True
        devices = provider.discover_devices()
        assert len(devices) == 1
        assert devices[0].device_type == DeviceType.CPU

    def test_vram_is_strictly_unavailable_on_cpu(self):
        """Invariant: Host RAM must NEVER substitute for VRAM on CPU-only execution."""
        provider = CPUHardwareProvider()
        mem = provider.get_memory_telemetry()
        assert mem.total_vram_bytes.source == DataSource.UNAVAILABLE
        assert mem.allocated_vram_bytes.source == DataSource.UNAVAILABLE
        assert mem.reserved_vram_bytes.source == DataSource.UNAVAILABLE
        assert mem.free_vram_bytes.source == DataSource.UNAVAILABLE
        assert mem.total_vram_bytes.value is None

        # Host RAM must be separate
        assert mem.host_ram_total_bytes.source in (DataSource.MEASURED, DataSource.UNAVAILABLE)
        if mem.host_ram_total_bytes.source == DataSource.MEASURED:
            assert mem.host_ram_total_bytes.value > 0

    def test_gpu_utilization_never_reported_as_zero_on_cpu(self):
        """Invariant: Unsupported GPU utilization must be UNAVAILABLE, never 0.0."""
        provider = CPUHardwareProvider()
        comp = provider.get_compute_telemetry()
        assert comp.gpu_utilization_pct.source == DataSource.UNAVAILABLE
        assert comp.gpu_utilization_pct.value is None
        assert comp.gpu_memory_utilization_pct.source == DataSource.UNAVAILABLE
        assert comp.gpu_memory_utilization_pct.value is None

    def test_measurement_timestamps_and_durations(self):
        provider = CPUHardwareProvider()
        t0 = time.monotonic()
        mem = provider.get_memory_telemetry()
        t1 = time.monotonic()
        assert t0 <= mem.timestamp <= t1
        assert mem.collection_duration_ms >= 0.0


# ===========================================================================
# 2. CUDA & Mock CUDA Provider Tests
# ===========================================================================

class TestCUDAHardwareProvider:
    def test_cuda_unavailable_fallback(self):
        # Strict=False should not crash if CUDA is not available
        provider = CUDAHardwareProvider(strict=False)
        assert provider.device_type == DeviceType.CUDA
        if not torch.cuda.is_available():
            assert provider.is_available is False
            devices = provider.discover_devices()
            assert len(devices) == 0
            mem = provider.get_memory_telemetry()
            assert mem.total_vram_bytes.source == DataSource.UNAVAILABLE

    def test_cuda_strict_raises_when_missing(self):
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="strict=True but CUDA is not available"):
                CUDAHardwareProvider(strict=True)

    def test_mock_cuda_provider(self):
        mock = MockCUDAHardwareProvider(
            device_name="Test GPU",
            total_vram_bytes=16 * 1024 * 1024 * 1024,
            initial_allocated_bytes=4 * 1024 * 1024 * 1024,
            initial_reserved_bytes=6 * 1024 * 1024 * 1024,
            simulated_gpu_utilization_pct=60.0,
        )
        assert mock.device_type == DeviceType.MOCK_CUDA
        assert mock.is_available is True

        devices = mock.discover_devices()
        assert len(devices) == 1
        assert devices[0].total_memory_bytes == 16 * 1024 * 1024 * 1024

        mem = mock.get_memory_telemetry()
        assert mem.total_vram_bytes.value == 16 * 1024 * 1024 * 1024
        assert mem.allocated_vram_bytes.value == 4 * 1024 * 1024 * 1024
        assert mem.reserved_vram_bytes.value == 6 * 1024 * 1024 * 1024
        assert mem.free_vram_bytes.value == 10 * 1024 * 1024 * 1024
        # Mock preserves DataSource.EMULATED provenance
        assert mem.total_vram_bytes.source == DataSource.EMULATED

        comp = mock.get_compute_telemetry()
        assert comp.gpu_utilization_pct.value == 60.0
        assert comp.gpu_utilization_pct.source == DataSource.EMULATED

    def test_mock_cuda_dynamic_memory_growth(self):
        mock = MockCUDAHardwareProvider(
            total_vram_bytes=8 * 1024 * 1024 * 1024,
            initial_allocated_bytes=1 * 1024 * 1024 * 1024,
            initial_reserved_bytes=2 * 1024 * 1024 * 1024,
        )
        mock.set_allocated_bytes(5 * 1024 * 1024 * 1024)
        mem = mock.get_memory_telemetry()
        assert mem.allocated_vram_bytes.value == 5 * 1024 * 1024 * 1024
        assert mem.reserved_vram_bytes.value >= 5 * 1024 * 1024 * 1024


# ===========================================================================
# 3. Factory & Capabilities Tests
# ===========================================================================

class TestCapabilityRegistry:
    def test_factory_creation(self):
        cpu_p = HardwareProviderFactory.create_provider("cpu")
        assert cpu_p.device_type == DeviceType.CPU
        mock_p = HardwareProviderFactory.create_provider("mock")
        assert mock_p.device_type == DeviceType.MOCK_CUDA
        auto_p = HardwareProviderFactory.create_provider("auto")
        assert auto_p.device_type in (DeviceType.CPU, DeviceType.CUDA)

    def test_factory_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown hardware provider"):
            HardwareProviderFactory.create_provider("invalid_type")

    def test_unknown_capacity_never_treated_as_unlimited(self):
        """Invariant: Do not interpret an unknown capacity as unlimited capacity."""
        registry = DeviceCapabilityRegistry()
        caps = TierCapabilities(
            tier_id=TierId.EDGE_A,
            device_type=DeviceType.CPU,
            device_name="Unknown Node",
            total_memory_bytes=None,  # UNKNOWN capacity
            total_memory_provenance=DataSource.UNAVAILABLE,
            available_memory_bytes=None,
            available_memory_provenance=DataSource.UNAVAILABLE,
            supported_dtypes=("float32",),
            dtypes_provenance=DataSource.ESTIMATED,
        )
        registry.register(caps)

        # Checking if 100 MB can fit should fail because capacity is unknown
        ok, reason = registry.validate_capacity(TierId.EDGE_A, required_bytes=100 * 1024 * 1024)
        assert ok is False
        assert "cannot assume unlimited capacity" in reason

    def test_capacity_validation_checks_bounds(self):
        registry = DeviceCapabilityRegistry()
        registry.configure_operator_tier(
            tier_id=TierId.EDGE_A,
            device_type=DeviceType.CUDA,
            device_name="Remote GPU",
            total_memory_bytes=8 * 1024 * 1024 * 1024,
            available_memory_bytes=2 * 1024 * 1024 * 1024,
        )
        # 1 GB fits
        ok1, _ = registry.validate_capacity(TierId.EDGE_A, 1 * 1024 * 1024 * 1024)
        assert ok1 is True
        # 3 GB exceeds available 2 GB
        ok2, reason2 = registry.validate_capacity(TierId.EDGE_A, 3 * 1024 * 1024 * 1024)
        assert ok2 is False
        assert "less than required" in reason2


# ===========================================================================
# 4. Network Measurement & Probing Tests
# ===========================================================================

class TestNetworkMeasurement:
    def test_mock_network_probe(self):
        probe = MockNetworkProbe(simulated_rtt_ms=25.0, simulated_jitter_ms=2.5, simulated_throughput_mbps=80.0)
        res = probe.measure_link("127.0.0.1", 8000)
        assert res.rtt_ms.value == 25.0
        assert res.rtt_ms.source == DataSource.MEASURED
        assert res.throughput_mbps.value == 80.0
        assert res.connection_state == "connected"

    def test_one_way_latency_always_unavailable_without_clock_sync(self):
        """Invariant: Mark one-way latency unavailable unless clock sync supports it."""
        probe = MockNetworkProbe()
        res = probe.measure_rtt("127.0.0.1", 8000)
        assert res.one_way_latency_ms.source == DataSource.UNAVAILABLE
        assert res.one_way_latency_ms.value is None

    def test_socket_probe_handles_connection_refusal_gracefully(self):
        """Probing a closed port must report failure cleanly without crashing."""
        probe = SocketNetworkProbe()
        # Find unused closed port on localhost
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        closed_port = s.getsockname()[1]
        s.close()

        res = probe.measure_rtt("127.0.0.1", closed_port, count=2, timeout_s=0.2)
        assert res.rtt_ms.source == DataSource.UNAVAILABLE
        assert res.packet_loss.value == 1.0
        assert res.connection_state in ("refused", "timeout", "unreachable")

    def test_snapshot_conversion_preserves_provenance(self):
        probe = MockNetworkProbe(simulated_rtt_ms=12.0)
        res = probe.measure_rtt("127.0.0.1", 8000)
        snap = res.to_network_snapshot(scenario_label="probe_test")
        assert snap.rtt_ms.value == 12.0
        assert snap.rtt_ms.source == DataSource.MEASURED
        assert snap.scenario_label == "probe_test"


# ===========================================================================
# 5. Safe TCP Transport & Protocol Integrity Tests
# ===========================================================================

class TestSafeTCPTransport:
    def test_safe_tensor_serialization_no_pickle(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        meta, raw = serialize_tensor_safe(t)
        assert meta["shape"] == [2, 2]
        assert meta["dtype"] == "float32"
        assert len(raw) == 2 * 2 * 4  # 16 bytes

        t_restored = deserialize_tensor_safe(meta, raw, target_device="cpu")
        assert torch.equal(t, t_restored)

    def test_safe_module_serialization(self):
        linear = nn.Linear(4, 2)
        meta, raw = serialize_module_safe(linear)
        assert "params" in meta

        target = nn.Linear(4, 2)
        deserialize_module_safe(target, meta, raw)
        for p1, p2 in zip(linear.parameters(), target.parameters()):
            assert torch.equal(p1, p2)

    def test_real_framed_tcp_transfer_loopback(self):
        auth_key = b"secret_unit_test_token"
        server = TransferServer(host="127.0.0.1", port=0, auth_key=auth_key)
        port = server.start()

        client = TCPNetworkTransferProvider(
            endpoint_host="127.0.0.1",
            endpoint_port=port,
            auth_key=auth_key,
            timeout_seconds=5.0,
        )

        test_t = torch.randn(8, 16, dtype=torch.float32)
        try:
            moved, wire_bytes, dur_ms, prov = client.transfer_tensor(
                tensor=test_t,
                source_tier=TierId.USER_DEVICE,
                destination_tier=TierId.EDGE_A,
            )
            assert prov == DataSource.MEASURED
            assert wire_bytes > 8 * 16 * 4
            assert dur_ms >= 0.0
            assert torch.equal(test_t, moved)
            assert server.received_frames_count == 1
        finally:
            server.stop()

    def test_tcp_transfer_hmac_authentication_rejection(self):
        server = TransferServer(host="127.0.0.1", port=0, auth_key=b"correct_secret")
        port = server.start()

        client = TCPNetworkTransferProvider(
            endpoint_host="127.0.0.1",
            endpoint_port=port,
            auth_key=b"wrong_secret",  # Attacker / unauthorized token
            timeout_seconds=2.0,
        )

        test_t = torch.randn(4, 4)
        try:
            with pytest.raises(TransferError):
                client.transfer_tensor(test_t, TierId.USER_DEVICE, TierId.EDGE_A)
        finally:
            server.stop()

    def test_tcp_transfer_payload_too_large_rejection(self):
        server = TransferServer(host="127.0.0.1", port=0, max_payload_bytes=100)  # Tiny 100 byte limit
        port = server.start()

        client = TCPNetworkTransferProvider(
            endpoint_host="127.0.0.1",
            endpoint_port=port,
            timeout_seconds=2.0,
        )

        # 4 * 64 * 4 = 1024 bytes > 100 bytes
        test_t = torch.randn(4, 64, dtype=torch.float32)
        try:
            with pytest.raises((PayloadTooLargeError, TransferError)):
                client.transfer_tensor(test_t, TierId.USER_DEVICE, TierId.EDGE_A)
        finally:
            server.stop()

    def test_tcp_transfer_cancellation(self):
        cancel_evt = threading.Event()
        cancel_evt.set()  # Cancelled immediately

        client = TCPNetworkTransferProvider(endpoint_host="127.0.0.1", endpoint_port=8000)
        test_t = torch.randn(2, 2)
        with pytest.raises(TransferError, match="cancelled"):
            client.transfer_tensor(test_t, TierId.USER_DEVICE, TierId.EDGE_A, cancel_event=cancel_evt)


# ===========================================================================
# 6. Network Profiles & Linux Netem Safety
# ===========================================================================

class TestNetworkProfilesAndNetem:
    def test_all_ten_suggested_profiles_exist(self):
        catalog = NetworkProfileCatalog()
        required_profiles = [
            "ideal_lan",
            "stable_wifi",
            "congested_wifi",
            "mobile_high_bandwidth",
            "mobile_low_bandwidth",
            "high_rtt_backhaul",
            "intermittent_connectivity",
            "bandwidth_degradation",
            "bandwidth_recovery",
            "oscillating_link",
        ]
        for p_id in required_profiles:
            p = catalog.get(p_id)
            assert p is not None, f"Profile {p_id} missing from catalog"
            assert p.bandwidth_mbps > 0
            assert p.rtt_ms > 0

    def test_profile_provenance_separation(self):
        catalog = NetworkProfileCatalog()
        p = catalog.get("six_g_sub_thz_extreme_bandwidth")
        assert p is not None
        assert p.provenance == "illustrative"

    def test_netem_refuses_to_modify_system_without_opt_in(self):
        """Invariant: Never change network settings during ordinary unit tests."""
        adapter = LinuxNetemAdapter(interface="lo", opt_in=False)
        profile = NetworkProfileCatalog().get("stable_wifi")
        # In non-dry-run mode without opt_in, must reject
        res = adapter.apply_profile(profile, dry_run=True)
        assert res.dry_run is True
        assert "tc qdisc replace" in res.applied_command

        with pytest.raises(RuntimeError, match="explicit user opt-in"):
            adapter.apply_profile(profile, dry_run=False)

    def test_netem_cleanup_guidance(self):
        adapter = LinuxNetemAdapter(interface="eth1")
        instructions = adapter.get_cleanup_instructions()
        assert "sudo tc qdisc del dev eth1 root" in instructions


# ===========================================================================
# 7. Memory Validation Engine Tests
# ===========================================================================

class TestMemoryValidationEngine:
    def test_memory_validation_run(self):
        provider = MockCUDAHardwareProvider(
            total_vram_bytes=8 * 1024 * 1024 * 1024,
            initial_allocated_bytes=1024 * 1024 * 1024,
        )
        engine = MemoryValidationEngine(provider)
        report = engine.evaluate_generation_run(num_tokens=8, bytes_per_token_kv=4096)

        assert report.num_steps == 8
        assert report.peak_allocated_mb > 0
        assert report.mean_sampling_overhead_ms >= 0.0
        assert len(report.step_records) == 8
        assert "vram_not_substituted_with_host_ram" in report.invariants_verified


# ===========================================================================
# 8. 6G Evaluation Scenarios & Runtime Integration
# ===========================================================================

class TestRuntimeIntegrationWithModule11:
    def test_six_g_scenarios_registered(self):
        for name in SIX_G_SCENARIOS.keys():
            scen = ScenarioRegistry.get(name)
            assert scen is not None
            assert len(scen.description) > 0

    def test_closed_loop_runtime_with_six_g_scenario(self):
        cfg = OrchestrationConfig(
            mode=OrchestrationMode.PREDICTIVE,
            scenario_name="six_g_high_bandwidth_low_rtt",
            control_interval_tokens=2,
            horizon=2,
        )
        rt = ClosedLoopRuntime.create_default(config=cfg, num_layers=4, hidden_size=64)
        trace = rt.run_simulation(num_steps=6, scenario_name="six_g_high_bandwidth_low_rtt")
        assert trace is not None
        assert len(trace.token_records) == 6
        assert trace.summary is not None
        assert trace.summary.total_switches >= 0

    def test_migration_manager_accepts_tcp_transfer_provider(self):
        from src.migration.adapter import RuntimeAdapter
        from src.runtime.executor import DistributedInferenceExecutor
        from src.runtime.model import LayeredTransformer

        model = LayeredTransformer.create_synthetic(num_layers=4, hidden_size=64)
        executor = DistributedInferenceExecutor(model=model)
        adapter = RuntimeAdapter(executor=executor)

        provider = TCPNetworkTransferProvider(endpoint_host="127.0.0.1", endpoint_port=8080)
        mgr = MigrationManager(runtime_adapter=adapter, transfer_provider=provider)
        assert mgr.transfer_provider is provider
