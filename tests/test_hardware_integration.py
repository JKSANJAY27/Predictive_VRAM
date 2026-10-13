"""
Optional Hardware and Testbed Integration Tests for Module 11.

These tests validate real physical execution against:
- Physical NVIDIA CUDA hardware (when attached and drivers are functional)
- Live network sockets and inter-process endpoints

Requirements:
- Tests must be cleanly skipped with explicit reasons when hardware/testbed prerequisites are absent.
- Skipped tests are NEVER reported as passed hardware validations.
"""

import socket
import pytest
import torch

from src.hardware.cuda import CUDAHardwareProvider
from src.hardware.network_measurement import SocketNetworkProbe
from src.hardware.transport import TCPNetworkTransferProvider, TransferServer
from src.runtime.tier import TierId
from src.telemetry.types import DataSource


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Hardware prerequisite unavailable: Physical CUDA GPU not detected.")
class TestPhysicalCUDAIntegration:
    """Tests executing against physical NVIDIA CUDA devices."""

    def test_real_cuda_memory_sampling(self):
        provider = CUDAHardwareProvider(strict=True)
        devices = provider.discover_devices()
        assert len(devices) > 0, "No CUDA devices discovered despite torch.cuda.is_available()=True"

        mem = provider.get_memory_telemetry(device_id=0)
        assert mem.total_vram_bytes.source == DataSource.MEASURED
        assert mem.total_vram_bytes.value > 0
        assert mem.free_vram_bytes.source == DataSource.MEASURED

    def test_gpu_memory_growth_during_tensor_allocation(self):
        provider = CUDAHardwareProvider(strict=True)
        mem_before = provider.get_memory_telemetry(device_id=0)

        # Allocate 16 MB tensor on GPU
        t_gpu = torch.empty((1024, 1024, 4), dtype=torch.float32, device="cuda:0")
        mem_during = provider.get_memory_telemetry(device_id=0)

        assert mem_during.allocated_vram_bytes.value > mem_before.allocated_vram_bytes.value

        del t_gpu
        torch.cuda.empty_cache()


class TestLocalLoopbackNetworkIntegration:
    """Validates real socket communication over loopback."""

    def test_real_socket_rtt_and_throughput(self):
        server = TransferServer(host="127.0.0.1", port=0)
        port = server.start()
        try:
            probe = SocketNetworkProbe()
            res = probe.measure_rtt("127.0.0.1", port, count=3, timeout_s=1.0)
            assert res.rtt_ms.source == DataSource.MEASURED
            assert res.rtt_ms.value is not None
            assert res.connection_state == "connected"
        finally:
            server.stop()

    def test_real_tcp_transfer_and_integrity_verification(self):
        auth_token = b"integration_test_secret"
        server = TransferServer(host="127.0.0.1", port=0, auth_key=auth_token)
        port = server.start()

        client = TCPNetworkTransferProvider(
            endpoint_host="127.0.0.1",
            endpoint_port=port,
            auth_key=auth_token,
        )

        test_tensor = torch.randn(128, 128, dtype=torch.float32)
        try:
            moved, wire_bytes, dur_ms, prov = client.transfer_tensor(
                tensor=test_tensor,
                source_tier=TierId.USER_DEVICE,
                destination_tier=TierId.EDGE_A,
            )
            assert prov == DataSource.MEASURED
            assert torch.equal(test_tensor, moved)
            assert server.received_frames_count == 1
        finally:
            server.stop()
