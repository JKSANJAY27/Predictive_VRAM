"""
Module 11: Real-Hardware Integration, Network Testbeds & 6G-Oriented Validation.

Provides hardware-agnostic telemetry, device capability discovery, real TCP
transport, real network measurement probes, configurable network profiles,
and 6G-oriented evaluation scenarios.
"""

from src.hardware.capabilities import DeviceCapabilityRegistry, TierCapabilities
from src.hardware.cpu import CPUHardwareProvider
from src.hardware.cuda import CUDAHardwareProvider, MockCUDAHardwareProvider
from src.hardware.evaluation_scenarios import SIX_G_SCENARIOS, register_six_g_scenarios
from src.hardware.factory import HardwareProviderFactory
from src.hardware.linux_netem import (
    LinuxNetemAdapter,
    NetemCommandResult,
    NetemSupportStatus,
)
from src.hardware.memory_validation import (
    MemoryStepRecord,
    MemoryValidationEngine,
    MemoryValidationReport,
)
from src.hardware.network_measurement import (
    MockNetworkProbe,
    NetworkMeasurementResult,
    NetworkProbe,
    SocketNetworkProbe,
)
from src.hardware.network_profiles import NetworkProfile, NetworkProfileCatalog
from src.hardware.provider import HardwareTelemetryProvider
from src.hardware.transport import (
    AuthenticationError,
    IntegrityError,
    PayloadTooLargeError,
    TCPNetworkTransferProvider,
    TransferError,
    TransferServer,
    deserialize_module_safe,
    deserialize_tensor_safe,
    serialize_module_safe,
    serialize_tensor_safe,
)
from src.hardware.types import (
    DeviceDescriptor,
    DeviceType,
    EndpointMetadata,
    HardwareComputeTelemetry,
    HardwareMemoryTelemetry,
    HardwareSnapshot,
    MeasurementRecord,
    TelemetryCapabilities,
)

__all__ = [
    # Types
    "DeviceType",
    "MeasurementRecord",
    "HardwareMemoryTelemetry",
    "HardwareComputeTelemetry",
    "HardwareSnapshot",
    "DeviceDescriptor",
    "TelemetryCapabilities",
    "EndpointMetadata",
    # Providers
    "HardwareTelemetryProvider",
    "CPUHardwareProvider",
    "CUDAHardwareProvider",
    "MockCUDAHardwareProvider",
    "HardwareProviderFactory",
    # Capabilities
    "TierCapabilities",
    "DeviceCapabilityRegistry",
    # Network Probing
    "NetworkMeasurementResult",
    "NetworkProbe",
    "SocketNetworkProbe",
    "MockNetworkProbe",
    # Network Transport
    "TCPNetworkTransferProvider",
    "TransferServer",
    "TransferError",
    "AuthenticationError",
    "IntegrityError",
    "PayloadTooLargeError",
    "serialize_tensor_safe",
    "deserialize_tensor_safe",
    "serialize_module_safe",
    "deserialize_module_safe",
    # Network Profiles
    "NetworkProfile",
    "NetworkProfileCatalog",
    # Linux Netem
    "LinuxNetemAdapter",
    "NetemSupportStatus",
    "NetemCommandResult",
    # Memory Validation
    "MemoryValidationEngine",
    "MemoryValidationReport",
    "MemoryStepRecord",
    # Scenarios
    "SIX_G_SCENARIOS",
    "register_six_g_scenarios",
]
