"""
Device Capability Registry for Module 11.

Maintains structured, provenance-aware capability records across logical and physical tiers.
Separates observed hardware metrics from operator-configured estimates.
Enforces the invariant that unknown capacity is NEVER interpreted as unlimited capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.hardware.provider import HardwareTelemetryProvider
from src.hardware.types import (
    DeviceDescriptor,
    DeviceType,
    EndpointMetadata,
    TelemetryCapabilities,
)
from src.runtime.tier import TierId
from src.telemetry.types import DataSource


@dataclass(frozen=True)
class TierCapabilities:
    """
    Capability record for an execution tier.

    Differentiates directly observed capabilities from operator-configured estimates.
    """
    tier_id: TierId
    device_type: DeviceType
    device_name: Optional[str]
    total_memory_bytes: Optional[int]
    total_memory_provenance: DataSource
    available_memory_bytes: Optional[int]
    available_memory_provenance: DataSource
    supported_dtypes: Tuple[str, ...]
    dtypes_provenance: DataSource
    compute_capability: Optional[Tuple[int, int]] = None
    telemetry_capabilities: TelemetryCapabilities = field(default_factory=TelemetryCapabilities)
    network_endpoint: Optional[EndpointMetadata] = None
    is_observed: bool = True
    operator_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id.value,
            "device_type": self.device_type.value,
            "device_name": self.device_name,
            "total_memory_bytes": self.total_memory_bytes,
            "total_memory_provenance": self.total_memory_provenance.value,
            "available_memory_bytes": self.available_memory_bytes,
            "available_memory_provenance": self.available_memory_provenance.value,
            "supported_dtypes": list(self.supported_dtypes),
            "dtypes_provenance": self.dtypes_provenance.value,
            "compute_capability": list(self.compute_capability) if self.compute_capability else None,
            "telemetry_capabilities": self.telemetry_capabilities.to_dict(),
            "network_endpoint": self.network_endpoint.to_dict() if self.network_endpoint else None,
            "is_observed": self.is_observed,
            "operator_notes": self.operator_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TierCapabilities:
        ep_data = data.get("network_endpoint")
        endpoint = EndpointMetadata(**ep_data) if ep_data else None
        telemetry_cap = TelemetryCapabilities(**data.get("telemetry_capabilities", {}))
        comp_cap = tuple(data["compute_capability"]) if data.get("compute_capability") else None

        return cls(
            tier_id=TierId(data["tier_id"]),
            device_type=DeviceType(data["device_type"]),
            device_name=data.get("device_name"),
            total_memory_bytes=data.get("total_memory_bytes"),
            total_memory_provenance=DataSource(data["total_memory_provenance"]),
            available_memory_bytes=data.get("available_memory_bytes"),
            available_memory_provenance=DataSource(data["available_memory_provenance"]),
            supported_dtypes=tuple(data.get("supported_dtypes", ["float32"])),
            dtypes_provenance=DataSource(data.get("dtypes_provenance", "estimated")),
            compute_capability=comp_cap,
            telemetry_capabilities=telemetry_cap,
            network_endpoint=endpoint,
            is_observed=bool(data.get("is_observed", True)),
            operator_notes=data.get("operator_notes", ""),
        )


class DeviceCapabilityRegistry:
    """
    Registry describing execution and telemetry capabilities for each tier.

    Invariants:
    - Unknown memory capacity is NEVER interpreted as unlimited capacity.
    - Operator-configured estimates carry DataSource.ESTIMATED.
    - Hardware-discovered capabilities carry DataSource.MEASURED.
    """

    def __init__(self) -> None:
        self._tiers: Dict[TierId, TierCapabilities] = {}

    def register(self, capabilities: TierCapabilities) -> None:
        """Register or update capabilities for a tier."""
        self._tiers[capabilities.tier_id] = capabilities

    def get(self, tier_id: TierId) -> Optional[TierCapabilities]:
        """Retrieve capability record for a tier."""
        return self._tiers.get(tier_id)

    def has_tier(self, tier_id: TierId) -> bool:
        """Check if a tier is registered."""
        return tier_id in self._tiers

    def list_tiers(self) -> List[TierId]:
        """List all registered tier identifiers."""
        return list(self._tiers.keys())

    def discover_tier(
        self,
        tier_id: TierId,
        provider: HardwareTelemetryProvider,
        device_index: Optional[int] = None,
        endpoint: Optional[EndpointMetadata] = None,
        operator_notes: str = "",
    ) -> TierCapabilities:
        """
        Discover and register tier capabilities directly from a HardwareTelemetryProvider.
        """
        devices = provider.discover_devices()
        dev_desc: Optional[DeviceDescriptor] = None
        if devices:
            target_idx = device_index if device_index is not None else 0
            dev_desc = next((d for d in devices if d.device_index == target_idx), devices[0])

        mem = provider.get_memory_telemetry(device_index)
        telemetry_caps = provider.get_telemetry_capabilities(device_index)

        # Determine total and available memory based on device type
        if provider.device_type == DeviceType.CPU:
            total_b = int(mem.host_ram_total_bytes.value) if mem.host_ram_total_bytes.value is not None else None
            total_prov = mem.host_ram_total_bytes.source
            avail_b = int(mem.host_ram_available_bytes.value) if mem.host_ram_available_bytes.value is not None else None
            avail_prov = mem.host_ram_available_bytes.source
            supported_dtypes = ("float32",)
        else:
            total_b = int(mem.total_vram_bytes.value) if mem.total_vram_bytes.value is not None else None
            total_prov = mem.total_vram_bytes.source
            avail_b = int(mem.free_vram_bytes.value) if mem.free_vram_bytes.value is not None else None
            avail_prov = mem.free_vram_bytes.source
            supported_dtypes = ("float32", "float16", "bfloat16")

        caps = TierCapabilities(
            tier_id=tier_id,
            device_type=provider.device_type,
            device_name=dev_desc.name if dev_desc else "Unknown",
            total_memory_bytes=total_b,
            total_memory_provenance=total_prov,
            available_memory_bytes=avail_b,
            available_memory_provenance=avail_prov,
            supported_dtypes=supported_dtypes,
            dtypes_provenance=DataSource.MEASURED if provider.is_available else DataSource.ESTIMATED,
            compute_capability=dev_desc.compute_capability if dev_desc else None,
            telemetry_capabilities=telemetry_caps,
            network_endpoint=endpoint,
            is_observed=True,
            operator_notes=operator_notes,
        )
        self.register(caps)
        return caps

    def configure_operator_tier(
        self,
        tier_id: TierId,
        device_type: DeviceType,
        device_name: str,
        total_memory_bytes: Optional[int],
        available_memory_bytes: Optional[int] = None,
        supported_dtypes: Tuple[str, ...] = ("float32",),
        compute_capability: Optional[Tuple[int, int]] = None,
        telemetry_capabilities: Optional[TelemetryCapabilities] = None,
        network_endpoint: Optional[EndpointMetadata] = None,
        operator_notes: str = "Operator-configured estimate",
    ) -> TierCapabilities:
        """
        Register operator-configured estimates for remote or future hardware tiers.
        """
        caps = TierCapabilities(
            tier_id=tier_id,
            device_type=device_type,
            device_name=device_name,
            total_memory_bytes=total_memory_bytes,
            total_memory_provenance=DataSource.ESTIMATED,
            available_memory_bytes=available_memory_bytes,
            available_memory_provenance=DataSource.ESTIMATED,
            supported_dtypes=supported_dtypes,
            dtypes_provenance=DataSource.ESTIMATED,
            compute_capability=compute_capability,
            telemetry_capabilities=telemetry_capabilities or TelemetryCapabilities(),
            network_endpoint=network_endpoint,
            is_observed=False,
            operator_notes=operator_notes,
        )
        self.register(caps)
        return caps

    def validate_capacity(self, tier_id: TierId, required_bytes: int) -> Tuple[bool, str]:
        """
        Check if the specified tier has sufficient verified memory capacity.

        Invariant: Unknown capacity is NEVER treated as unlimited capacity.
        """
        caps = self.get(tier_id)
        if caps is None:
            return False, f"Tier '{tier_id}' is not registered in capability registry"

        if caps.total_memory_bytes is None:
            return (
                False,
                f"Tier '{tier_id}' total memory capacity is unknown; "
                f"cannot assume unlimited capacity (Research Invariant).",
            )

        if caps.available_memory_bytes is not None:
            if caps.available_memory_bytes < required_bytes:
                return (
                    False,
                    f"Tier '{tier_id}' available memory ({caps.available_memory_bytes} B) "
                    f"is less than required ({required_bytes} B).",
                )
        elif caps.total_memory_bytes < required_bytes:
            return (
                False,
                f"Tier '{tier_id}' total memory ({caps.total_memory_bytes} B) "
                f"is less than required ({required_bytes} B).",
            )

        return True, "Capacity verified"

    def to_dict(self) -> Dict[str, Any]:
        return {tier.value: caps.to_dict() for tier, caps in self._tiers.items()}
