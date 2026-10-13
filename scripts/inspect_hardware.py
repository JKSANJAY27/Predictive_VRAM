"""
Hardware Inspection Utility for Module 11.

Discovers local hardware capabilities, queries memory allocations,
inspects CPU/GPU utilization, and exports machine-readable telemetry summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware.capabilities import DeviceCapabilityRegistry
from src.hardware.factory import HardwareProviderFactory
from src.runtime.tier import TierId


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local hardware and telemetry capabilities.")
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda", "mock"], default="auto", help="Hardware provider type.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for JSON export.")
    args = parser.parse_args()

    print("=" * 80)
    print(" HARDWARE & TELEMETRY INSPECTION REPORT (MODULE 11)")
    print("=" * 80)

    provider = HardwareProviderFactory.create_provider(args.provider)
    print(f"Provider Type    : {provider.device_type.value}")
    print(f"Driver Available : {provider.is_available}")

    # Discover devices
    devices = provider.discover_devices()
    print(f"\nDiscovered Devices ({len(devices)}):")
    for d in devices:
        tot_mb = f"{d.total_memory_bytes / (1024*1024):.1f} MB" if d.total_memory_bytes is not None else "Unknown"
        print(f"  [{d.device_index}] {d.name} | Total Memory: {tot_mb} | Compute Cap: {d.compute_capability}")

    # Memory telemetry sample
    mem = provider.get_memory_telemetry()
    print("\nMemory Telemetry:")
    vram_tot = f"{mem.total_vram_bytes.value / (1024*1024):.1f} MB" if mem.total_vram_bytes.value is not None else "None"
    vram_alloc = f"{mem.allocated_vram_bytes.value / (1024*1024):.1f} MB" if mem.allocated_vram_bytes.value is not None else "None"
    vram_res = f"{mem.reserved_vram_bytes.value / (1024*1024):.1f} MB" if mem.reserved_vram_bytes.value is not None else "None"
    vram_free = f"{mem.free_vram_bytes.value / (1024*1024):.1f} MB" if mem.free_vram_bytes.value is not None else "None"

    ram_tot = f"{mem.host_ram_total_bytes.value / (1024*1024):.1f} MB" if mem.host_ram_total_bytes.value is not None else "None"
    ram_used = f"{mem.host_ram_used_bytes.value / (1024*1024):.1f} MB" if mem.host_ram_used_bytes.value is not None else "None"

    print(f"  VRAM Total     : {vram_tot} (source={mem.total_vram_bytes.source.value})")
    print(f"  VRAM Allocated : {vram_alloc} (source={mem.allocated_vram_bytes.source.value})")
    print(f"  VRAM Reserved  : {vram_res} (source={mem.reserved_vram_bytes.source.value})")
    print(f"  VRAM Free      : {vram_free} (source={mem.free_vram_bytes.source.value})")
    print(f"  Host RAM Total : {ram_tot} (source={mem.host_ram_total_bytes.source.value})")
    print(f"  Host RAM Used  : {ram_used} (source={mem.host_ram_used_bytes.source.value})")
    print(f"  Sampling Time  : {mem.collection_duration_ms:.3f} ms")

    # Compute telemetry sample
    comp = provider.get_compute_telemetry()
    print("\nCompute Telemetry:")
    cpu_pct = f"{comp.cpu_utilization_pct.value:.1f} %" if comp.cpu_utilization_pct.value is not None else "None"
    gpu_pct = f"{comp.gpu_utilization_pct.value:.1f} %" if comp.gpu_utilization_pct.value is not None else "None"
    print(f"  CPU Utilization: {cpu_pct} (source={comp.cpu_utilization_pct.source.value})")
    print(f"  GPU Utilization: {gpu_pct} (source={comp.gpu_utilization_pct.source.value})")
    print(f"  CPU Cores Log. : {comp.cpu_count_logical.value}")

    # Register into CapabilityRegistry
    registry = DeviceCapabilityRegistry()
    caps = registry.discover_tier(TierId.USER_DEVICE, provider)
    print("\nRegistered Tier Capabilities:")
    print(f"  Tier ID        : {caps.tier_id.value}")
    print(f"  Dtypes         : {caps.supported_dtypes} (source={caps.dtypes_provenance.value})")

    report = {
        "provider": provider.device_type.value,
        "is_available": provider.is_available,
        "devices": [d.to_dict() for d in devices],
        "memory": mem.to_dict(),
        "compute": comp.to_dict(),
        "capabilities": caps.to_dict(),
    }

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[INFO] Saved report to {out_p.resolve()}")

    print("=" * 80)


if __name__ == "__main__":
    main()
