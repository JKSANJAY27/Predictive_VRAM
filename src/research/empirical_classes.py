"""
Empirical Experiment Classification (Module 13).

Strictly partitions experimental campaigns into five distinct, non-fungible classes:
1. CLASS_1_REAL_GPU          : Real GPU execution with genuine PyTorch CUDA VRAM telemetry.
2. CLASS_2_REAL_SOCKET       : Physical OS network socket transport benchmarking.
3. CLASS_3_NETWORK_EMULATED  : Controlled network link emulation (e.g., netem or discrete steps).
4. CLASS_4_FULLY_SIMULATED   : Synthetic traces with simulated latency and transfer costs.
5. CLASS_5_MIXED             : Hybrid experiments combining physical and emulated components.

Guarantees that results from different classes are never silently aggregated or conflated.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class ExperimentClass(str, enum.Enum):
    """The 5 mutually exclusive scientific evidence classes for Module 13."""
    CLASS_1_REAL_GPU = "CLASS_1_REAL_GPU"
    CLASS_2_REAL_SOCKET = "CLASS_2_REAL_SOCKET"
    CLASS_3_NETWORK_EMULATED = "CLASS_3_NETWORK_EMULATED"
    CLASS_4_FULLY_SIMULATED = "CLASS_4_FULLY_SIMULATED"
    CLASS_5_MIXED = "CLASS_5_MIXED"


class DevicePlacementMode(str, enum.Enum):
    """Classification of how model partitions are physically or logically mapped."""
    PHYSICAL_MULTI_DEVICE = "PHYSICAL_MULTI_DEVICE"
    PHYSICAL_SINGLE_DEVICE_MULTI_STREAM = "PHYSICAL_SINGLE_DEVICE_MULTI_STREAM"
    PHYSICAL_SINGLE_DEVICE_LOGICAL = "PHYSICAL_SINGLE_DEVICE_LOGICAL"
    CPU_PARTITIONED = "CPU_PARTITIONED"
    SIMULATED_DISTRIBUTED = "SIMULATED_DISTRIBUTED"


@dataclass
class EvidenceQualityDescriptor:
    """
    Detailed audit descriptor specifying the physical reality of an experiment.
    """
    experiment_class: ExperimentClass
    device_placement_mode: DevicePlacementMode
    gpu_telemetry_genuine: bool
    network_transport_physical: bool
    network_dynamics_emulated: bool
    inference_execution_genuine: bool
    cpu_only_fallback: bool
    limitations: List[str] = field(default_factory=list)

    @classmethod
    def classify(
        cls,
        cuda_available: bool,
        is_simulation_mode: bool,
        use_real_socket: bool,
        use_emulated_network: bool,
        tier_devices: Optional[Dict[str, str]] = None,
    ) -> EvidenceQualityDescriptor:
        """
        Deterministically classify an execution run into one of the 5 canonical classes.
        """
        limits: List[str] = []

        if is_simulation_mode:
            limits.append("Inference execution latency and token generation simulated via synthetic clocks.")
            return cls(
                experiment_class=ExperimentClass.CLASS_4_FULLY_SIMULATED,
                device_placement_mode=DevicePlacementMode.SIMULATED_DISTRIBUTED,
                gpu_telemetry_genuine=False,
                network_transport_physical=False,
                network_dynamics_emulated=True,
                inference_execution_genuine=False,
                cpu_only_fallback=not cuda_available,
                limitations=limits,
            )

        # Execution mode
        if not cuda_available:
            placement = DevicePlacementMode.CPU_PARTITIONED
            gpu_genuine = False
            limits.append("CUDA hardware unavailable: VRAM telemetry is UNAVAILABLE and CPU host RAM was queried.")
        else:
            # Check devices in tiers
            dev_set = set(tier_devices.values()) if tier_devices else {"cpu"}
            cuda_devs = [d for d in dev_set if "cuda" in str(d).lower()]
            if len(cuda_devs) > 1:
                placement = DevicePlacementMode.PHYSICAL_MULTI_DEVICE
            elif len(cuda_devs) == 1:
                placement = DevicePlacementMode.PHYSICAL_SINGLE_DEVICE_LOGICAL
                limits.append("Single GPU available: partitions mapped to identical physical GPU or CPU-GPU boundary.")
            else:
                placement = DevicePlacementMode.CPU_PARTITIONED
            gpu_genuine = len(cuda_devs) > 0

        # Class determination
        if gpu_genuine and use_real_socket and not use_emulated_network:
            exp_class = ExperimentClass.CLASS_1_REAL_GPU
        elif use_real_socket and not gpu_genuine and not use_emulated_network:
            exp_class = ExperimentClass.CLASS_2_REAL_SOCKET
            limits.append("Physical TCP transport exercised; inference executed on CPU.")
        elif use_emulated_network and not gpu_genuine:
            exp_class = ExperimentClass.CLASS_3_NETWORK_EMULATED
            limits.append("Network conditions shaped or emulated via discrete profiles.")
        else:
            exp_class = ExperimentClass.CLASS_5_MIXED
            limits.append("Hybrid execution combining physical compute with emulated network traces.")

        return cls(
            experiment_class=exp_class,
            device_placement_mode=placement,
            gpu_telemetry_genuine=gpu_genuine,
            network_transport_physical=use_real_socket,
            network_dynamics_emulated=use_emulated_network,
            inference_execution_genuine=True,
            cpu_only_fallback=not cuda_available,
            limitations=limits,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["experiment_class"] = self.experiment_class.value
        d["device_placement_mode"] = self.device_placement_mode.value
        return d
