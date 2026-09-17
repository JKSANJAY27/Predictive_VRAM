"""
Network emulation state for the Predictive VRAM split inference system.

Since this project runs entirely on a single development machine with no
real edge nodes, all network conditions are simulated via this injectable
state object. The NetworkConditionCollector reads from this state.

Emulated scenarios allow controlled experiments:
- baseline: high-bandwidth, low-latency (ideal edge)
- degraded: reduced bandwidth and increased RTT (congested link)
- lossy: packet loss scenario
- weak_wifi: realistic home wireless
- mobile_4g: mobile network simulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class NetworkEmulationState:
    """
    Mutable emulation state representing network conditions between tiers.

    All values are injected externally — they do NOT reflect real physical
    measurements. This class is the authoritative source of network metrics
    for the NetworkConditionCollector.

    Attributes:
        rtt_ms: Round-trip time in milliseconds (User Device ↔ Edge A).
        bandwidth_mbps: Uplink bandwidth in Megabits per second.
        packet_loss_rate: Fraction of packets lost [0.0, 1.0].
        jitter_ms: Standard deviation of RTT in milliseconds.
        scenario_label: Human-readable label for this scenario.
    """
    rtt_ms: float = 20.0
    bandwidth_mbps: float = 100.0
    packet_loss_rate: float = 0.0
    jitter_ms: float = 2.0
    scenario_label: str = "baseline"

    def update(self, **kwargs: Any) -> None:
        """
        Update emulation parameters with validation.

        Raises:
            ValueError: If any parameter value is out of range or unknown.
            TypeError: If a value has the wrong type.
        """
        valid_fields = {
            "rtt_ms", "bandwidth_mbps", "packet_loss_rate",
            "jitter_ms", "scenario_label",
        }
        for key, value in kwargs.items():
            if key not in valid_fields:
                raise ValueError(
                    f"Unknown NetworkEmulationState field '{key}'. "
                    f"Valid fields: {sorted(valid_fields)}"
                )
            if key == "scenario_label":
                if not isinstance(value, str):
                    raise TypeError(f"scenario_label must be a str, got {type(value).__name__}")
                self.scenario_label = value
            else:
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"'{key}' must be numeric, got {type(value).__name__}"
                    )
                value = float(value)
                # Range validation
                if key in ("rtt_ms", "bandwidth_mbps", "jitter_ms") and value < 0:
                    raise ValueError(f"'{key}' must be >= 0, got {value}")
                if key == "bandwidth_mbps" and value <= 0:
                    raise ValueError(f"'bandwidth_mbps' must be > 0, got {value}")
                if key == "packet_loss_rate" and not (0.0 <= value <= 1.0):
                    raise ValueError(
                        f"'packet_loss_rate' must be in [0.0, 1.0], got {value}"
                    )
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary."""
        return {
            "rtt_ms": self.rtt_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "packet_loss_rate": self.packet_loss_rate,
            "jitter_ms": self.jitter_ms,
            "scenario_label": self.scenario_label,
        }

    @classmethod
    def preset_scenarios(cls) -> Dict[str, "NetworkEmulationState"]:
        """
        Return a dictionary of named network scenario presets.

        These are used in experiments to simulate varying network conditions
        without changing the inference infrastructure.

        Returns:
            Dict mapping scenario name to a configured NetworkEmulationState.
        """
        return {
            "baseline": cls(
                rtt_ms=20.0,
                bandwidth_mbps=100.0,
                packet_loss_rate=0.0,
                jitter_ms=2.0,
                scenario_label="baseline",
            ),
            "degraded": cls(
                rtt_ms=80.0,
                bandwidth_mbps=10.0,
                packet_loss_rate=0.01,
                jitter_ms=15.0,
                scenario_label="degraded",
            ),
            "lossy": cls(
                rtt_ms=40.0,
                bandwidth_mbps=50.0,
                packet_loss_rate=0.10,
                jitter_ms=8.0,
                scenario_label="lossy",
            ),
            "weak_wifi": cls(
                rtt_ms=35.0,
                bandwidth_mbps=25.0,
                packet_loss_rate=0.02,
                jitter_ms=10.0,
                scenario_label="weak_wifi",
            ),
            "mobile_4g": cls(
                rtt_ms=60.0,
                bandwidth_mbps=15.0,
                packet_loss_rate=0.03,
                jitter_ms=20.0,
                scenario_label="mobile_4g",
            ),
            "ideal": cls(
                rtt_ms=1.0,
                bandwidth_mbps=1000.0,
                packet_loss_rate=0.0,
                jitter_ms=0.1,
                scenario_label="ideal",
            ),
        }

    def estimate_transfer_ms(self, byte_size: int) -> float:
        """
        Estimate one-way transfer latency for a payload of the given size.

        Formula:
            latency_ms = (rtt_ms / 2) + (byte_size * 8) / (bandwidth_mbps * 1e6) * 1000

        This is a first-order estimate: half the RTT plus serialisation delay.
        Always tagged EMULATED by the NetworkConditionCollector.

        Args:
            byte_size: Payload size in bytes.

        Returns:
            Estimated one-way transfer latency in milliseconds.
        """
        serialisation_ms = (byte_size * 8) / (self.bandwidth_mbps * 1e6) * 1000.0
        propagation_ms = self.rtt_ms / 2.0
        return propagation_ms + serialisation_ms
