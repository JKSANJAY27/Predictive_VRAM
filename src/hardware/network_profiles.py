"""
Network Profile Catalog for Module 11.

Defines standardized, configurable network profiles representing edge inference operating conditions,
including LAN, Wi-Fi, cellular, degrading links, and 6G-oriented exploratory conditions.

Research Invariant:
- Profiles clearly distinguish illustrative configurations from calibrated or measured replay traces.
- Illustrative values are NEVER presented as authoritative 6G hardware specifications.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.telemetry.network_emulation import NetworkEmulationState


@dataclass(frozen=True)
class NetworkProfile:
    """
    Specification for a network operating condition profile.

    Attributes:
        profile_id: Canonical identifier.
        name: Human-readable title.
        description: Technical summary and research motivation.
        bandwidth_mbps: Configured or mean bandwidth rate.
        bandwidth_min_mbps: Lower bound for bandwidth variations.
        bandwidth_max_mbps: Upper bound for bandwidth variations.
        rtt_ms: Configured or mean round-trip latency.
        rtt_min_ms: Lower bound for RTT.
        rtt_max_ms: Upper bound for RTT.
        jitter_ms: Standard deviation of round-trip latency.
        packet_loss_rate: Fractional packet loss in [0.0, 1.0].
        outage_probability: Probability of link drop per sample step.
        outage_duration_s: Duration of outage when triggered.
        random_seed: Random seed for deterministic simulation.
        provenance: 'illustrative', 'calibrated', or 'measured_replay'.
        source_reference: Literature citation or testbed provenance.
        is_empirical_trace: True if derived from physical measurement trace.
    """
    profile_id: str
    name: str
    description: str
    bandwidth_mbps: float
    bandwidth_min_mbps: float
    bandwidth_max_mbps: float
    rtt_ms: float
    rtt_min_ms: float
    rtt_max_ms: float
    jitter_ms: float
    packet_loss_rate: float
    outage_probability: float = 0.0
    outage_duration_s: float = 0.0
    random_seed: int = 42
    provenance: str = "illustrative"
    source_reference: str = ""
    is_empirical_trace: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "bandwidth_mbps": self.bandwidth_mbps,
            "bandwidth_min_mbps": self.bandwidth_min_mbps,
            "bandwidth_max_mbps": self.bandwidth_max_mbps,
            "rtt_ms": self.rtt_ms,
            "rtt_min_ms": self.rtt_min_ms,
            "rtt_max_ms": self.rtt_max_ms,
            "jitter_ms": self.jitter_ms,
            "packet_loss_rate": self.packet_loss_rate,
            "outage_probability": self.outage_probability,
            "outage_duration_s": self.outage_duration_s,
            "random_seed": self.random_seed,
            "provenance": self.provenance,
            "source_reference": self.source_reference,
            "is_empirical_trace": self.is_empirical_trace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NetworkProfile:
        return cls(**data)

    def to_network_emulation_state(self) -> NetworkEmulationState:
        """Create a Module 2 NetworkEmulationState configured with this profile's base values."""
        return NetworkEmulationState(
            rtt_ms=self.rtt_ms,
            bandwidth_mbps=self.bandwidth_mbps,
            packet_loss_rate=self.packet_loss_rate,
            jitter_ms=self.jitter_ms,
            scenario_label=self.profile_id,
        )

    def sample_step_conditions(self, step: int, total_steps: int) -> Dict[str, float]:
        """
        Compute step-specific conditions accounting for dynamics (oscillations, degradation, outages).
        """
        rng = random.Random(self.random_seed + step)
        bw = self.bandwidth_mbps
        rtt = self.rtt_ms
        loss = self.packet_loss_rate
        progress = max(0.0, min(1.0, step / max(1, total_steps - 1)))

        # Dynamic profile adjustments
        if self.profile_id == "bandwidth_degradation":
            # Linear ramp from max to min
            bw = self.bandwidth_max_mbps - progress * (self.bandwidth_max_mbps - self.bandwidth_min_mbps)
            rtt = self.rtt_min_ms + progress * (self.rtt_max_ms - self.rtt_min_ms)
        elif self.profile_id == "bandwidth_recovery":
            # Linear ramp from min to max
            bw = self.bandwidth_min_mbps + progress * (self.bandwidth_max_mbps - self.bandwidth_min_mbps)
            rtt = self.rtt_max_ms - progress * (self.rtt_max_ms - self.rtt_min_ms)
        elif self.profile_id == "oscillating_link":
            # Sinusoidal oscillation
            period = max(4, total_steps // 2)
            sine_val = (math.sin(2 * math.pi * step / period) + 1.0) / 2.0
            bw = self.bandwidth_min_mbps + sine_val * (self.bandwidth_max_mbps - self.bandwidth_min_mbps)
            rtt = self.rtt_min_ms + (1.0 - sine_val) * (self.rtt_max_ms - self.rtt_min_ms)
        elif self.profile_id == "intermittent_connectivity":
            # Check for simulated outage
            if rng.random() < self.outage_probability:
                bw = 0.1
                rtt = self.rtt_max_ms * 2.0
                loss = 1.0

        # Add Gaussian jitter
        rtt_sample = max(0.1, rtt + rng.gauss(0.0, self.jitter_ms))
        bw_sample = max(0.1, bw)

        return {
            "bandwidth_mbps": round(bw_sample, 3),
            "rtt_ms": round(rtt_sample, 3),
            "packet_loss": round(loss, 4),
            "jitter_ms": round(self.jitter_ms, 3),
        }


# ---------------------------------------------------------------------------
# Canonical Profile Catalog
# ---------------------------------------------------------------------------

_CANONICAL_PROFILES: Dict[str, NetworkProfile] = {
    "ideal_lan": NetworkProfile(
        profile_id="ideal_lan",
        name="Ideal Local Area Network",
        description="High-speed wired Gigabit Ethernet (1000 Mbps, 1.0 ms RTT) representative of on-premise compute cluster.",
        bandwidth_mbps=1000.0,
        bandwidth_min_mbps=900.0,
        bandwidth_max_mbps=1000.0,
        rtt_ms=1.0,
        rtt_min_ms=0.8,
        rtt_max_ms=1.5,
        jitter_ms=0.1,
        packet_loss_rate=0.0,
        provenance="illustrative",
        source_reference="IEEE 802.3ab baseline",
    ),
    "stable_wifi": NetworkProfile(
        profile_id="stable_wifi",
        name="Stable Wi-Fi 6 (802.11ax)",
        description="Low-interference enterprise/home wireless link (150 Mbps, 15 ms RTT).",
        bandwidth_mbps=150.0,
        bandwidth_min_mbps=120.0,
        bandwidth_max_mbps=180.0,
        rtt_ms=15.0,
        rtt_min_ms=10.0,
        rtt_max_ms=25.0,
        jitter_ms=2.0,
        packet_loss_rate=0.001,
        provenance="illustrative",
        source_reference="IEEE 802.11ax edge measurement guidelines",
    ),
    "congested_wifi": NetworkProfile(
        profile_id="congested_wifi",
        name="Congested Multi-User Wi-Fi",
        description="High contention wireless channel with contention queuing and bufferbloat (25 Mbps, 60 ms RTT, 18 ms jitter).",
        bandwidth_mbps=25.0,
        bandwidth_min_mbps=8.0,
        bandwidth_max_mbps=40.0,
        rtt_ms=60.0,
        rtt_min_ms=30.0,
        rtt_max_ms=120.0,
        jitter_ms=18.0,
        packet_loss_rate=0.03,
        provenance="illustrative",
        source_reference="Empirical campus Wi-Fi contention trace",
    ),
    "mobile_high_bandwidth": NetworkProfile(
        profile_id="mobile_high_bandwidth",
        name="Mobile 5G Sub-6GHz eMBB",
        description="Line-of-sight 5G Enhanced Mobile Broadband (300 Mbps, 25 ms RTT).",
        bandwidth_mbps=300.0,
        bandwidth_min_mbps=150.0,
        bandwidth_max_mbps=450.0,
        rtt_ms=25.0,
        rtt_min_ms=18.0,
        rtt_max_ms=40.0,
        jitter_ms=5.0,
        packet_loss_rate=0.005,
        provenance="illustrative",
        source_reference="3GPP TR 38.901 Rel-16 Urban Macro (UMa)",
    ),
    "mobile_low_bandwidth": NetworkProfile(
        profile_id="mobile_low_bandwidth",
        name="Mobile LTE / Edge Coverage",
        description="Cell-edge wireless with limited allocation (15 Mbps, 90 ms RTT).",
        bandwidth_mbps=15.0,
        bandwidth_min_mbps=4.0,
        bandwidth_max_mbps=25.0,
        rtt_ms=90.0,
        rtt_min_ms=60.0,
        rtt_max_ms=150.0,
        jitter_ms=25.0,
        packet_loss_rate=0.02,
        provenance="illustrative",
        source_reference="3GPP TS 36.300 LTE cell-edge specification",
    ),
    "high_rtt_backhaul": NetworkProfile(
        profile_id="high_rtt_backhaul",
        name="High-RTT Edge-to-Cloud Backhaul",
        description="Transcontinental or GEO satellite backhaul link (80 Mbps, 180 ms RTT).",
        bandwidth_mbps=80.0,
        bandwidth_min_mbps=50.0,
        bandwidth_max_mbps=100.0,
        rtt_ms=180.0,
        rtt_min_ms=150.0,
        rtt_max_ms=220.0,
        jitter_ms=12.0,
        packet_loss_rate=0.008,
        provenance="illustrative",
        source_reference="WAN transit / Satellite backhaul benchmark",
    ),
    "intermittent_connectivity": NetworkProfile(
        profile_id="intermittent_connectivity",
        name="Intermittent Mobility Link with Outages",
        description="Channel experiencing shadow fading, mmWave beam blockage, or handovers with periodic disconnects.",
        bandwidth_mbps=50.0,
        bandwidth_min_mbps=0.1,
        bandwidth_max_mbps=100.0,
        rtt_ms=40.0,
        rtt_min_ms=20.0,
        rtt_max_ms=300.0,
        jitter_ms=30.0,
        packet_loss_rate=0.05,
        outage_probability=0.15,
        outage_duration_s=2.0,
        provenance="illustrative",
        source_reference="3GPP TR 38.802 High Mobility & Beam Blockage",
    ),
    "bandwidth_degradation": NetworkProfile(
        profile_id="bandwidth_degradation",
        name="Continuous Bandwidth Degradation",
        description="Dynamic link degradation from 120 Mbps down to 5 Mbps, testing predictive offloading before link exhaustion.",
        bandwidth_mbps=60.0,
        bandwidth_min_mbps=5.0,
        bandwidth_max_mbps=120.0,
        rtt_ms=45.0,
        rtt_min_ms=20.0,
        rtt_max_ms=120.0,
        jitter_ms=10.0,
        packet_loss_rate=0.01,
        provenance="illustrative",
        source_reference="Synthetic stress test scenario",
    ),
    "bandwidth_recovery": NetworkProfile(
        profile_id="bandwidth_recovery",
        name="Bandwidth Recovery",
        description="Dynamic link recovery from 8 Mbps up to 100 Mbps, evaluating opportunistic edge partition expansion.",
        bandwidth_mbps=50.0,
        bandwidth_min_mbps=8.0,
        bandwidth_max_mbps=100.0,
        rtt_ms=40.0,
        rtt_min_ms=15.0,
        rtt_max_ms=110.0,
        jitter_ms=8.0,
        packet_loss_rate=0.005,
        provenance="illustrative",
        source_reference="Synthetic recovery stress test",
    ),
    "oscillating_link": NetworkProfile(
        profile_id="oscillating_link",
        name="Oscillating Channel",
        description="Rapid sinusoidal link variation between 10 Mbps and 80 Mbps testing controller anti-thrashing hysteresis.",
        bandwidth_mbps=45.0,
        bandwidth_min_mbps=10.0,
        bandwidth_max_mbps=80.0,
        rtt_ms=35.0,
        rtt_min_ms=15.0,
        rtt_max_ms=75.0,
        jitter_ms=12.0,
        packet_loss_rate=0.015,
        provenance="illustrative",
        source_reference="Synthetic cyclic channel model",
    ),
    # 6G-Oriented Exploratory Profiles (Clearly tagged illustrative exploratory)
    "six_g_sub_thz_extreme_bandwidth": NetworkProfile(
        profile_id="six_g_sub_thz_extreme_bandwidth",
        name="6G Sub-THz Ultra-High Bandwidth (Exploratory)",
        description="Simulated 6G Sub-THz short-range link (5000 Mbps, 0.8 ms RTT). Illustrative operating condition; not physical 6G hardware measurement.",
        bandwidth_mbps=5000.0,
        bandwidth_min_mbps=2000.0,
        bandwidth_max_mbps=8000.0,
        rtt_ms=0.8,
        rtt_min_ms=0.4,
        rtt_max_ms=1.5,
        jitter_ms=0.05,
        packet_loss_rate=0.0,
        provenance="illustrative",
        source_reference="ITU-R M.2160 / 6G Vision Exploratory Parameters",
    ),
    "six_g_high_mobility_doppler": NetworkProfile(
        profile_id="six_g_high_mobility_doppler",
        name="6G High-Speed Train Doppler Tracking (Exploratory)",
        description="Simulated 6G high-mobility scenario (1200 Mbps with periodic beam reconfiguration jitter). Illustrative operating condition; not physical 6G hardware measurement.",
        bandwidth_mbps=1200.0,
        bandwidth_min_mbps=200.0,
        bandwidth_max_mbps=2500.0,
        rtt_ms=3.0,
        rtt_min_ms=1.2,
        rtt_max_ms=15.0,
        jitter_ms=4.0,
        packet_loss_rate=0.008,
        provenance="illustrative",
        source_reference="ITU-R M.2160 High-Mobility Concept",
    ),
}


class NetworkProfileCatalog:
    """Registry and query interface for edge network profiles."""

    def __init__(self) -> None:
        self._profiles: Dict[str, NetworkProfile] = dict(_CANONICAL_PROFILES)

    def register(self, profile: NetworkProfile) -> None:
        """Register or override a network profile."""
        self._profiles[profile.profile_id] = profile

    def get(self, profile_id: str) -> Optional[NetworkProfile]:
        """Retrieve profile by ID."""
        return self._profiles.get(profile_id)

    def list_profiles(self) -> List[str]:
        """List all available profile identifiers."""
        return list(self._profiles.keys())

    def list_by_provenance(self, provenance: str) -> List[NetworkProfile]:
        """Filter profiles by provenance tag ('illustrative', 'calibrated', 'measured_replay')."""
        return [p for p in self._profiles.values() if p.provenance == provenance]

    def load_trace_file(self, trace_path: str, profile_id: str, name: str) -> NetworkProfile:
        """
        Load an empirical measurement trace file (JSON or CSV) and register it as a measured_replay profile.
        """
        p = Path(trace_path)
        if not p.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

        data = json.loads(p.read_text(encoding="utf-8"))
        profile = NetworkProfile(
            profile_id=profile_id,
            name=name,
            description=f"Empirical trace replayed from {p.name}",
            bandwidth_mbps=float(data.get("mean_bandwidth_mbps", 50.0)),
            bandwidth_min_mbps=float(data.get("min_bandwidth_mbps", 1.0)),
            bandwidth_max_mbps=float(data.get("max_bandwidth_mbps", 100.0)),
            rtt_ms=float(data.get("mean_rtt_ms", 30.0)),
            rtt_min_ms=float(data.get("min_rtt_ms", 10.0)),
            rtt_max_ms=float(data.get("max_rtt_ms", 80.0)),
            jitter_ms=float(data.get("jitter_ms", 5.0)),
            packet_loss_rate=float(data.get("packet_loss_rate", 0.01)),
            provenance="measured_replay",
            source_reference=str(p.resolve()),
            is_empirical_trace=True,
        )
        self.register(profile)
        return profile
