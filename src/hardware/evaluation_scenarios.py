"""
6G-Oriented Evaluation Scenarios for Module 11.

Defines reproducible operating conditions representing selected edge computing environments
relevant to 6G research questions (extreme bandwidth, ultra-low RTT, mmWave/sub-THz beam blockage,
and multi-parameter stress).

Research Invariant:
- These scenarios simulate prospective operating conditions; they do NOT establish performance
  on actual 6G radio equipment.
- All scenarios carry explicit research provenance metadata.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List

from src.orchestration.scenarios import (
    SCENARIOS,
    ScenarioConditions,
    SyntheticScenario,
)


# ---------------------------------------------------------------------------
# Generators for 6G-Oriented Operating Conditions
# ---------------------------------------------------------------------------

def _six_g_high_bw_low_rtt_generator(step: int, total_steps: int) -> ScenarioConditions:
    """Extreme throughput (5000 Mbps) and sub-millisecond latency (0.8 ms)."""
    return ScenarioConditions(
        bandwidth_mbps=5000.0,
        rtt_ms=0.8,
        packet_loss=0.0,
        jitter_ms=0.05,
        notes="6G Sub-THz exploratory: extreme bandwidth, sub-millisecond RTT",
    )


def _six_g_high_bw_high_rtt_generator(step: int, total_steps: int) -> ScenarioConditions:
    """Extreme bandwidth (2500 Mbps) with high RTT (120 ms) backhaul/satellite link."""
    return ScenarioConditions(
        bandwidth_mbps=2500.0,
        rtt_ms=120.0,
        packet_loss=0.001,
        jitter_ms=5.0,
        notes="6G Non-Terrestrial Network (NTN) backhaul: high bandwidth with satellite RTT",
    )


def _six_g_bandwidth_degradation_generator(step: int, total_steps: int) -> ScenarioConditions:
    """High bandwidth degrading dynamically from 4000 Mbps down to 25 Mbps."""
    progress = max(0.0, min(1.0, step / max(1, total_steps - 1)))
    bw = 4000.0 - (progress * 3975.0)  # 4000 -> 25 Mbps
    rtt = 1.0 + (progress * 40.0)      # 1.0 -> 41.0 ms
    return ScenarioConditions(
        bandwidth_mbps=max(10.0, bw),
        rtt_ms=rtt,
        packet_loss=0.005 * progress,
        jitter_ms=0.2 + (progress * 8.0),
        notes="6G rapid link degradation: tests proactive migration before channel depletion",
    )


def _six_g_intermittent_connectivity_generator(step: int, total_steps: int) -> ScenarioConditions:
    """High-bandwidth link (2000 Mbps) with periodic beam blockage outages."""
    # Periodic blockage every 6 steps for 2 steps
    in_outage = (step % 6 in (3, 4))
    if in_outage:
        return ScenarioConditions(
            bandwidth_mbps=0.1,
            rtt_ms=250.0,
            packet_loss=1.0,
            jitter_ms=50.0,
            inject_dropout=True,
            notes="6G beam blockage outage",
        )
    return ScenarioConditions(
        bandwidth_mbps=2000.0,
        rtt_ms=1.2,
        packet_loss=0.0001,
        jitter_ms=0.1,
        notes="6G line-of-sight tracking",
    )


def _six_g_rapid_fluctuation_generator(step: int, total_steps: int) -> ScenarioConditions:
    """Fast-fading channel variations testing controller stability."""
    period = 4.0
    sine = (math.sin(2 * math.pi * step / period) + 1.0) / 2.0
    bw = 200.0 + (sine * 1800.0)  # Oscillates between 200 and 2000 Mbps
    rtt = 1.0 + ((1.0 - sine) * 15.0)
    return ScenarioConditions(
        bandwidth_mbps=bw,
        rtt_ms=rtt,
        packet_loss=0.002,
        jitter_ms=2.0,
        notes="6G high-Doppler rapid channel fluctuation",
    )


def _six_g_edge_compute_saturation_generator(step: int, total_steps: int) -> ScenarioConditions:
    """Bandwidth remains high (1500 Mbps), but edge node latency spikes due to compute contention."""
    if step < total_steps // 2:
        return ScenarioConditions(bandwidth_mbps=1500.0, rtt_ms=1.5, packet_loss=0.0, jitter_ms=0.1, notes="edge_unloaded")
    return ScenarioConditions(bandwidth_mbps=1500.0, rtt_ms=65.0, packet_loss=0.01, jitter_ms=20.0, notes="edge_compute_saturated")


def _six_g_kv_cache_growth_network_shifts_generator(step: int, total_steps: int) -> ScenarioConditions:
    """Joint KV-cache growth with stepped bandwidth reduction."""
    step_group = step // 4
    bw = max(50.0, 3000.0 / (1.5 ** step_group))
    rtt = 1.0 + (step_group * 5.0)
    return ScenarioConditions(
        bandwidth_mbps=bw,
        rtt_ms=rtt,
        packet_loss=0.001 * step_group,
        jitter_ms=0.5 * step_group,
        notes="Concurrent KV-cache context expansion and channel capacity decline",
    )


def _six_g_combined_pressure_generator(step: int, total_steps: int) -> ScenarioConditions:
    """Triple pressure: channel degradation + edge compute queueing + KV-cache scaling."""
    progress = max(0.0, min(1.0, step / max(1, total_steps - 1)))
    bw = 3500.0 - (progress * 3400.0)
    rtt = 1.0 + (progress * 80.0)
    loss = min(0.15, progress * 0.1)
    jitter = 0.2 + (progress * 25.0)
    return ScenarioConditions(
        bandwidth_mbps=max(15.0, bw),
        rtt_ms=rtt,
        packet_loss=loss,
        jitter_ms=jitter,
        notes="Combined 6G stress: simultaneous channel, compute, and memory pressure",
    )


# ---------------------------------------------------------------------------
# Registration and Catalog Exposure
# ---------------------------------------------------------------------------

SIX_G_SCENARIOS: Dict[str, SyntheticScenario] = {
    "six_g_high_bandwidth_low_rtt": SyntheticScenario(
        "six_g_high_bandwidth_low_rtt",
        "6G Sub-THz: High bandwidth (5000 Mbps) with sub-ms RTT (0.8 ms)",
        _six_g_high_bw_low_rtt_generator,
    ),
    "six_g_high_bandwidth_high_rtt": SyntheticScenario(
        "six_g_high_bandwidth_high_rtt",
        "6G NTN Satellite: Extreme bandwidth (2500 Mbps) with high latency (120 ms)",
        _six_g_high_bw_high_rtt_generator,
    ),
    "six_g_bandwidth_degradation": SyntheticScenario(
        "six_g_bandwidth_degradation",
        "Dynamic high-bandwidth degradation (4000 Mbps down to 25 Mbps)",
        _six_g_bandwidth_degradation_generator,
    ),
    "six_g_intermittent_connectivity": SyntheticScenario(
        "six_g_intermittent_connectivity",
        "Intermittent beam blockage with complete periodic connection dropouts",
        _six_g_intermittent_connectivity_generator,
    ),
    "six_g_rapid_fluctuation": SyntheticScenario(
        "six_g_rapid_fluctuation",
        "High-Doppler fast-fading rapid channel variations",
        _six_g_rapid_fluctuation_generator,
    ),
    "six_g_edge_compute_saturation": SyntheticScenario(
        "six_g_edge_compute_saturation",
        "High network bandwidth with mid-generation edge compute queue saturation",
        _six_g_edge_compute_saturation_generator,
    ),
    "six_g_kv_cache_growth_network_shifts": SyntheticScenario(
        "six_g_kv_cache_growth_network_shifts",
        "Simultaneous context length scaling and progressive network degradation",
        _six_g_kv_cache_growth_network_shifts_generator,
    ),
    "six_g_combined_pressure": SyntheticScenario(
        "six_g_combined_pressure",
        "Combined 6G stress: joint channel degradation, edge saturation, and memory pressure",
        _six_g_combined_pressure_generator,
    ),
}


def register_six_g_scenarios() -> None:
    """Register all 6G evaluation scenarios into the global ScenarioRegistry."""
    for name, scen in SIX_G_SCENARIOS.items():
        SCENARIOS[name] = scen


# Register automatically on import
register_six_g_scenarios()
