"""
Synthetic integrated scenarios for closed-loop runtime evaluation (Module 9).

Implements 10 deterministic scenarios:
1. stable: High bandwidth, constant low memory
2. kv_cache_growth: High bandwidth, rising KV-cache memory pressure
3. gradual_bandwidth_degradation: Smooth ramp-down from 100 to 5 Mbps
4. sudden_bandwidth_drop: Step function drop from 100 to 2 Mbps
5. bandwidth_oscillation: Square-wave bandwidth channel
6. edge_compute_saturation: Spike in edge node latency/load
7. mixed_degradation (Canonical Research Scenario):
   - Phase 1: Stable channel & flat memory
   - Phase 2: KV cache grows & bandwidth degrades
   - Phase 3: Current partition predicted to become bottlenecked
   - Phase 4: Alternative partition becomes preferred
   - Phase 5: Conditions recover
8. recovery: Degraded channel recovers back to optimal
9. telemetry_dropout: Intermittent missing measurements (DataSource.UNAVAILABLE)
10. prediction_error: Channel behavior intentionally diverges from linear trend
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass
class ScenarioConditions:
    """Network and environmental conditions for a single discrete step."""
    bandwidth_mbps: float
    rtt_ms: float
    packet_loss: float
    jitter_ms: float
    inject_dropout: bool = False
    notes: str = ""


class SyntheticScenario:
    """Base definition of a repeatable synthetic scenario."""

    def __init__(
        self,
        name: str,
        description: str,
        generator_fn: Callable[[int, int], ScenarioConditions],
    ) -> None:
        self.name = name
        self.description = description
        self.generator_fn = generator_fn

    def get_conditions(self, step: int, total_steps: int = 32) -> ScenarioConditions:
        return self.generator_fn(step, total_steps)


def _stable_generator(step: int, total_steps: int) -> ScenarioConditions:
    return ScenarioConditions(bandwidth_mbps=100.0, rtt_ms=5.0, packet_loss=0.0, jitter_ms=0.5, notes="stable")


def _kv_growth_generator(step: int, total_steps: int) -> ScenarioConditions:
    return ScenarioConditions(bandwidth_mbps=100.0, rtt_ms=5.0, packet_loss=0.0, jitter_ms=0.5, notes="kv_growth")


def _gradual_bw_generator(step: int, total_steps: int) -> ScenarioConditions:
    progress = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    bw = 100.0 - progress * 95.0  # 100 -> 5 Mbps
    rtt = 5.0 + progress * 40.0   # 5 -> 45 ms
    return ScenarioConditions(bandwidth_mbps=round(bw, 2), rtt_ms=round(rtt, 2), packet_loss=0.0, jitter_ms=1.0)


def _sudden_drop_generator(step: int, total_steps: int) -> ScenarioConditions:
    midpoint = total_steps // 3
    if step < midpoint:
        return ScenarioConditions(bandwidth_mbps=100.0, rtt_ms=5.0, packet_loss=0.0, jitter_ms=0.5)
    else:
        return ScenarioConditions(bandwidth_mbps=2.0, rtt_ms=80.0, packet_loss=0.05, jitter_ms=10.0, notes="sudden_drop")


def _oscillation_generator(step: int, total_steps: int) -> ScenarioConditions:
    # 4-step period square wave
    high = (step // 4) % 2 == 0
    bw = 100.0 if high else 5.0
    rtt = 5.0 if high else 50.0
    return ScenarioConditions(bandwidth_mbps=bw, rtt_ms=rtt, packet_loss=0.0, jitter_ms=2.0)


def _edge_saturation_generator(step: int, total_steps: int) -> ScenarioConditions:
    saturated = step >= (total_steps // 3)
    rtt = 100.0 if saturated else 5.0
    loss = 0.08 if saturated else 0.0
    return ScenarioConditions(bandwidth_mbps=50.0, rtt_ms=rtt, packet_loss=loss, jitter_ms=5.0)


def _mixed_degradation_generator(step: int, total_steps: int) -> ScenarioConditions:
    """
    Canonical 5-Phase Research Scenario:
    Phase 1 (0 to 20%): Stable baseline (100 Mbps, 5ms)
    Phase 2 (20% to 45%): KV cache grows & bandwidth declines (100 -> 25 Mbps)
    Phase 3 (45% to 65%): Predicted threshold breach; bandwidth drops to 5 Mbps
    Phase 4 (65% to 85%): Bottleneck active; alternative partition preferred
    Phase 5 (85% to 100%): Channel recovery (recovers to 80 Mbps)
    """
    p = step / max(1, total_steps - 1)
    if p < 0.20:
        # Phase 1: Stable
        return ScenarioConditions(bandwidth_mbps=100.0, rtt_ms=5.0, packet_loss=0.0, jitter_ms=0.5, notes="phase_1_stable")
    elif p < 0.45:
        # Phase 2: KV growth + mild decline
        frac = (p - 0.20) / 0.25
        bw = 100.0 - frac * 75.0  # 100 -> 25 Mbps
        rtt = 5.0 + frac * 20.0
        return ScenarioConditions(bandwidth_mbps=round(bw, 2), rtt_ms=round(rtt, 2), packet_loss=0.01, jitter_ms=1.5, notes="phase_2_decline")
    elif p < 0.65:
        # Phase 3: Severe predicted constraint
        frac = (p - 0.45) / 0.20
        bw = 25.0 - frac * 20.0   # 25 -> 5 Mbps
        rtt = 25.0 + frac * 35.0
        return ScenarioConditions(bandwidth_mbps=round(bw, 2), rtt_ms=round(rtt, 2), packet_loss=0.04, jitter_ms=3.0, notes="phase_3_predicted_risk")
    elif p < 0.85:
        # Phase 4: Bottleneck sustained
        return ScenarioConditions(bandwidth_mbps=5.0, rtt_ms=60.0, packet_loss=0.05, jitter_ms=4.0, notes="phase_4_bottleneck")
    else:
        # Phase 5: Recovery
        frac = (p - 0.85) / 0.15
        bw = 5.0 + frac * 75.0    # 5 -> 80 Mbps
        rtt = 60.0 - frac * 50.0
        return ScenarioConditions(bandwidth_mbps=round(bw, 2), rtt_ms=round(rtt, 2), packet_loss=0.0, jitter_ms=1.0, notes="phase_5_recovery")


def _recovery_generator(step: int, total_steps: int) -> ScenarioConditions:
    # Starts bad, recovers halfway
    if step < total_steps // 2:
        return ScenarioConditions(bandwidth_mbps=5.0, rtt_ms=60.0, packet_loss=0.04, jitter_ms=3.0, notes="degraded")
    return ScenarioConditions(bandwidth_mbps=100.0, rtt_ms=5.0, packet_loss=0.0, jitter_ms=0.5, notes="recovered")


def _telemetry_dropout_generator(step: int, total_steps: int) -> ScenarioConditions:
    # Dropout during middle 25% of run
    mid = total_steps // 2
    dropout = (mid - 2 <= step <= mid + 2)
    return ScenarioConditions(bandwidth_mbps=80.0, rtt_ms=8.0, packet_loss=0.0, jitter_ms=1.0, inject_dropout=dropout, notes="dropout" if dropout else "ok")


def _prediction_error_generator(step: int, total_steps: int) -> ScenarioConditions:
    # Sudden sharp non-linear turn that challenges linear predictors
    if step < total_steps // 2:
        bw = 80.0 - step * 2.0  # Appears to be declining gradually
    else:
        bw = 10.0 if (step % 2 == 0) else 90.0  # Wild chaotic swings
    return ScenarioConditions(bandwidth_mbps=bw, rtt_ms=15.0, packet_loss=0.01, jitter_ms=2.0, notes="prediction_challenge")


SCENARIOS: Dict[str, SyntheticScenario] = {
    "stable": SyntheticScenario("stable", "Constant high bandwidth, low latency, flat memory", _stable_generator),
    "kv_cache_growth": SyntheticScenario("kv_cache_growth", "High bandwidth with linear KV-cache growth", _kv_growth_generator),
    "gradual_bandwidth_degradation": SyntheticScenario("gradual_bandwidth_degradation", "Smooth monotonic ramp-down from 100 to 5 Mbps", _gradual_bw_generator),
    "sudden_bandwidth_drop": SyntheticScenario("sudden_bandwidth_drop", "Step function drop from 100 to 2 Mbps", _sudden_drop_generator),
    "bandwidth_oscillation": SyntheticScenario("bandwidth_oscillation", "Square-wave channel fluctuation", _oscillation_generator),
    "edge_compute_saturation": SyntheticScenario("edge_compute_saturation", "Edge node CPU and queue saturation", _edge_saturation_generator),
    "mixed_degradation": SyntheticScenario("mixed_degradation", "Canonical 5-Phase research scenario", _mixed_degradation_generator),
    "recovery": SyntheticScenario("recovery", "Degraded network transitions to full recovery", _recovery_generator),
    "telemetry_dropout": SyntheticScenario("telemetry_dropout", "Intermittent metric unavailability", _telemetry_dropout_generator),
    "prediction_error": SyntheticScenario("prediction_error", "Non-linear channel behavior testing predictor resilience", _prediction_error_generator),
}


class ScenarioRegistry:
    """Lookup and factory for synthetic closed-loop scenarios."""

    @classmethod
    def get(cls, name: str) -> SyntheticScenario:
        if name not in SCENARIOS:
            raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
        return SCENARIOS[name]

    @classmethod
    def list_all(cls) -> Dict[str, str]:
        return {k: v.description for k, v in SCENARIOS.items()}
