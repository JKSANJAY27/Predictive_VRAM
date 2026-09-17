#!/usr/bin/env python
"""
scripts/run_controller_demo.py

Module 7 -- Adaptive Predictive Partition Controller
Demonstration Script

Purpose:
  Walks through the control loop functionality of Module 7:
    1. Static mode operation (enforces active plan)
    2. Reactive mode operation (evaluates current state only)
    3. Predictive mode operation (evaluates predicted horizon state)
    4. Anti-thrashing and stability guards (cooldown, dwell time, threshold, hysteresis)
    5. Safety policy hierarchy & emergency fallback override
    6. Multi-step trace replay through changing network/memory conditions
    7. Performance, decision latency, and control metrics reporting

Constraints:
  - Pure ASCII output (compatible with Windows cp1252/consoles)
  - Zero GPU/CUDA dependencies (pure CPU execution)
  - High observability of decision rationale and migration requests

Usage:
  python scripts/run_controller_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.controller import (
    AdaptivePartitionController,
    ControlAction,
    ControlDecision,
    ControllerConfig,
    ControllerMode,
    DecisionReason,
    MigrationRequest,
    SafetyStatus,
)
from src.cost import CostModel
from src.partitioning import (
    CandidatePlan,
    ModelMetadata,
    SplitCatalog,
    TierCapacity,
    generate_plan_id,
)
from src.prediction.types import PredictionResult, TargetForecast
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.state.types import (
    ActivationState,
    ComputeState,
    EnergyState,
    InferenceState,
    MemoryState,
    NetworkState,
    RuntimeState,
)
from src.telemetry.types import DataSource, TaggedValue


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def subheader(title: str) -> None:
    print()
    print(f"--- {title} ---")


def make_state(
    t: float = 1000.0,
    bandwidth_mbps: float = 50.0,
    latency_ms: float = 15.0,
    ram_used_mb: float = 4096.0,
    ram_avail_mb: float = 4096.0,
) -> RuntimeState:
    """Helper to synthesize canonical RuntimeState snapshots."""
    return RuntimeState(
        timestamp=t,
        wall_clock=t,
        step_index=int(t),
        network=NetworkState(
            bandwidth_mbps=TaggedValue(bandwidth_mbps, DataSource.EMULATED, "Mbps"),
            latency_ms=TaggedValue(latency_ms, DataSource.EMULATED, "ms"),
            packet_loss=TaggedValue(0.01, DataSource.EMULATED, "ratio"),
            jitter_ms=TaggedValue(2.0, DataSource.EMULATED, "ms"),
        ),
        memory=MemoryState(
            vram_allocated_mb=TaggedValue(None, DataSource.UNAVAILABLE, "MB"),
            vram_free_mb=TaggedValue(None, DataSource.UNAVAILABLE, "MB"),
            ram_used_mb=TaggedValue(ram_used_mb, DataSource.MEASURED, "MB"),
            ram_available_mb=TaggedValue(ram_avail_mb, DataSource.MEASURED, "MB"),
        ),
        compute=ComputeState(
            gpu_utilization=TaggedValue(None, DataSource.UNAVAILABLE, "%"),
            cpu_utilization=TaggedValue(25.0, DataSource.MEASURED, "%"),
        ),
        inference=InferenceState(
            kv_cache_bytes=TaggedValue(10485760.0, DataSource.MEASURED, "bytes"),
            kv_cache_growth_bytes=TaggedValue(262144.0, DataSource.MEASURED, "bytes"),
            generated_tokens=32,
            context_length=128,
        ),
        activation=ActivationState(
            latest_activation_bytes=TaggedValue(4096.0, DataSource.MEASURED, "bytes"),
            latest_transfer_latency_ms=TaggedValue(1.5, DataSource.MEASURED, "ms"),
        ),
        energy=EnergyState(
            energy_proxy=TaggedValue(12.5, DataSource.ESTIMATED, "proxy_units"),
        ),
    )


def make_forecast(
    t: float = 1000.0,
    bw_trend: float = 10.0,
    lat_trend: float = 80.0,
) -> PredictionResult:
    """Helper to synthesize multi-step PredictionResult."""
    ts = [t + 0.5 * (i + 1) for i in range(4)]
    return PredictionResult(
        prediction_timestamp=t,
        horizon_seconds=2.0,
        step_interval_seconds=0.5,
        forecast_timestamps=ts,
        targets={
            "bandwidth_mbps": TargetForecast(
                target_name="bandwidth_mbps",
                values=[bw_trend] * 4,
                raw_values=[bw_trend] * 4,
                timestamps=ts,
                is_available=True,
                status="ok",
                provenance="emulated",
            ),
            "latency_ms": TargetForecast(
                target_name="latency_ms",
                values=[lat_trend] * 4,
                raw_values=[lat_trend] * 4,
                timestamps=ts,
                is_available=True,
                status="ok",
                provenance="emulated",
            ),
        },
        predictor_name="LinearTrendPredictor",
        input_window_length=4,
        is_valid=True,
        status_message="ok",
    )


def print_decision(d: ControlDecision) -> None:
    action_str = f"[{d.action.value}]"
    print(f"  Cycle {d.controller_cycle:<3} -> Action: {action_str:<18} Reason: {d.reason.value}")
    print(f"    Current:  {d.current_plan_id}")
    print(f"    Selected: {d.selected_plan_id}")
    print(f"    Costs:    Current={d.current_cost:.4f}  Selected={d.selected_cost:.4f}  Gain={d.expected_gain:.4f}")
    print(f"    Safety:   {d.safety_status.value:<10} Override={d.is_safety_override}  Proactive={d.is_proactive}")
    if d.migration_request:
        req = d.migration_request
        print(f"    Migration Request:")
        print(f"      Source:  {req.source_plan_id} -> Target: {req.target_plan_id}")
        print(f"      Layers:  {req.changed_layers} | Tiers: {req.affected_tiers}")
        print(f"      KV Est:  {req.estimated_kv_transfer_bytes} bytes")


def main() -> int:
    banner("MODULE 7 -- ADAPTIVE PREDICTIVE PARTITION CONTROLLER DEMO")

    # 1. Setup Infrastructure
    subheader("1. System Setup & Catalog Initialization")
    meta = ModelMetadata.synthetic_default(total_layers=12)
    caps = {
        TierId.USER_DEVICE: TierCapacity(
            tier_id=TierId.USER_DEVICE,
            device="cpu",
            available_memory_mb=2048.0,
            memory_provenance=DataSource.EMULATED,
        ),
        TierId.EDGE_A: TierCapacity(
            tier_id=TierId.EDGE_A,
            device="cpu",
            available_memory_mb=8192.0,
            memory_provenance=DataSource.EMULATED,
        ),
        TierId.EDGE_B: TierCapacity(
            tier_id=TierId.EDGE_B,
            device="cpu",
            available_memory_mb=16384.0,
            memory_provenance=DataSource.EMULATED,
        ),
    }

    catalog = SplitCatalog(tier_capacities=caps)
    candidates = catalog.generate(meta, mode="exhaustive")
    print(f"  Model: 12-layer GPT-2 Synthetic (124M params)")
    print(f"  Generated {len(candidates)} partition candidates across tiers.")

    cost_model = CostModel()
    initial_plan = PartitionPlan.monolithic(total_layers=12)

    # 2. Static Mode Demo
    subheader("2. Static Mode Execution (Partition Adaptation Disabled)")
    cfg_static = ControllerConfig(mode=ControllerMode.STATIC)
    ctrl_static = AdaptivePartitionController(
        config=cfg_static,
        initial_plan=initial_plan,
        cost_model=cost_model,
    )
    s_normal = make_state(t=1000.0, bandwidth_mbps=50.0)
    d_static = ctrl_static.decide(state=s_normal, candidates=candidates)
    print_decision(d_static)
    assert d_static.action == ControlAction.KEEP_CURRENT
    assert d_static.reason == DecisionReason.STATIC_POLICY

    # 3. Reactive Mode Demo
    subheader("3. Reactive Mode Execution (Zero Lookahead)")
    cfg_reactive = ControllerConfig(
        mode=ControllerMode.REACTIVE,
        switch_threshold=0.0,
        minimum_dwell_seconds=0.0,
        cooldown_seconds=0.0,
        hysteresis_cycles=1,
    )
    ctrl_reactive = AdaptivePartitionController(
        config=cfg_reactive,
        initial_plan=initial_plan,
        cost_model=cost_model,
    )
    # High bandwidth environment favors offloading to edge
    s_fast = make_state(t=1001.0, bandwidth_mbps=100.0, latency_ms=5.0)
    d_reactive = ctrl_reactive.decide(state=s_fast, candidates=candidates)
    print_decision(d_reactive)

    # 4. Predictive Mode Demo
    subheader("4. Predictive Mode Execution (Lookahead Forecast Integration)")
    cfg_pred = ControllerConfig(
        mode=ControllerMode.PREDICTIVE,
        switch_threshold=0.0,
        minimum_dwell_seconds=0.0,
        cooldown_seconds=0.0,
        hysteresis_cycles=1,
    )
    ctrl_pred = AdaptivePartitionController(
        config=cfg_pred,
        initial_plan=initial_plan,
        cost_model=cost_model,
    )
    # Current bandwidth is healthy (50 Mbps), but forecast anticipates severe drop (5 Mbps)
    s_ok = make_state(t=1002.0, bandwidth_mbps=50.0, latency_ms=15.0)
    fc_drop = make_forecast(t=1002.0, bw_trend=5.0, lat_trend=120.0)
    d_pred = ctrl_pred.decide(state=s_ok, candidates=candidates, forecast=fc_drop)
    print("  Current network is 50 Mbps, but forecast predicts 5 Mbps collapse:")
    print_decision(d_pred)
    print(f"  Prediction used: {d_pred.prediction_used} | Proactive: {d_pred.is_proactive}")

    # 5. Stability & Anti-Thrashing Guardrails
    subheader("5. Stability & Anti-Thrashing Guardrails")

    # A: Threshold Rejection
    print("  [Guard A] Minimum Benefit Threshold Enforcement:")
    cfg_thresh = ControllerConfig(
        mode=ControllerMode.REACTIVE,
        switch_threshold=0.99,  # Unreachable 99% improvement required
        minimum_dwell_seconds=0.0,
        cooldown_seconds=0.0,
        hysteresis_cycles=1,
    )
    ctrl_thresh = AdaptivePartitionController(
        config=cfg_thresh,
        initial_plan=initial_plan,
        cost_model=cost_model,
    )
    d_thresh = ctrl_thresh.decide(state=s_fast, candidates=candidates)
    print(f"    Required gain: 99%  Observed gain: {d_thresh.expected_gain:.4f}")
    print(f"    Action: [{d_thresh.action.value}] Reason: {d_thresh.reason.value}")
    print(f"    Explanation: {d_thresh.explanation}")

    # B: Cooldown / Dwell Suppression
    print("\n  [Guard B] Cooldown & Dwell Time Guard:")
    cfg_dwell = ControllerConfig(
        mode=ControllerMode.REACTIVE,
        switch_threshold=0.0,
        minimum_dwell_seconds=10.0,
        cooldown_seconds=10.0,
        hysteresis_cycles=1,
    )
    ctrl_dwell = AdaptivePartitionController(
        config=cfg_dwell,
        initial_plan=initial_plan,
        cost_model=cost_model,
    )
    # First cycle switches
    d1 = ctrl_dwell.decide(state=make_state(t=1000.0, bandwidth_mbps=100.0), candidates=candidates)
    # Second cycle immediately after (t=1002.0 < 1000.0 + 10.0)
    d2 = ctrl_dwell.decide(state=make_state(t=1002.0, bandwidth_mbps=10.0), candidates=candidates)
    print(f"    Cycle 1 at t=1000.0 -> Action: [{d1.action.value}]")
    print(f"    Cycle 2 at t=1002.0 (dwell active) -> Action: [{d2.action.value}] Reason: {d2.reason.value}")
    print(f"    Explanation: {d2.explanation}")

    # C: Hysteresis Confirmation
    print("\n  [Guard C] Hysteresis Confirmation Cycles (Requires 2 consecutive votes):")
    cfg_hyst = ControllerConfig(
        mode=ControllerMode.REACTIVE,
        switch_threshold=0.0,
        minimum_dwell_seconds=0.0,
        cooldown_seconds=0.0,
        hysteresis_cycles=2,
    )
    ctrl_hyst = AdaptivePartitionController(
        config=cfg_hyst,
        initial_plan=initial_plan,
        cost_model=cost_model,
    )
    # Cycle 1: first vote -> held by hysteresis
    dh1 = ctrl_hyst.decide(state=make_state(t=1000.0, bandwidth_mbps=100.0), candidates=candidates)
    # Cycle 2: confirmed second vote -> approved
    dh2 = ctrl_hyst.decide(state=make_state(t=1001.0, bandwidth_mbps=100.0), candidates=candidates)
    print(f"    Cycle 1 (Vote 1 of 2) -> Action: [{dh1.action.value}] Reason: {dh1.reason.value}")
    print(f"    Cycle 2 (Vote 2 of 2) -> Action: [{dh2.action.value}] Reason: {dh2.reason.value}")

    # 6. Safety Emergency Override
    subheader("6. Safety Hierarchy & Emergency Override")
    cfg_safe = ControllerConfig(
        mode=ControllerMode.REACTIVE,
        switch_threshold=0.50,  # High threshold
        minimum_dwell_seconds=60.0,  # Strict dwell
        cooldown_seconds=60.0,
        emergency_override=True,
        max_memory_pressure=0.90,
    )
    # Start on a partitioned plan
    two_tier_plan = candidates[10].partition_plan
    ctrl_safe = AdaptivePartitionController(
        config=cfg_safe,
        initial_plan=two_tier_plan,
        cost_model=cost_model,
    )
    # Simulate acute memory exhaustion (OOM danger on active plan)
    s_emerg = make_state(
        t=1005.0,
        ram_used_mb=7800.0,
        ram_avail_mb=200.0,  # Extreme memory exhaustion
    )
    d_emerg = ctrl_safe.decide(state=s_emerg, candidates=candidates)
    print("  Severe resource pressure triggers emergency intervention:")
    print_decision(d_emerg)
    print(f"  Emergency status: {d_emerg.safety_status.value} | Override: {d_emerg.is_safety_override}")

    # 7. 5-Phase Trace Replay
    subheader("7. Multi-Step Trace Replay Walkthrough")
    trace_events = [
        ("Phase 1: Healthy Network", 1000.0, 60.0, 12.0),
        ("Phase 2: Network Degrades", 1001.0, 20.0, 45.0),
        ("Phase 3: Severe Congestion", 1002.0, 4.0, 150.0),
        ("Phase 4: Cooldown Holding", 1003.0, 5.0, 140.0),
        ("Phase 5: Network Recovers", 1008.0, 80.0, 10.0),
    ]

    ctrl_sim = AdaptivePartitionController(
        config=ControllerConfig(
            mode=ControllerMode.PREDICTIVE,
            switch_threshold=0.05,
            minimum_dwell_seconds=4.0,
            cooldown_seconds=4.0,
            hysteresis_cycles=1,
        ),
        initial_plan=initial_plan,
        cost_model=cost_model,
    )

    print(f"  {'Step':<6} {'Phase':<28} {'Action':<16} {'Current Plan':<22} {'Selected Plan':<22} {'Reason':<26}")
    print("  " + "-" * 122)

    for i, (name, t_val, bw, lat) in enumerate(trace_events):
        st = make_state(t=t_val, bandwidth_mbps=bw, latency_ms=lat)
        fc = make_forecast(t=t_val, bw_trend=bw, lat_trend=lat)
        dec = ctrl_sim.decide(state=st, candidates=candidates, forecast=fc)
        print(f"  {i+1:<6} {name:<28} {dec.action.value:<16} {dec.current_plan_id:<22} {dec.selected_plan_id:<22} {dec.reason.value:<26}")

    # 8. Controller Metrics Reporting
    subheader("8. Controller Performance Metrics")
    metrics = ctrl_sim.get_metrics()
    print("  Aggregated Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<32}: {v:.4f}")
        else:
            print(f"    {k:<32}: {v}")

    banner("MODULE 7 CONTROLLER DEMO COMPLETED SUCCESSFULLY [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
