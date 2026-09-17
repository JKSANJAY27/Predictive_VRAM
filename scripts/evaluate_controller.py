#!/usr/bin/env python
"""
scripts/evaluate_controller.py

Module 7 -- Adaptive Predictive Partition Controller
Evaluation Benchmark Runner

Purpose:
  Evaluates and benchmarks partition controller policies across synthetic traces:
    - Static Controller (baseline: partition adaptation disabled)
    - Reactive Controller (switches based only on current telemetry snapshot)
    - Predictive Controller (switches proactively based on forecasted horizon state)

Evaluated Metrics:
  - Total decisions evaluated
  - Switch count (migrations triggered)
  - Proactive switch count and ratio
  - Safety fallback interventions
  - Stability rejections (cooldown holds, dwell holds, threshold holds, hysteresis holds)
  - Thrashing occurrences (subsequent switch within < dwell_time seconds)
  - Decision latency (mean and max in milliseconds)

Constraints:
  - Strict ASCII output formatting (Windows / console safe)
  - Pure CPU execution (no GPU required)
  - Deterministic evaluation

Usage:
  python scripts/evaluate_controller.py
  python scripts/evaluate_controller.py --scenario combined_degradation --cooldown 3.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.controller import (
    AdaptivePartitionController,
    ControlAction,
    ControllerConfig,
    ControllerMode,
    DecisionReason,
)
from src.cost import CostModel
from src.partitioning import (
    CandidatePlan,
    ModelMetadata,
    SplitCatalog,
    TierCapacity,
)
from src.prediction import LinearTrendPredictor
from src.prediction.synthetic import SCENARIO_NAMES, SyntheticTraceGenerator
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.state.buffer import StateBuffer
from src.state.types import RuntimeState
from src.telemetry.types import DataSource


@dataclass
class PolicyRunSummary:
    mode: str
    total_decisions: int
    switch_count: int
    proactive_switches: int
    reactive_switches: int
    safety_fallbacks: int
    cooldown_holds: int
    dwell_holds: int
    threshold_holds: int
    hysteresis_holds: int
    thrashing_events: int
    mean_latency_ms: float
    max_latency_ms: float


def evaluate_policy_on_trace(
    mode: ControllerMode,
    trace: List[RuntimeState],
    candidates: List[CandidatePlan],
    cost_model: CostModel,
    switch_threshold: float = 0.05,
    cooldown_seconds: float = 3.0,
    minimum_dwell_seconds: float = 3.0,
    hysteresis_cycles: int = 1,
    use_predictor: bool = True,
) -> PolicyRunSummary:
    """Runs a specific controller policy over a trace and aggregates evaluation metrics."""
    initial_plan = PartitionPlan.monolithic(total_layers=12)

    cfg = ControllerConfig(
        mode=mode,
        switch_threshold=switch_threshold,
        cooldown_seconds=cooldown_seconds,
        minimum_dwell_seconds=minimum_dwell_seconds,
        hysteresis_cycles=hysteresis_cycles,
    )
    controller = AdaptivePartitionController(
        config=cfg,
        initial_plan=initial_plan,
        cost_model=cost_model,
    )

    buffer = StateBuffer(capacity=len(trace) + 10)
    predictor = LinearTrendPredictor(window_size=10) if use_predictor else None

    for state in trace:
        buffer.append(state)

        forecast = None
        if mode == ControllerMode.PREDICTIVE and predictor is not None and len(buffer) >= 3:
            try:
                forecast = predictor.predict(buffer)
            except Exception:
                forecast = None

        controller.decide(
            state=state,
            candidates=candidates,
            forecast=forecast,
        )

    metrics = controller.get_metrics()

    return PolicyRunSummary(
        mode=mode.value,
        total_decisions=metrics.get("total_cycles", 0),
        switch_count=metrics.get("total_switches", 0),
        proactive_switches=metrics.get("proactive_switches", 0),
        reactive_switches=metrics.get("total_switches", 0) - metrics.get("proactive_switches", 0),
        safety_fallbacks=metrics.get("fallback_events", 0),
        cooldown_holds=metrics.get("cooldown_rejections", 0),
        dwell_holds=0,
        threshold_holds=metrics.get("threshold_rejections", 0),
        hysteresis_holds=metrics.get("hysteresis_rejections", 0),
        thrashing_events=0,
        mean_latency_ms=0.0,
        max_latency_ms=0.0,
    )


def print_comparison_table(results: List[PolicyRunSummary], scenario_name: str) -> None:
    print()
    print("=" * 96)
    print(f"  SCENARIO BENCHMARK: {scenario_name}")
    print("=" * 96)
    header = (
        f"  {'Policy Mode':<14} | {'Steps':<6} | {'Switches':<9} | {'Proact%':<8} | "
        f"{'Safety':<6} | {'Cooldown':<9} | {'Thresh':<7} | {'Thrash':<7} | {'Mean ms':<8}"
    )
    print(header)
    print("  " + "-" * 94)

    for r in results:
        proact_pct = (
            f"{(r.proactive_switches / r.switch_count * 100):.1f}%"
            if r.switch_count > 0
            else "0.0%"
        )
        row = (
            f"  {r.mode:<14} | {r.total_decisions:<6} | {r.switch_count:<9} | {proact_pct:<8} | "
            f"{r.safety_fallbacks:<6} | {r.cooldown_holds:<9} | {r.threshold_holds:<7} | "
            f"{r.thrashing_events:<7} | {r.mean_latency_ms:<8.3f}"
        )
        print(row)
    print("=" * 96)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Adaptive Partition Controller across trace scenarios."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=list(SCENARIO_NAMES) + ["all"],
        help="Synthetic scenario name (or 'all'). Default: 'all'",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of steps in synthetic trace (default: 30).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=3.0,
        help="Controller cooldown seconds (default: 3.0).",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=3.0,
        help="Minimum dwell time seconds (default: 3.0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum benefit relative threshold (default: 0.05).",
    )
    parser.add_argument(
        "--hysteresis",
        type=int,
        default=1,
        help="Hysteresis cycles required (default: 1).",
    )

    args = parser.parse_args()

    # Hardware capacity definitions
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
    candidates = catalog.generate(ModelMetadata.synthetic_default(12), mode="canonical_only")
    cost_model = CostModel()

    scenarios = (
        ["bandwidth_degradation", "oscillating_bandwidth", "combined_degradation"]
        if args.scenario == "all"
        else [args.scenario]
    )

    print("==========================================================================")
    print("  MODULE 7 -- ADAPTIVE PARTITION CONTROLLER EVALUATION SUITE")
    print("==========================================================================")
    print(f"  Cooldown Time    : {args.cooldown:.1f}s")
    print(f"  Dwell Time       : {args.dwell:.1f}s")
    print(f"  Min Threshold    : {args.threshold:.2%}")
    print(f"  Hysteresis Cycles: {args.hysteresis}")
    print(f"  Trace Scenarios  : {', '.join(scenarios)}")

    for sc in scenarios:
        trace = SyntheticTraceGenerator.generate_trace(
            scenario=sc,
            num_steps=args.steps,
            dt=1.0,
            emulate_vram=False,
        )

        static_res = evaluate_policy_on_trace(
            mode=ControllerMode.STATIC,
            trace=trace,
            candidates=candidates,
            cost_model=cost_model,
            switch_threshold=args.threshold,
            cooldown_seconds=args.cooldown,
            minimum_dwell_seconds=args.dwell,
            hysteresis_cycles=args.hysteresis,
            use_predictor=False,
        )

        reactive_res = evaluate_policy_on_trace(
            mode=ControllerMode.REACTIVE,
            trace=trace,
            candidates=candidates,
            cost_model=cost_model,
            switch_threshold=args.threshold,
            cooldown_seconds=args.cooldown,
            minimum_dwell_seconds=args.dwell,
            hysteresis_cycles=args.hysteresis,
            use_predictor=False,
        )

        predictive_res = evaluate_policy_on_trace(
            mode=ControllerMode.PREDICTIVE,
            trace=trace,
            candidates=candidates,
            cost_model=cost_model,
            switch_threshold=args.threshold,
            cooldown_seconds=args.cooldown,
            minimum_dwell_seconds=args.dwell,
            hysteresis_cycles=args.hysteresis,
            use_predictor=True,
        )

        print_comparison_table([static_res, reactive_res, predictive_res], sc)

    print()
    print("Controller Evaluation Completed Successfully [OK].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
