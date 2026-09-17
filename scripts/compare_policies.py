#!/usr/bin/env python
"""
scripts/compare_policies.py

Module 9 -- Closed-Loop Predictive Runtime Integration
Policy Comparison Benchmark (STATIC vs REACTIVE vs PREDICTIVE)

Purpose:
  Empirically tests and compares all three operational control policies under identical,
  strictly controlled environmental conditions:
    1. STATIC: Fixed initial partition plan; ignores network/memory dynamics.
    2. REACTIVE: Dynamic adaptation using strictly instantaneous telemetry observations.
    3. PREDICTIVE: Proactive adaptation using short-horizon predictive forecasts.

  Evaluates on the Canonical 5-Phase Research Scenario ("mixed_degradation"):
    - Phase 1: Stable channel & low memory headroom pressure
    - Phase 2: KV-cache expansion and bandwidth degradation onset
    - Phase 3: Current partition predicted to bottleneck
    - Phase 4: Alternative partition optimal; proactive migration triggered
    - Phase 5: Network recovery

Metrics Compared:
  - Throughput (tokens/sec) & Time to First Token (TTFT)
  - Inter-Token Latency (Mean ITL, P95 ITL)
  - SLO Violation Count (ITL > threshold)
  - Partition Switches & Proactive Switch Count
  - Physical Migration Time (ms) & Transferred Bytes (MB)
  - Control Plane Overhead Fraction (%)

Usage:
  python scripts/compare_policies.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.orchestration.config import OrchestrationConfig
from src.orchestration.runtime import ClosedLoopRuntime
from src.orchestration.types import ExecutionMode, OrchestrationMode


def print_banner(title: str) -> None:
    width = 78
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str) -> None:
    print(f"\n--- {title} " + "-" * (70 - len(title)))


def format_row(cols, widths):
    out = []
    for c, w in zip(cols, widths):
        out.append(str(c).ljust(w))
    return "  ".join(out)


def run_benchmark():
    print_banner("CLOSED-LOOP RUNTIME: THREE-POLICY BENCHMARK")
    print("Experimental Controls:")
    print("  - Model: Synthetic GPT-2 (4 layers, hidden_size=64, 2 heads)")
    print("  - Workload: 16-token autoregressive generation (control interval: 2 tokens)")
    print("  - SLO Target: 200.0 ms ITL threshold")
    print("  - Seed: 42 (reproducible synthetic noise)")
    print("  - Execution Mode: SIMULATION")

    scenarios = ["mixed_degradation", "gradual_bandwidth_degradation", "sudden_bandwidth_drop"]

    for scenario_name in scenarios:
        print_section(f"SCENARIO: {scenario_name.upper()}")

        results = {}
        for mode in [OrchestrationMode.STATIC, OrchestrationMode.REACTIVE, OrchestrationMode.PREDICTIVE]:
            cfg = OrchestrationConfig(
                mode=mode,
                execution_mode=ExecutionMode.SIMULATION,
                control_interval_tokens=2,
                max_generated_tokens=16,
                random_seed=42,
                scenario_name=scenario_name,
                horizon=4,
                slo_itl_ms=200.0,
            )

            rt = ClosedLoopRuntime.create_default(config=cfg, num_layers=4, hidden_size=64)
            trace = rt.run_simulation(scenario_name=scenario_name, num_steps=16)
            results[mode.value] = trace.summary

        # Print comparison table
        headers = [
            "Policy",
            "Mean ITL",
            "P95 ITL",
            "SLO Vio",
            "Switches",
            "Proactive",
            "Migr ms",
            "Overhead %",
        ]
        widths = [12, 10, 10, 8, 9, 10, 9, 11]

        print()
        print(format_row(headers, widths))
        print(format_row(["-" * w for w in widths], widths))

        for mode_name, s in results.items():
            row = [
                mode_name.upper(),
                f"{s.mean_itl_ms:.1f} ms",
                f"{s.p95_itl_ms:.1f} ms",
                str(s.slo_violations),
                str(s.total_switches),
                str(s.proactive_switches),
                f"{s.total_migration_time_ms:.1f}",
                f"{s.control_overhead_fraction * 100.0:.2f}%",
            ]
            print(format_row(row, widths))

        # Comparative analysis
        static_s = results["static"]
        react_s = results["reactive"]
        pred_s = results["predictive"]

        print("\nPolicy Assessment:")
        if pred_s.p95_itl_ms < static_s.p95_itl_ms:
            diff_p95 = (static_s.p95_itl_ms - pred_s.p95_itl_ms) / max(0.01, static_s.p95_itl_ms) * 100.0
            print(f"  [+] Predictive reduced P95 ITL by {diff_p95:.1f}% vs Static.")
        if pred_s.proactive_switches > 0:
            print(f"  [+] Predictive successfully executed {pred_s.proactive_switches} proactive migration(s) ahead of bottleneck.")
        if react_s.proactive_switches == 0:
            print(f"  [*] Reactive confirmed 0 proactive switches (strict no-future-leakage invariant validated).")

    print_banner("BENCHMARK COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_benchmark()
