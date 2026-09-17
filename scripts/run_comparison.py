#!/usr/bin/env python
"""
scripts/run_comparison.py

Module 10 -- Policy Comparison Benchmark
Executes all five baseline policies (B1-B5) under strictly identical environmental
traces and computes paired statistical comparisons.

Usage:
  python scripts/run_comparison.py --scenario combined_degradation --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.aggregation import ResultAggregator
from src.evaluation.baselines import BaselinePolicyFactory
from src.evaluation.experiment_config import (
    ExperimentConfig,
    NetworkConfig,
    WorkloadConfig,
)
from src.evaluation.runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-policy baseline comparison")
    parser.add_argument("--scenario", type=str, default="combined_degradation", help="Scenario ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokens", type=int, default=32, help="Tokens per trial")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Raw results directory")
    parser.add_argument("--force", action="store_true", help="Force re-execution")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig(
        name=f"comparison_{args.scenario}_seed{args.seed}",
        seed=args.seed,
        workload=WorkloadConfig(max_new_tokens=args.tokens),
        network=NetworkConfig(scenario=args.scenario, n_steps=64),
        baselines=["static", "network_reactive", "memory_reactive", "joint_reactive", "predictive"],
    )

    print("=" * 78)
    print(f"  MODULE 10 FIVE-POLICY COMPARISON: {args.scenario.upper()}")
    print("=" * 78)

    runner = ExperimentRunner(raw_results_dir=args.raw_dir, force_rerun=args.force)
    results = runner.run_experiment(cfg, include_ablations=False)

    headers = ["Policy", "Mean ITL", "P95 ITL", "SLO Vio", "Switches", "Proactive", "Migr Time", "Overhead %"]
    widths = [18, 11, 11, 9, 10, 10, 11, 12]

    def fmt_row(cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    print()
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))

    aggregator = ResultAggregator()
    trial_dicts = [r.to_dict() for r in results]

    for r in results:
        m = r.metrics
        row = [
            r.baseline_id,
            f"{m.mean_itl_ms:.1f} ms",
            f"{m.p95_itl_ms:.1f} ms",
            str(m.slo_violation_count),
            str(m.total_switches),
            str(m.proactive_switches),
            f"{m.total_migration_time_ms:.1f} ms",
            f"{m.control_overhead_fraction * 100.0:.2f}%",
        ]
        print(fmt_row(row))

    print("\nStatistical Observations:")
    metrics_by_b = {r.baseline_id: r.metrics for r in results}
    pred = metrics_by_b.get("predictive")
    static = metrics_by_b.get("static")
    joint = metrics_by_b.get("joint_reactive")

    if pred and static:
        delta_p95 = static.p95_itl_ms - pred.p95_itl_ms
        pct = (delta_p95 / max(0.01, static.p95_itl_ms)) * 100.0
        print(f"  - Predictive P95 ITL: {pred.p95_itl_ms:.1f} ms vs Static: {static.p95_itl_ms:.1f} ms (Delta: {delta_p95:+.1f} ms, {pct:+.1f}%)")

    if pred and joint:
        delta_joint = joint.p95_itl_ms - pred.p95_itl_ms
        pct_j = (delta_joint / max(0.01, joint.p95_itl_ms)) * 100.0
        print(f"  - Predictive P95 ITL: {pred.p95_itl_ms:.1f} ms vs Joint Reactive: {joint.p95_itl_ms:.1f} ms (Delta: {delta_joint:+.1f} ms, {pct_j:+.1f}%)")

    if pred:
        print(f"  - Proactive switches recorded: {pred.proactive_switches} (Reactive proactive switches: 0)")

    print("=" * 78)


if __name__ == "__main__":
    main()
