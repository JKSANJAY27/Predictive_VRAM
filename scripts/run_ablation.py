#!/usr/bin/env python
"""
scripts/run_ablation.py

Module 10 -- Ablation Suite Runner
Executes the ablation matrix (A1-A4 mandatory, optionally A5-A8) under identical
environmental conditions to isolate individual component contributions.

Usage:
  python scripts/run_ablation.py --scenario combined_degradation --seed 42
  python scripts/run_ablation.py --all-ablations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ablations import AblationSuite
from src.evaluation.experiment_config import (
    ExperimentConfig,
    NetworkConfig,
    WorkloadConfig,
)
from src.evaluation.runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run component ablations")
    parser.add_argument("--scenario", type=str, default="combined_degradation", help="Scenario ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokens", type=int, default=32, help="Tokens per trial")
    parser.add_argument("--all-ablations", action="store_true", help="Run A1-A8 (default: A1-A4 mandatory)")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Raw results directory")
    parser.add_argument("--force", action="store_true", help="Force re-execution")
    return parser.parse_args()


def main():
    args = parse_args()

    abl_list = (
        ["prediction_removed", "switching_penalty_removed", "network_signal_removed", "vram_signal_removed",
         "hysteresis_disabled", "cooldown_disabled", "threshold_zero"]
        if args.all_ablations
        else ["prediction_removed", "switching_penalty_removed", "network_signal_removed", "vram_signal_removed"]
    )

    cfg = ExperimentConfig(
        name=f"ablation_{args.scenario}_seed{args.seed}",
        seed=args.seed,
        workload=WorkloadConfig(max_new_tokens=args.tokens),
        network=NetworkConfig(scenario=args.scenario, n_steps=64),
        baselines=["predictive"],
        ablations=abl_list,
    )

    print("=" * 78)
    print(f"  MODULE 10 ABLATION SUITE: {args.scenario.upper()}")
    print("=" * 78)
    print(f"Ablations to evaluate: {', '.join(abl_list)}")
    print("-" * 78)

    runner = ExperimentRunner(raw_results_dir=args.raw_dir, force_rerun=args.force)
    results = runner.run_experiment(cfg, include_ablations=True)

    # First find baseline predictive result
    base_res = [r for r in results if not r.ablation_id]
    base_m = base_res[0].metrics if base_res else None

    headers = ["Ablation ID", "Component Changed", "P95 ITL", "Delta vs Base", "Switches", "SLO Vio"]
    widths = [14, 26, 12, 16, 10, 10]

    def fmt_row(cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    print()
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))

    if base_m:
        print(fmt_row(["Base (Full)", "None (Predictive)", f"{base_m.p95_itl_ms:.1f} ms", "0.0 ms (0.0%)", str(base_m.total_switches), str(base_m.slo_violation_count)]))

    for r in results:
        if not r.ablation_id:
            continue
        m = r.metrics
        spec = AblationSuite.get(r.ablation_id)
        comp_name = spec.changed_component if spec else r.ablation_id
        if base_m:
            delta = m.p95_itl_ms - base_m.p95_itl_ms
            pct = (delta / max(0.01, base_m.p95_itl_ms)) * 100.0
            delta_str = f"{delta:+.1f} ms ({pct:+.1f}%)"
        else:
            delta_str = "N/A"

        row = [
            r.ablation_id,
            comp_name[:24],
            f"{m.p95_itl_ms:.1f} ms",
            delta_str,
            str(m.total_switches),
            str(m.slo_violation_count),
        ]
        print(fmt_row(row))

    print("=" * 78)


if __name__ == "__main__":
    main()
