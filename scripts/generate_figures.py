#!/usr/bin/env python
"""
scripts/generate_figures.py

Module 10 -- Scientific Publication Figures Generator
Produces high-resolution figures from trial results and aggregated statistics.
Gracefully handles environments without matplotlib.

Usage:
  python scripts/generate_figures.py --raw-dir results/raw --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.aggregation import ResultAggregator
from src.evaluation.visualization import (
    plot_ablation_waterfall,
    plot_baseline_comparison_bar,
    plot_itl_timeline,
    plot_migration_time_cdf,
    plot_p95_vs_overhead_scatter,
    plot_partition_state_strip,
    plot_slo_violation_rate,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Directory of raw trial JSONs")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Output directory for figures")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_dir)
    fig_path = Path(args.output_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  MODULE 10 SCIENTIFIC FIGURE GENERATOR")
    print("=" * 78)

    trials = []
    if raw_path.is_dir():
        for p in sorted(raw_path.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and d.get("trial_id"):
                        trials.append(d)
            except Exception:
                continue

    if not trials:
        print("No trials found to plot. Run experiments first.")
        sys.exit(0)

    aggregator = ResultAggregator()
    aggregated = [a.to_dict() for a in aggregator.aggregate_trials(trials)]

    # Generate figures
    p1 = plot_itl_timeline(trials, fig_path / "fig1_itl_timeline.png")
    p2 = plot_baseline_comparison_bar(aggregated, "p95_itl_ms", fig_path / "fig2_p95_itl_comparison.png", ylabel="P95 ITL (ms)")
    p3 = plot_baseline_comparison_bar(aggregated, "total_switches", fig_path / "fig3_switches_comparison.png", ylabel="Partition Switches")
    p4 = plot_p95_vs_overhead_scatter(aggregated, fig_path / "fig4_pareto_overhead_latency.png")
    p5 = plot_migration_time_cdf(trials, fig_path / "fig5_migration_cdf.png")
    p6 = plot_slo_violation_rate(aggregated, fig_path / "fig6_slo_violation_rate.png")

    if trials:
        plot_partition_state_strip(trials[0], fig_path / "fig7_partition_strip.png")

    generated = [p for p in [p1, p2, p3, p4, p5, p6] if p is not None]
    if generated:
        print(f"Generated {len(generated)} publication figures in: {fig_path}")
    else:
        print("Matplotlib is not installed or headless; figure generation skipped.")
    print("=" * 78)


if __name__ == "__main__":
    main()
