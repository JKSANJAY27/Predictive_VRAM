#!/usr/bin/env python
"""
scripts/analyze_results.py

Module 10 -- Statistical Result Analyzer
Loads trial results from results/raw/, runs validation & invariant checks,
computes aggregated distributions (mean, std, p50, p95, CI), and performs
paired comparisons with effect sizes (Cohen's d, Wilcoxon/t-test).

Usage:
  python scripts/analyze_results.py --raw-dir results/raw --output-dir results/aggregated
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
from src.evaluation.export import CSVExporter, JSONExporter
from src.evaluation.validation import ResultValidator, SanityChecker


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze experimental trial results")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Directory of raw trial JSONs")
    parser.add_argument("--output-dir", type=str, default="results/aggregated", help="Output directory")
    parser.add_argument("--export-csv", action="store_true", default=True, help="Export aggregated CSV table")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  MODULE 10 STATISTICAL EVALUATION & RESULT ANALYZER")
    print("=" * 78)

    # 1. Load all raw trials
    trials = []
    if raw_path.is_dir():
        for p in sorted(raw_path.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and d.get("trial_id"):
                        trials.append(d)
            except Exception as e:
                print(f"Warning: could not load {p}: {e}")

    print(f"Loaded {len(trials)} trial records from {raw_path}")
    if not trials:
        print("No trials found. Run experiments first using scripts/run_experiment.py")
        sys.exit(0)

    # 2. Sanity and Invariant Verification
    print("\nRunning Invariant & Sanity Checks...")
    report = SanityChecker.check_batch(trials)
    print(f"  Total Checks:  {report.total_checks}")
    print(f"  Passed Checks: {report.passed_checks}")
    print(f"  Errors:        {report.error_count}")
    print(f"  Warnings:      {report.warning_count}")

    if not report.is_valid:
        print("\n  [!] Validation Invariant Violations Detected:")
        for issue in report.issues:
            if issue.level == "ERROR":
                print(f"      - {issue.code}: {issue.message}")
    else:
        print("  [+] All experimental invariants verified successfully.")

    # 3. Aggregate by (baseline_id, scenario_id)
    aggregator = ResultAggregator()
    aggregated_list = aggregator.aggregate_trials(trials)

    print("\nAggregated Baseline Performance:")
    headers = ["Baseline", "Scenario", "N", "Mean ITL (ms)", "P95 ITL (ms)", "SLO Vio", "Switches", "Overhead %"]
    widths = [18, 22, 4, 15, 15, 9, 10, 11]

    def fmt_row(cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))

    for agg in aggregated_list:
        m = agg.metrics
        mean_itl = m.get("mean_itl_ms")
        p95_itl = m.get("p95_itl_ms")
        slo_vio = m.get("slo_violation_count")
        switches = m.get("total_switches")
        oh = m.get("control_overhead_fraction")

        mean_str = f"{mean_itl.mean:.1f} +/- {mean_itl.std:.1f}" if mean_itl else "N/A"
        p95_str = f"{p95_itl.mean:.1f} +/- {p95_itl.std:.1f}" if p95_itl else "N/A"
        slo_str = f"{slo_vio.mean:.1f}" if slo_vio else "N/A"
        sw_str = f"{switches.mean:.1f}" if switches else "N/A"
        oh_str = f"{oh.mean * 100.0:.2f}%" if oh else "N/A"

        row = [
            agg.baseline_id,
            agg.scenario_id[:20],
            str(agg.n_trials),
            mean_str,
            p95_str,
            slo_str,
            sw_str,
            oh_str,
        ]
        print(fmt_row(row))

    # Save aggregated JSON and CSV
    agg_json_path = out_path / "aggregated_results.json"
    agg_data = [a.to_dict() for a in aggregated_list]
    JSONExporter.export_aggregated(agg_data, agg_json_path)

    if args.export_csv:
        agg_csv_path = out_path / "aggregated_results.csv"
        CSVExporter.export_aggregated_to_csv(agg_data, agg_csv_path)
        print(f"\nSaved aggregated results to:")
        print(f"  - {agg_json_path}")
        print(f"  - {agg_csv_path}")

    print("=" * 78)


if __name__ == "__main__":
    main()
