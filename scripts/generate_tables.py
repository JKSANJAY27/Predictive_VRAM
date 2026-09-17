#!/usr/bin/env python
"""
scripts/generate_tables.py

Module 10 -- Scientific Tables Generator
Formats aggregated experimental results into Markdown and LaTeX tables for
research papers and documentation.

Usage:
  python scripts/generate_tables.py --raw-dir results/raw --output-dir results/tables
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LaTeX & Markdown tables")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Raw results directory")
    parser.add_argument("--output-dir", type=str, default="results/tables", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_dir)
    tbl_path = Path(args.output_dir)
    tbl_path.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  MODULE 10 SCIENTIFIC TABLES GENERATOR")
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
        print("No trials found. Run experiments first.")
        sys.exit(0)

    aggregator = ResultAggregator()
    aggregated = aggregator.aggregate_trials(trials)

    # 1. Generate Markdown Table
    md_lines = [
        "# Experimental Evaluation: Policy Comparison",
        "",
        "| Baseline Policy | Scenario | N | Mean ITL (ms) | P95 ITL (ms) | SLO Violations | Switches | Proactive | Overhead (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for agg in aggregated:
        m = agg.metrics
        mean_itl = m.get("mean_itl_ms")
        p95_itl = m.get("p95_itl_ms")
        slo_vio = m.get("slo_violation_count")
        sw = m.get("total_switches")
        pro = m.get("proactive_switches")
        oh = m.get("control_overhead_fraction")

        mean_s = f"{mean_itl.mean:.1f} ± {mean_itl.std:.1f}" if mean_itl else "N/A"
        p95_s = f"{p95_itl.mean:.1f} ± {p95_itl.std:.1f}" if p95_itl else "N/A"
        slo_s = f"{slo_vio.mean:.1f}" if slo_vio else "0"
        sw_s = f"{sw.mean:.1f}" if sw else "0"
        pro_s = f"{pro.mean:.1f}" if pro else "0"
        oh_s = f"{oh.mean * 100.0:.2f}%" if oh else "0.0%"

        md_lines.append(
            f"| {agg.baseline_id} | {agg.scenario_id} | {agg.n_trials} | {mean_s} | {p95_s} | {slo_s} | {sw_s} | {pro_s} | {oh_s} |"
        )

    md_content = "\n".join(md_lines) + "\n"
    md_file = tbl_path / "table1_policy_comparison.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. Generate LaTeX Table
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{End-to-end split-inference performance across control policies (mean $\pm$ std).}",
        r"\label{tab:policy_comparison}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"\textbf{Policy} & \textbf{Scenario} & \textbf{N} & \textbf{Mean ITL (ms)} & \textbf{P95 ITL (ms)} & \textbf{SLO Vio.} & \textbf{Switches} & \textbf{Proactive} & \textbf{Overhead (\%)} \\",
        r"\midrule",
    ]

    for agg in aggregated:
        m = agg.metrics
        mean_itl = m.get("mean_itl_ms")
        p95_itl = m.get("p95_itl_ms")
        slo_vio = m.get("slo_violation_count")
        sw = m.get("total_switches")
        pro = m.get("proactive_switches")
        oh = m.get("control_overhead_fraction")

        mean_s = f"{mean_itl.mean:.1f} $\\pm$ {mean_itl.std:.1f}" if mean_itl else "N/A"
        p95_s = f"{p95_itl.mean:.1f} $\\pm$ {p95_itl.std:.1f}" if p95_itl else "N/A"
        slo_s = f"{slo_vio.mean:.1f}" if slo_vio else "0"
        sw_s = f"{sw.mean:.1f}" if sw else "0"
        pro_s = f"{pro.mean:.1f}" if pro else "0"
        oh_s = f"{oh.mean * 100.0:.2f}" if oh else "0.0"

        b_clean = agg.baseline_id.replace("_", r"\_")
        sc_clean = agg.scenario_id.replace("_", r"\_")

        tex_lines.append(
            f"{b_clean} & {sc_clean} & {agg.n_trials} & {mean_s} & {p95_s} & {slo_s} & {sw_s} & {pro_s} & {oh_s}\\% \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    tex_content = "\n".join(tex_lines) + "\n"
    tex_file = tbl_path / "table1_policy_comparison.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"Generated tables:")
    print(f"  - Markdown: {md_file}")
    print(f"  - LaTeX:    {tex_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
